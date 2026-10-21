# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for chunked LLM validation.

Conflict grouping and prompt rendering are tested directly. Sync and async
dispatch use lifecycle rows and fake model facades to test request binding,
pool failover, concurrency, and chunk-size parity.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import logging
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from data_designer.engine.models.clients.errors import SyncClientUnavailableError

from anonymizer.engine.constants import (
    COL_ADMISSION_RESULTS,
    COL_MERGED_TAGGED_TEXT,
    COL_SEED_DETECTION_RESULT,
    COL_TAG_NOTATION,
    COL_TEXT,
    COL_VALIDATION_CANDIDATES,
    COL_VALIDATION_DECISIONS,
    COL_VALIDATION_INVENTORIES,
    COL_VALIDATION_SKELETON,
)
from anonymizer.engine.detection.chunked_validation import (
    ChunkedValidationParams,
    _dispatch_chunk,
    _dispatch_chunk_async,
    build_validation_conflict_groups,
    chunk_validation_conflict_groups,
    chunked_validate_row,
    chunked_validate_row_async,
    make_chunked_validation_generator,
    prepare_validation_requests,
    render_chunk_prompt,
)
from anonymizer.engine.detection.custom_columns import admit_validation_decisions
from anonymizer.engine.detection.postprocess import TagNotation
from anonymizer.engine.schemas import (
    AdmissionResults,
    AdmissionRoute,
    AdmissionStatus,
    DetectionResult,
    EffectiveLabelPolicy,
    Observation,
    RawValidationDecisionSchema,
    SourceTextRef,
    SourceView,
    ValidationChoice,
    ValidationRequestDecisionSets,
    ValidationRequestInventory,
)
from anonymizer.engine.workflow_columns.detection.config import ChunkedValidationConfig
from anonymizer.engine.workflow_columns.detection.impl import ChunkedValidationGenerator

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class FakeFacade:
    """Test double for ``ModelFacade`` recording invocations and replaying canned responses.

    Exposes both ``generate()`` and ``agenerate()`` so the plugin's synchronous
    and asynchronous paths can share assertions.

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
        max_parallel_requests: int = 4,
    ) -> None:
        self.alias = alias
        self._response = response
        self._raise = raise_on_call
        self.max_parallel_requests = max_parallel_requests
        self.calls: list[dict[str, Any]] = []

    def generate(self, *, prompt, parser, system_prompt=None, purpose=None, **kwargs):
        return self._call(prompt=prompt, parser=parser, system_prompt=system_prompt, purpose=purpose, kwargs=kwargs)

    async def agenerate(self, *, prompt, parser, system_prompt=None, purpose=None, **kwargs):
        return self._call(prompt=prompt, parser=parser, system_prompt=system_prompt, purpose=purpose, kwargs=kwargs)

    def _call(self, *, prompt, parser, system_prompt, purpose, kwargs):
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
            response = response(prompt)  # ty: ignore[call-top-callable]
        raw = response if isinstance(response, str) else f"```json\n{json.dumps(response)}\n```"
        return parser(raw), []


class AsyncTrackingFacade(FakeFacade):
    """Async-only facade that tracks concurrent ``agenerate`` calls."""

    def __init__(
        self,
        alias: str,
        response: dict | str | Callable[[str], dict | str] | None = None,
        *,
        raise_on_call: bool = False,
    ) -> None:
        super().__init__(alias, response, raise_on_call=raise_on_call)
        self.active_calls = 0
        self.max_active_calls = 0

    def generate(self, *, prompt, parser, system_prompt=None, purpose=None, **kwargs):
        raise AssertionError("sync generate should not be used by async chunk validation")

    async def agenerate(self, *, prompt, parser, system_prompt=None, purpose=None, **kwargs):
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            await asyncio.sleep(0.01)
            return self._call(prompt=prompt, parser=parser, system_prompt=system_prompt, purpose=purpose, kwargs=kwargs)
        finally:
            self.active_calls -= 1


class SyncUnavailableFacade(FakeFacade):
    def __init__(
        self,
        alias: str,
        response: dict | str | Callable[[str], dict | str] | None = None,
        *,
        max_parallel_requests: int = 1,
    ) -> None:
        super().__init__(alias, response, max_parallel_requests=max_parallel_requests)
        self.request_timeout = 1.0
        self.sync_calls = 0
        self.async_calls = 0

    def generate(self, *, prompt, parser, system_prompt=None, purpose=None, **kwargs):
        self.sync_calls += 1
        raise SyncClientUnavailableError("sync client unavailable")

    async def agenerate(self, *, prompt, parser, system_prompt=None, purpose=None, **kwargs):
        self.async_calls += 1
        return self._call(prompt=prompt, parser=parser, system_prompt=system_prompt, purpose=purpose, kwargs=kwargs)


class FakeModelRegistry:
    def __init__(self, models: dict[str, Any]) -> None:
        self.models = models

    def get_model(self, *, model_alias: str) -> Any:
        return self.models[model_alias]


def _lifecycle_observation(
    observation_id: str,
    value: str,
    label: str,
    start: int,
    end: int,
) -> Observation:
    return Observation(
        detection_result_id="r1",
        observation_id=observation_id,
        source_text_ref=SourceTextRef(record_id="record-1", field_id=COL_TEXT),
        value=value,
        label=label,
        start_position=start,
        end_position=end,
        source_kind="initial_detector",
        source_instance="gliner-pii-detector",
        admission_route=AdmissionRoute.llm_validation,
        native_score=0.9,
    )


def _detection_result(text: str, observations: list[Observation]) -> DetectionResult:
    source_text_ref = SourceTextRef(record_id="record-1", field_id=COL_TEXT)
    return DetectionResult(
        detection_result_id="r1",
        source_view=SourceView(
            source_text_ref=source_text_ref,
            start_position=0,
            end_position=len(text),
        ),
        source_kind="initial_detector",
        source_instance="gliner-pii-detector",
        observations=tuple(observations),
    )


# ---------------------------------------------------------------------------
# Pure helpers: conflict grouping and request prompts
# ---------------------------------------------------------------------------


class TestValidationConflictGroups:
    def test_builds_transitive_connected_components(self) -> None:
        observations = [
            _lifecycle_observation("o1", "Alice", "first_name", 0, 5),
            _lifecycle_observation("o2", "ce John", "full_name", 3, 10),
            _lifecycle_observation("o3", "hn", "last_name", 9, 11),
            _lifecycle_observation("o4", "Bob", "first_name", 12, 15),
        ]

        groups = build_validation_conflict_groups(observations)

        assert [[item.observation_id for item in group] for group in groups] == [
            ["o1", "o2", "o3"],
            ["o4"],
        ]

    def test_chunking_keeps_groups_atomic_within_target(self) -> None:
        overlap_group = (
            _lifecycle_observation("o1", "Alice", "first_name", 0, 5),
            _lifecycle_observation("o2", "Alice", "full_name", 0, 5),
        )
        singleton = (_lifecycle_observation("o3", "Bob", "first_name", 10, 13),)

        chunks = chunk_validation_conflict_groups([overlap_group, singleton], max_entities_per_call=2)

        assert [[item.observation_id for group in chunk for item in group] for chunk in chunks] == [
            ["o1", "o2"],
            ["o3"],
        ]

    def test_oversized_transitive_group_is_sent_alone_between_other_groups(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        observations = [
            _lifecycle_observation("before", "Bob", "first_name", 0, 3),
            _lifecycle_observation("o1", "Alice", "first_name", 4, 9),
            _lifecycle_observation("o2", "ce Johnson", "full_name", 7, 17),
            _lifecycle_observation("o3", "on Smith", "last_name", 15, 23),
            _lifecycle_observation("after", "Eve", "first_name", 24, 27),
        ]
        groups = build_validation_conflict_groups(observations)

        with caplog.at_level(logging.WARNING, logger="anonymizer.detection.chunked_validation"):
            chunks = chunk_validation_conflict_groups(groups, max_entities_per_call=2)

        assert [[item.observation_id for group in chunk for item in group] for chunk in chunks] == [
            ["before"],
            ["o1", "o2", "o3"],
            ["after"],
        ]
        assert "3 candidates" in caplog.text
        assert "configured target 2" in caplog.text
        assert all(item.value not in caplog.text for item in observations)

    @pytest.mark.parametrize(
        ("notation", "expected_marker"),
        [
            (TagNotation.xml, "<NAGROUP_g1>Alice</NAGROUP_g1>"),
            (TagNotation.bracket, "[[Alice|NAGROUP_g1]]"),
            (TagNotation.paren, "((SENSITIVE:NAGROUP_g1|Alice))"),
            (
                TagNotation.sentinel,
                "<<SENSITIVE:NAGROUP_g1>>Alice<</SENSITIVE:NAGROUP_g1>>",
            ),
        ],
    )
    def test_request_inventory_uses_one_neutral_marker_per_overlap_group(
        self,
        notation: TagNotation,
        expected_marker: str,
    ) -> None:
        text = "Alice met Bob"
        detection_result = _detection_result(
            text,
            [
                _lifecycle_observation("o1", "Alice", "first_name", 0, 5),
                _lifecycle_observation("o2", "Alice", "full_name", 0, 5),
                _lifecycle_observation("o3", "Bob", "first_name", 10, 13),
            ],
        )

        requests = prepare_validation_requests(
            detection_result=detection_result,
            canonical_text=text,
            max_entities_per_call=3,
            excerpt_window_chars=20,
            single_chunk_full_text=True,
            notation=notation,
            prompt_template=_MINIMAL_TEMPLATE,
            effective_label_policy=EffectiveLabelPolicy(allowed_labels=("first_name", "full_name")),
        )

        assert len(requests) == 1
        request = requests[0]
        assert expected_marker in request.prompt
        tagged_context = request.prompt.split("|SKELETON:", maxsplit=1)[0]
        assert tagged_context.count("NAGROUP_g1") == expected_marker.count("NAGROUP_g1")
        assert [(entry.alias, entry.conflict_group_id) for entry in request.inventory.entries] == [
            ("c1", "g1"),
            ("c2", "g1"),
            ("c3", None),
        ]
        assert [(entry.start_position, entry.end_position) for entry in request.inventory.entries] == [
            (0, 5),
            (0, 5),
            (10, 13),
        ]

    @pytest.mark.parametrize("notation", list(TagNotation))
    @pytest.mark.parametrize("request_mode", ["full_view", "compact", "multiple"])
    def test_prompt_offsets_index_the_untagged_excerpt_while_inventory_offsets_stay_canonical(
        self, notation: TagNotation, request_mode: str
    ) -> None:
        text = "Prelude " * 20 + "Alice met Bob. " + "padding " * 12 + "Alice met Eve."
        first_start = text.index("Alice")
        second_start = text.rindex("Alice")
        detection_result = _detection_result(
            text,
            [
                _lifecycle_observation("o1", "Alice", "first_name", first_start, first_start + 5),
                _lifecycle_observation("o2", "Alice", "full_name", first_start, first_start + 5),
                _lifecycle_observation("o3", "Alice", "first_name", second_start, second_start + 5),
            ],
        )
        detection_result = detection_result.model_copy(
            update={
                "source_view": SourceView(
                    source_text_ref=detection_result.source_view.source_text_ref,
                    start_position=40,
                    end_position=len(text),
                )
            }
        )
        requests = prepare_validation_requests(
            detection_result=detection_result,
            canonical_text=text,
            max_entities_per_call=2 if request_mode == "multiple" else 3,
            excerpt_window_chars=7,
            single_chunk_full_text=request_mode != "compact",
            notation=notation,
            prompt_template=_MINIMAL_TEMPLATE,
            effective_label_policy=EffectiveLabelPolicy(),
        )

        assert len(requests) == (2 if request_mode == "multiple" else 1)
        original_bounds = {
            observation.observation_id: (observation.start_position, observation.end_position)
            for observation in detection_result.observations
        }
        for request in requests:
            inventory = request.inventory
            assert inventory.context_view.start_position > 0
            skeleton_text = request.prompt.split("|SKELETON:", maxsplit=1)[1].split("|NOTATION:", maxsplit=1)[0]
            skeleton = ast.literal_eval(skeleton_text)
            for shown, entry in zip(skeleton["decisions"], inventory.entries, strict=True):
                assert (entry.start_position, entry.end_position) == original_bounds[
                    entry.observation_ref.observation_id
                ]
                assert shown["start_position"] == entry.start_position - inventory.context_view.start_position
                assert shown["end_position"] == entry.end_position - inventory.context_view.start_position
                assert inventory.source_context[shown["start_position"] : shown["end_position"]] == entry.value
                assert text[entry.start_position : entry.end_position] == entry.value
                assert shown["id"] == entry.alias
                assert shown["conflict_group_id"] == entry.conflict_group_id

    @pytest.mark.parametrize(
        ("notation", "first_context", "second_context"),
        [
            (TagNotation.xml, "<first_name>Alice</first_name> met Bob", "Alice met <first_name>Bob</first_name>"),
            (TagNotation.bracket, "[[Alice|first_name]] met Bob", "Alice met [[Bob|first_name]]"),
            (TagNotation.paren, "((SENSITIVE:first_name|Alice)) met Bob", "Alice met ((SENSITIVE:first_name|Bob))"),
            (
                TagNotation.sentinel,
                "<<SENSITIVE:first_name>>Alice<</SENSITIVE:first_name>> met Bob",
                "Alice met <<SENSITIVE:first_name>>Bob<</SENSITIVE:first_name>>",
            ),
        ],
    )
    def test_other_requests_observations_stay_as_untagged_context(
        self, notation: TagNotation, first_context: str, second_context: str
    ) -> None:
        text = "Alice met Bob"
        requests = prepare_validation_requests(
            detection_result=_detection_result(
                text,
                [
                    _lifecycle_observation("o1", "Alice", "first_name", 0, 5),
                    _lifecycle_observation("o2", "Bob", "first_name", 10, 13),
                ],
            ),
            canonical_text=text,
            max_entities_per_call=1,
            excerpt_window_chars=50,
            single_chunk_full_text=True,
            notation=notation,
            prompt_template=_MINIMAL_TEMPLATE,
            effective_label_policy=EffectiveLabelPolicy(),
        )

        assert [request.prompt.split("|SKELETON:", maxsplit=1)[0] for request in requests] == [
            f"TAGGED:{first_context}",
            f"TAGGED:{second_context}",
        ]
        assert [
            [entry.observation_ref.observation_id for entry in request.inventory.entries] for request in requests
        ] == [
            ["o1"],
            ["o2"],
        ]


class TestRenderChunkPrompt:
    def test_substitutes_excerpt_skeleton_and_notation(self) -> None:
        template = (
            "Input: {{ _seed_tagged_text }}\n"
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


# ---------------------------------------------------------------------------
# Async dispatch: chunked_validate_row end-to-end with fake facades
# ---------------------------------------------------------------------------


def _build_row(
    *,
    text: str,
    observations: list[Observation],
    notation: TagNotation = TagNotation.xml,
) -> dict[str, Any]:
    return {
        COL_TEXT: text,
        COL_SEED_DETECTION_RESULT: _detection_result(text, observations).model_dump(mode="json"),
        COL_TAG_NOTATION: notation.value,
    }


_MINIMAL_TEMPLATE = "TAGGED:{{ _seed_tagged_text }}|SKELETON:{{ _validation_skeleton }}|NOTATION:{{ _tag_notation }}"


class TestLifecycleValidationRequests:
    def test_chunked_validation_persists_request_bound_inventories_and_decisions(self) -> None:
        text = "Alice Johnson"
        detection_result = _detection_result(
            text,
            [
                _lifecycle_observation("o1", "Alice", "first_name", 0, 5),
                _lifecycle_observation("o2", "Alice Johnson", "full_name", 0, 13),
            ],
        )
        row = {
            COL_TEXT: text,
            COL_SEED_DETECTION_RESULT: detection_result.model_dump(mode="json"),
            COL_TAG_NOTATION: TagNotation.xml.value,
        }

        facade = FakeFacade(
            "v0",
            response={
                "decisions": [
                    {"id": "c1", "decision": "keep"},
                    {"id": "c2", "decision": "drop"},
                ]
            },
        )
        params = ChunkedValidationParams(
            pool=["v0"],
            max_entities_per_call=2,
            excerpt_window_chars=20,
            prompt_template=_MINIMAL_TEMPLATE,
            allowed_entity_labels=["first_name", "full_name"],
        )

        output = chunked_validate_row(row, params, {"v0": facade})

        assert len(facade.calls) == 1
        assert "<NAGROUP_g1>Alice Johnson</NAGROUP_g1>" in facade.calls[0]["prompt"]
        inventories = [ValidationRequestInventory.model_validate(item) for item in output[COL_VALIDATION_INVENTORIES]]
        assert len(inventories) == 1
        assert [entry.alias for entry in inventories[0].entries] == ["c1", "c2"]
        decision_sets = ValidationRequestDecisionSets.model_validate(output[COL_VALIDATION_DECISIONS])
        assert len(decision_sets.responses) == 1
        assert decision_sets.responses[0].request_id == inventories[0].request_id
        assert [item.id for item in decision_sets.responses[0].decisions] == ["c1", "c2"]

    @pytest.mark.parametrize("run_async", [False, True])
    def test_oversized_conflict_group_is_dispatched_intact(self, run_async: bool) -> None:
        text = "Alice"
        detection_result = _detection_result(
            text,
            [
                _lifecycle_observation("o1", "Alice", "first_name", 0, 5),
                _lifecycle_observation("o2", "Alice", "full_name", 0, 5),
            ],
        )
        row = {
            COL_TEXT: text,
            COL_SEED_DETECTION_RESULT: detection_result.model_dump(mode="json"),
            COL_TAG_NOTATION: TagNotation.xml.value,
        }
        facade = FakeFacade(
            "v0", response={"decisions": [{"id": "c1", "decision": "keep"}, {"id": "c2", "decision": "drop"}]}
        )
        params = ChunkedValidationParams(
            pool=["v0"],
            max_entities_per_call=1,
            excerpt_window_chars=20,
            prompt_template=_MINIMAL_TEMPLATE,
        )

        if run_async:
            output = asyncio.run(chunked_validate_row_async(row, params, {"v0": facade}))
        else:
            output = chunked_validate_row(row, params, {"v0": facade})

        assert len(facade.calls) == 1
        inventories = [ValidationRequestInventory.model_validate(item) for item in output[COL_VALIDATION_INVENTORIES]]
        assert len(inventories) == 1
        assert [entry.observation_ref.observation_id for entry in inventories[0].entries] == ["o1", "o2"]
        responses = ValidationRequestDecisionSets.model_validate(output[COL_VALIDATION_DECISIONS]).responses
        assert [decision.id for decision in responses[0].decisions] == ["c1", "c2"]

    def test_duplicate_assigned_alias_retries_same_single_alias_request(self) -> None:
        text = "Alice"
        detection_result = _detection_result(
            text,
            [_lifecycle_observation("o1", "Alice", "first_name", 0, 5)],
        )
        row = {
            COL_TEXT: text,
            COL_SEED_DETECTION_RESULT: detection_result.model_dump(mode="json"),
            COL_TAG_NOTATION: TagNotation.xml.value,
        }
        responses = iter(
            [
                {
                    "decisions": [
                        {"id": "c1", "decision": "drop"},
                        {"id": "c1", "decision": "keep"},
                    ]
                },
                {"decisions": [{"id": "c1", "decision": "keep"}]},
            ]
        )
        facade = FakeFacade("v0", response=lambda _prompt: next(responses))
        params = ChunkedValidationParams(
            pool=["v0"],
            max_entities_per_call=5,
            excerpt_window_chars=20,
            prompt_template=_MINIMAL_TEMPLATE,
        )

        output = chunked_validate_row(row, params, {"v0": facade})

        assert len(facade.calls) == 2
        decision_sets = ValidationRequestDecisionSets.model_validate(output[COL_VALIDATION_DECISIONS])
        assert [item.decision for item in decision_sets.responses[0].decisions] == [ValidationChoice.keep]

    def test_duplicate_assigned_alias_fails_over_to_next_pool_member(self) -> None:
        text = "Alice"
        detection_result = _detection_result(
            text,
            [_lifecycle_observation("o1", "Alice", "first_name", 0, 5)],
        )
        row = {
            COL_TEXT: text,
            COL_SEED_DETECTION_RESULT: detection_result.model_dump(mode="json"),
            COL_TAG_NOTATION: TagNotation.xml.value,
        }
        invalid = FakeFacade(
            "v0",
            response={
                "decisions": [
                    {"id": "c1", "decision": "drop"},
                    {"id": "c1", "decision": "keep"},
                ]
            },
        )
        valid = FakeFacade(
            "v1",
            response={"decisions": [{"id": "c1", "decision": "keep"}]},
        )
        params = ChunkedValidationParams(
            pool=["v0", "v1"],
            max_entities_per_call=5,
            excerpt_window_chars=20,
            prompt_template=_MINIMAL_TEMPLATE,
        )

        output = chunked_validate_row(row, params, {"v0": invalid, "v1": valid})

        assert len(invalid.calls) == 1
        assert len(valid.calls) == 1
        decision_sets = ValidationRequestDecisionSets.model_validate(output[COL_VALIDATION_DECISIONS])
        assert decision_sets.responses[0].decisions[0].decision is ValidationChoice.keep

    def test_duplicate_assigned_alias_exhaustion_fails_the_row(self) -> None:
        text = "Alice"
        detection_result = _detection_result(
            text,
            [_lifecycle_observation("o1", "Alice", "first_name", 0, 5)],
        )
        row = {
            COL_TEXT: text,
            COL_SEED_DETECTION_RESULT: detection_result.model_dump(mode="json"),
            COL_TAG_NOTATION: TagNotation.xml.value,
        }
        facade = FakeFacade(
            "v0",
            response={
                "decisions": [
                    {"id": "c1", "decision": "drop"},
                    {"id": "c1", "decision": "keep"},
                ]
            },
        )
        params = ChunkedValidationParams(
            pool=["v0"],
            max_entities_per_call=5,
            excerpt_window_chars=20,
            prompt_template=_MINIMAL_TEMPLATE,
        )

        with pytest.raises(RuntimeError, match="duplicate assigned aliases"):
            chunked_validate_row(row, params, {"v0": facade})
        assert len(facade.calls) == 2

    def test_async_path_uses_the_same_bounded_semantic_retry(self) -> None:
        text = "Alice"
        detection_result = _detection_result(
            text,
            [_lifecycle_observation("o1", "Alice", "first_name", 0, 5)],
        )
        row = {
            COL_TEXT: text,
            COL_SEED_DETECTION_RESULT: detection_result.model_dump(mode="json"),
            COL_TAG_NOTATION: TagNotation.xml.value,
        }
        responses = iter(
            [
                {
                    "decisions": [
                        {"id": "c1", "decision": "drop"},
                        {"id": "c1", "decision": "keep"},
                    ]
                },
                {"decisions": [{"id": "c1", "decision": "keep"}]},
            ]
        )
        facade = FakeFacade("v0", response=lambda _prompt: next(responses))
        params = ChunkedValidationParams(
            pool=["v0"],
            max_entities_per_call=5,
            excerpt_window_chars=20,
            prompt_template=_MINIMAL_TEMPLATE,
        )

        output = asyncio.run(chunked_validate_row_async(row, params, {"v0": facade}))

        assert len(facade.calls) == 2
        decision_sets = ValidationRequestDecisionSets.model_validate(output[COL_VALIDATION_DECISIONS])
        assert decision_sets.responses[0].decisions[0].decision is ValidationChoice.keep


class TestChunkedValidateRowPoolOfOne:
    def test_single_chunk_single_alias_dispatches_once_and_binds_response(self) -> None:
        text = "Alice and Bob met."
        spans = [
            _lifecycle_observation("a", "Alice", "first_name", 0, 5),
            _lifecycle_observation("b", "Bob", "first_name", 10, 13),
        ]
        row = _build_row(text=text, observations=spans)

        facade = FakeFacade(
            "v0",
            response={
                "decisions": [
                    {"id": "c1", "decision": "keep", "proposed_label": "", "reason": "real"},
                    {"id": "c2", "decision": "drop", "proposed_label": "", "reason": "placeholder"},
                ]
            },
        )
        params = ChunkedValidationParams(
            pool=["v0"],
            max_entities_per_call=10,
            excerpt_window_chars=100,
            prompt_template=_MINIMAL_TEMPLATE,
        )

        out = chunked_validate_row(row, params, {"v0": facade})

        assert len(facade.calls) == 1
        responses = ValidationRequestDecisionSets.model_validate(out[COL_VALIDATION_DECISIONS]).responses
        assert len(responses) == 1
        assert {decision.id: decision.decision for decision in responses[0].decisions} == {
            "c1": ValidationChoice.keep,
            "c2": ValidationChoice.drop,
        }

    def test_numeric_value_echo_does_not_fail_response_parsing(self) -> None:
        text = "Reference 42."
        spans = [_lifecycle_observation("id1", "42", "account_number", 10, 12)]
        row = _build_row(text=text, observations=spans)
        facade = FakeFacade(
            "v0",
            response={
                "decisions": [
                    {
                        "id": "c1",
                        "value": 42,
                        "decision": "keep",
                        "proposed_label": "",
                        "reason": "numeric identifier",
                    }
                ]
            },
        )
        params = ChunkedValidationParams(
            pool=["v0"],
            max_entities_per_call=10,
            excerpt_window_chars=100,
            prompt_template=_MINIMAL_TEMPLATE,
        )

        out = chunked_validate_row(row, params, {"v0": facade})

        responses = ValidationRequestDecisionSets.model_validate(out[COL_VALIDATION_DECISIONS]).responses
        assert len(responses[0].decisions) == 1
        assert responses[0].decisions[0].id == "c1"
        assert responses[0].decisions[0].decision is ValidationChoice.keep
        inventory = ValidationRequestInventory.model_validate(out[COL_VALIDATION_INVENTORIES][0])
        assert inventory.entries[0].value == "42"
        assert inventory.entries[0].label == "account_number"

    def test_single_chunk_sends_single_chunk_tagged_text_not_windowed_excerpt(self) -> None:
        """A single chunk sees the full document unless compact excerpts are enabled."""
        prefix = "HEADER_MARKER_ALPHA " * 80
        suffix = " FOOTER_MARKER_OMEGA" * 80
        middle = "Alice met Bob."
        text = prefix + middle + suffix
        alice_start = len(prefix)
        bob_start = alice_start + 10

        spans = [
            _lifecycle_observation("a", "Alice", "first_name", alice_start, alice_start + 5),
            _lifecycle_observation("b", "Bob", "first_name", bob_start, bob_start + 3),
        ]
        row = _build_row(text=text, observations=spans)

        facade = FakeFacade(
            "v0",
            response={
                "decisions": [
                    {"id": "c1", "decision": "keep"},
                    {"id": "c2", "decision": "keep"},
                ]
            },
        )
        params = ChunkedValidationParams(
            pool=["v0"],
            max_entities_per_call=10,
            excerpt_window_chars=50,
            prompt_template=_MINIMAL_TEMPLATE,
        )

        chunked_validate_row(row, params, {"v0": facade})

        assert len(facade.calls) == 1
        prompt = facade.calls[0]["prompt"]
        assert "HEADER_MARKER_ALPHA" in prompt, (
            "single-chunk row should receive the full text prefix; a 50-char window "
            "around Alice/Bob would clip the prefix entirely."
        )
        assert "FOOTER_MARKER_OMEGA" in prompt, (
            "single-chunk row should receive the full text suffix; a 50-char window "
            "around Alice/Bob would clip the suffix entirely."
        )

    def test_single_chunk_can_use_compact_excerpt_when_configured(self) -> None:
        prefix = "START_ONLY_MARKER " + ("prefix filler " * 80)
        suffix = (" suffix filler" * 80) + " END_ONLY_MARKER"
        middle = "Alice met Bob."
        text = prefix + middle + suffix
        alice_start = len(prefix)
        bob_start = alice_start + 10

        spans = [
            _lifecycle_observation("a", "Alice", "first_name", alice_start, alice_start + 5),
            _lifecycle_observation("b", "Bob", "first_name", bob_start, bob_start + 3),
        ]
        row = _build_row(text=text, observations=spans)

        facade = FakeFacade("v0", response={"decisions": [{"id": "c1", "decision": "keep"}]})
        params = ChunkedValidationParams(
            pool=["v0"],
            max_entities_per_call=10,
            excerpt_window_chars=20,
            single_chunk_full_text=False,
            prompt_template=_MINIMAL_TEMPLATE,
        )

        chunked_validate_row(row, params, {"v0": facade})

        prompt = facade.calls[0]["prompt"]
        assert "Alice" in prompt
        assert "Bob" in prompt
        assert "START_ONLY_MARKER" not in prompt
        assert "END_ONLY_MARKER" not in prompt

    def test_empty_candidates_short_circuits_without_calls(self) -> None:
        row = _build_row(text="hello", observations=[])
        facade = FakeFacade("v0", response={"decisions": []})
        params = ChunkedValidationParams(
            pool=["v0"], max_entities_per_call=10, excerpt_window_chars=50, prompt_template=_MINIMAL_TEMPLATE
        )

        out = chunked_validate_row(row, params, {"v0": facade})

        assert facade.calls == []
        assert ValidationRequestDecisionSets.model_validate(out[COL_VALIDATION_DECISIONS]).responses == ()
        assert out[COL_VALIDATION_INVENTORIES] == []

    def test_system_prompt_is_forwarded_to_facade(self) -> None:
        # ``ChunkedValidationParams.system_prompt`` must reach ``facade.generate``.
        # The recipe appends JSON task instructions before dispatch, so we assert
        # substring containment with a distinctive sentinel rather than equality.
        text = "Alice spoke."
        spans = [_lifecycle_observation("a", "Alice", "first_name", 0, 5)]
        row = _build_row(text=text, observations=spans)

        facade = FakeFacade(
            "v0",
            response={"decisions": [{"id": "c1", "decision": "keep", "proposed_label": "", "reason": "x"}]},
        )
        sentinel = "SYSPROMPT_SENTINEL_CHUNKED_VALIDATION_TEST"
        params = ChunkedValidationParams(
            pool=["v0"],
            max_entities_per_call=10,
            excerpt_window_chars=50,
            prompt_template=_MINIMAL_TEMPLATE,
            system_prompt=f"You are a validator. {sentinel}",
        )

        chunked_validate_row(row, params, {"v0": facade})

        assert len(facade.calls) == 1
        forwarded = facade.calls[0]["system_prompt"]
        assert forwarded is not None
        assert sentinel in forwarded

    def test_system_prompt_default_none_is_forwarded_untouched(self) -> None:
        # The recipe maps ``None`` input to ``None`` output; this test pins
        # that no intermediate layer replaces it with a placeholder string
        # when ``ChunkedValidationParams.system_prompt`` is left at its
        # default.
        text = "Alice spoke."
        spans = [_lifecycle_observation("a", "Alice", "first_name", 0, 5)]
        row = _build_row(text=text, observations=spans)

        facade = FakeFacade(
            "v0",
            response={"decisions": [{"id": "c1", "decision": "keep", "proposed_label": "", "reason": "x"}]},
        )
        params = ChunkedValidationParams(
            pool=["v0"],
            max_entities_per_call=10,
            excerpt_window_chars=50,
            prompt_template=_MINIMAL_TEMPLATE,
            # system_prompt intentionally omitted (default None)
        )

        chunked_validate_row(row, params, {"v0": facade})

        assert len(facade.calls) == 1
        # The recipe maps ``None`` input to ``None`` output, so the facade
        # should receive no system prompt.
        assert facade.calls[0]["system_prompt"] is None


class TestChunkedValidateRowPoolOfTwoRoundRobin:
    def test_chunks_assigned_round_robin_across_pool(self) -> None:
        text = "A B C D E F" + " " * 50  # pad so excerpts don't overlap
        spans = [_lifecycle_observation(f"e{i}", chr(ord("A") + i), "first_name", i * 2, i * 2 + 1) for i in range(6)]
        row = _build_row(text=text, observations=spans)

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
        chunked_validate_row(row, params, {"v0": v0, "v1": v1})
        # 6 candidates / 2 per chunk = 3 chunks; round-robin → v0,v1,v0
        assert len(v0.calls) == 2
        assert len(v1.calls) == 1


class TestChunkedValidateRowAsync:
    def test_async_path_uses_agenerate_and_caps_parallel_chunks(self) -> None:
        async def run() -> AsyncTrackingFacade:
            text = "A B C D" + " " * 50
            spans = [
                _lifecycle_observation(f"e{i}", chr(ord("A") + i), "first_name", i * 2, i * 2 + 1) for i in range(4)
            ]
            row = _build_row(text=text, observations=spans)
            facade = AsyncTrackingFacade("v0", response={"decisions": []})
            params = ChunkedValidationParams(
                pool=["v0"],
                max_entities_per_call=1,
                excerpt_window_chars=50,
                max_parallel_chunks=2,
                prompt_template=_MINIMAL_TEMPLATE,
            )

            await chunked_validate_row_async(row, params, {"v0": facade})
            return facade

        facade = asyncio.run(run())
        assert len(facade.calls) == 4
        assert facade.max_active_calls == 2

    def test_async_primary_failure_falls_over_to_secondary(self) -> None:
        async def run() -> tuple[dict[str, Any], AsyncTrackingFacade, AsyncTrackingFacade]:
            spans = [_lifecycle_observation("a", "Alice", "first_name", 0, 5)]
            row = _build_row(text="Alice", observations=spans)
            v0 = AsyncTrackingFacade("v0", raise_on_call=True)
            v1 = AsyncTrackingFacade("v1", response={"decisions": [{"id": "c1", "decision": "keep"}]})
            params = ChunkedValidationParams(
                pool=["v0", "v1"],
                max_entities_per_call=5,
                excerpt_window_chars=50,
                prompt_template=_MINIMAL_TEMPLATE,
            )

            out = await chunked_validate_row_async(row, params, {"v0": v0, "v1": v1})
            return out, v0, v1

        out, v0, v1 = asyncio.run(run())
        responses = ValidationRequestDecisionSets.model_validate(out[COL_VALIDATION_DECISIONS]).responses
        assert responses[0].decisions[0].id == "c1"
        assert len(v0.calls) == 1
        assert len(v1.calls) == 1


class TestChunkedValidationGenerator:
    def test_generate_bridges_sync_unavailable_to_agenerate(self) -> None:
        spans = [_lifecycle_observation("a", "Alice", "first_name", 0, 5)]
        row = _build_row(text="Alice", observations=spans)
        facade = SyncUnavailableFacade("v0", response={"decisions": [{"id": "c1", "decision": "keep"}]})
        generator = ChunkedValidationGenerator(
            ChunkedValidationConfig(
                name=COL_VALIDATION_DECISIONS,
                pool=["v0"],
                max_entities_per_call=5,
                excerpt_window_chars=50,
                prompt_template=_MINIMAL_TEMPLATE,
            ),
            resource_provider=SimpleNamespace(  # ty: ignore[invalid-argument-type]
                model_registry=FakeModelRegistry({"v0": facade})
            ),
        )

        out = generator.generate(row)

        responses = ValidationRequestDecisionSets.model_validate(out[COL_VALIDATION_DECISIONS]).responses
        assert responses[0].decisions[0].id == "c1"
        assert facade.sync_calls == 1
        assert facade.async_calls == 1


class TestChunkedValidateRowMultiChunkReassembly:
    def test_decisions_remain_bound_to_each_chunk_inventory(self) -> None:
        text = "one two three four five"
        spans = [
            _lifecycle_observation("c1", "one", "first_name", 0, 3),
            _lifecycle_observation("c2", "two", "first_name", 4, 7),
            _lifecycle_observation("c3", "three", "first_name", 8, 13),
            _lifecycle_observation("c4", "four", "first_name", 14, 18),
        ]
        row = _build_row(text=text, observations=spans)

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
        out = chunked_validate_row(row, params, {"v0": v0, "v1": v1})
        responses = ValidationRequestDecisionSets.model_validate(out[COL_VALIDATION_DECISIONS]).responses
        assert len(responses) == 2
        assert [[decision.id for decision in response.decisions] for response in responses] == [
            ["c1", "c2"],
            ["c1", "c2"],
        ]
        decisions = _decisions_by_observation(out)
        assert set(decisions) == {"c1", "c2", "c3", "c4"}
        # Chunk 0 (c1,c2) → v0; chunk 1 (c3,c4) → v1.
        assert decisions["c1"].reason == "v0"
        assert decisions["c2"].reason == "v0"
        assert decisions["c3"].reason == "v1"
        assert decisions["c4"].reason == "v1"


class TestChunkedValidateRowFailurePropagation:
    def test_row_fails_only_when_every_pool_member_raises_for_a_chunk(self) -> None:
        """With failover enabled, a chunk only fails when *every* pool member raises.

        Downstream DD reporting then turns that row into a FailedRecord via
        ``NddAdapter._detect_missing_records`` — no unscrubbed passthrough.
        """
        spans = [
            _lifecycle_observation("a", "Alice", "first_name", 0, 5),
            _lifecycle_observation("b", "Bob", "first_name", 10, 13),
        ]
        row = _build_row(text="Alice and Bob", observations=spans)

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
            chunked_validate_row(row, params, {"v0": v0, "v1": v1})


def test_dispatch_requires_assigned_aliases() -> None:
    for dispatch in (_dispatch_chunk, _dispatch_chunk_async):
        assert inspect.signature(dispatch).parameters["assigned_aliases"].default is inspect.Parameter.empty


@pytest.mark.parametrize("run_async", [False, True])
@pytest.mark.parametrize("recover", [False, True])
def test_failover_logs_exception_type_without_source_text(
    run_async: bool, recover: bool, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "sensitive-exception-message"

    def fail(_prompt: str) -> dict[str, Any]:
        raise RuntimeError(secret)

    row = _build_row(text="Alice", observations=[_lifecycle_observation("o1", "Alice", "first_name", 0, 5)])
    pool = ["primary", "secondary"] if recover else ["primary"]
    models = {
        "primary": FakeFacade("primary", response=fail),
        "secondary": FakeFacade("secondary", response={"decisions": [{"id": "c1", "decision": "keep"}]}),
    }
    params = ChunkedValidationParams(
        pool=pool, max_entities_per_call=5, excerpt_window_chars=50, prompt_template=_MINIMAL_TEMPLATE
    )

    def run() -> dict[str, Any]:
        if run_async:
            return asyncio.run(chunked_validate_row_async(row, params, models))
        return chunked_validate_row(row, params, models)

    with caplog.at_level(logging.WARNING, logger="anonymizer.detection.chunked_validation"):
        if recover:
            run()
        else:
            with pytest.raises(RuntimeError, match=secret):
                run()

    assert "RuntimeError" in caplog.text
    assert secret not in caplog.text


class TestChunkedValidateRowCrossAliasFailover:
    def test_primary_alias_failure_falls_over_to_secondary(self) -> None:
        """Primary raises, secondary in pool returns a valid response → chunk succeeds, row does not fail."""
        spans = [_lifecycle_observation("a", "Alice", "first_name", 0, 5)]
        row = _build_row(text="Alice", observations=spans)

        v0 = FakeFacade("v0", raise_on_call=True)
        v1 = FakeFacade("v1", response={"decisions": [{"id": "c1", "decision": "keep"}]})
        params = ChunkedValidationParams(
            pool=["v0", "v1"],
            max_entities_per_call=5,
            excerpt_window_chars=50,
            prompt_template=_MINIMAL_TEMPLATE,
        )
        out = chunked_validate_row(row, params, {"v0": v0, "v1": v1})
        responses = ValidationRequestDecisionSets.model_validate(out[COL_VALIDATION_DECISIONS]).responses
        decisions = responses[0].decisions
        assert len(decisions) == 1
        assert decisions[0].id == "c1"
        assert decisions[0].decision is ValidationChoice.keep
        # Each alias was tried exactly once for this single-chunk row.
        assert len(v0.calls) == 1
        assert len(v1.calls) == 1

    def test_single_alias_pool_does_not_failover(self) -> None:
        """A one-alias pool has no fallback — one attempt, propagate immediately.

        This keeps the behavioural guarantee that pools of size 1 behave
        exactly as the pre-failover dispatch did: no hidden extra attempts.
        """
        spans = [_lifecycle_observation("a", "Alice", "first_name", 0, 5)]
        row = _build_row(text="Alice", observations=spans)

        v0 = FakeFacade("v0", raise_on_call=True)
        params = ChunkedValidationParams(
            pool=["v0"],
            max_entities_per_call=5,
            excerpt_window_chars=50,
            prompt_template=_MINIMAL_TEMPLATE,
        )
        with pytest.raises(RuntimeError, match="forced failure from v0"):
            chunked_validate_row(row, params, {"v0": v0})
        assert len(v0.calls) == 1

    def test_failover_wraps_starting_from_round_robin_primary(self) -> None:
        """With pool [v0,v1] and 3 chunks, chunk 1's primary is v1.

        If v1 fails, failover wraps to v0 (next position mod len(pool)).
        We verify by ensuring chunk 1 succeeds on v0's response, while
        chunk 0 and chunk 2 succeed on v0 (their primary) directly.
        """
        spans = [
            _lifecycle_observation("a", "Alice", "first_name", 0, 5),
            _lifecycle_observation("b", "Bob", "first_name", 10, 13),
            _lifecycle_observation("c", "Carol", "first_name", 18, 23),
        ]
        row = _build_row(text="Alice and Bob and Carol", observations=spans)

        def v0_response(prompt: str) -> dict:
            # The chunk's skeleton is serialized into the prompt; pick the id
            # from there. We can't use the raw text excerpt to distinguish
            # chunks because the text is short enough that every chunk's
            # excerpt window covers the whole string.
            assert "'id': 'c1'" in prompt
            return {"decisions": [{"id": "c1", "decision": "keep"}]}

        v0 = FakeFacade("v0", response=v0_response)
        v1 = FakeFacade("v1", raise_on_call=True)
        params = ChunkedValidationParams(
            pool=["v0", "v1"],
            max_entities_per_call=1,
            excerpt_window_chars=50,
            prompt_template=_MINIMAL_TEMPLATE,
        )
        out = chunked_validate_row(row, params, {"v0": v0, "v1": v1})
        decisions = {
            observation_id: decision.decision for observation_id, decision in _decisions_by_observation(out).items()
        }
        assert decisions == {"a": "keep", "b": "keep", "c": "keep"}
        # v0 serviced all three chunks: chunk 0 + chunk 2 directly, chunk 1 via failover.
        assert len(v0.calls) == 3
        # v1 saw exactly one call — chunk 1's primary attempt that raised.
        assert len(v1.calls) == 1


class TestChunkedValidateRowUnknownAlias:
    def test_unknown_alias_is_preserved_without_an_observation_mapping(self) -> None:
        """Raw responses retain unknown aliases; only assigned aliases identify observations."""
        spans = [_lifecycle_observation("a", "Alice", "first_name", 0, 5)]
        row = _build_row(text="Alice", observations=spans)
        facade = FakeFacade(
            "v0",
            response={
                "decisions": [
                    {"id": "c1", "decision": "keep"},
                    {"id": "unknown", "decision": "drop"},
                ]
            },
        )
        params = ChunkedValidationParams(
            pool=["v0"], max_entities_per_call=5, excerpt_window_chars=20, prompt_template=_MINIMAL_TEMPLATE
        )
        out = chunked_validate_row(row, params, {"v0": facade})
        responses = ValidationRequestDecisionSets.model_validate(out[COL_VALIDATION_DECISIONS]).responses
        assert [decision.id for decision in responses[0].decisions] == ["c1", "unknown"]
        assert set(_decisions_by_observation(out)) == {"a"}


class TestChunkedValidateRowGuardsBadConfig:
    def test_pool_alias_missing_from_models_raises_helpful_error(self) -> None:
        spans = [_lifecycle_observation("a", "Alice", "first_name", 0, 5)]
        row = _build_row(text="Alice", observations=spans)
        params = ChunkedValidationParams(
            pool=["missing_alias"],
            max_entities_per_call=5,
            excerpt_window_chars=20,
            prompt_template=_MINIMAL_TEMPLATE,
        )
        with pytest.raises(KeyError, match="missing_alias"):
            chunked_validate_row(row, params, {"v0": FakeFacade("v0", response={"decisions": []})})


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
            COL_SEED_DETECTION_RESULT,
            COL_TAG_NOTATION,
        }
        assert meta["side_effect_columns"] == [COL_VALIDATION_INVENTORIES]
        # Must not declare columns we deliberately don't read; an over-broad
        # required_columns would distort DAG ordering elsewhere. In particular
        # the post-augmentation views (COL_MERGED_TAGGED_TEXT, COL_VALIDATION_CANDIDATES)
        # are downstream of this step and would create a cycle if declared.
        assert COL_VALIDATION_SKELETON not in meta["required_columns"]
        assert COL_MERGED_TAGGED_TEXT not in meta["required_columns"]
        assert COL_VALIDATION_CANDIDATES not in meta["required_columns"]

    def test_factory_rejects_empty_pool(self) -> None:
        with pytest.raises(ValueError, match="pool is empty"):
            make_chunked_validation_generator([])

    def test_generator_forwards_to_chunked_validate_row(self) -> None:
        """The DD-exposed wrapper is sync; calling it directly must return a
        dict populated by the row logic (not a coroutine).
        """
        spans = [_lifecycle_observation("a", "Alice", "first_name", 0, 5)]
        row = _build_row(text="Alice", observations=spans)
        fn = make_chunked_validation_generator(["v0"])
        facade = FakeFacade("v0", response={"decisions": [{"id": "c1", "decision": "keep"}]})
        params = ChunkedValidationParams(
            pool=["v0"], max_entities_per_call=5, excerpt_window_chars=20, prompt_template=_MINIMAL_TEMPLATE
        )
        out = fn(row, params, {"v0": facade})
        assert isinstance(out, dict)
        responses = ValidationRequestDecisionSets.model_validate(out[COL_VALIDATION_DECISIONS]).responses
        assert responses[0].decisions[0].id == "c1"

    def test_generator_is_sync_callable_returning_dict(self) -> None:
        """Regression: DD's default thread-pool engine calls the wrapper
        synchronously and rejects coroutine returns with "must return a dict,
        got coroutine". Guard against accidentally re-introducing an async
        outer wrapper."""
        import asyncio as _asyncio
        import inspect as _inspect

        fn = make_chunked_validation_generator(["v0"])
        inner = _inspect.unwrap(fn)
        assert not _asyncio.iscoroutinefunction(inner), (
            "DD-exposed generator must be sync; an async outer wrapper breaks the default thread-pool engine."
        )


# ---------------------------------------------------------------------------
# Behavioral regression: single-chunk vs multi-chunk parity (pool-of-one).
# ---------------------------------------------------------------------------


def _decisions_by_observation(row: dict[str, Any]) -> dict[str, RawValidationDecisionSchema]:
    inventories = {
        inventory.request_id: inventory
        for inventory in (ValidationRequestInventory.model_validate(item) for item in row[COL_VALIDATION_INVENTORIES])
    }
    responses = ValidationRequestDecisionSets.model_validate(row[COL_VALIDATION_DECISIONS]).responses
    assert len(responses) == len(inventories)
    decisions: dict[str, RawValidationDecisionSchema] = {}
    for response in responses:
        inventory = inventories[response.request_id]
        refs_by_alias = {entry.alias: entry.observation_ref for entry in inventory.entries}
        for decision in response.decisions:
            if decision.id in refs_by_alias:
                observation_id = refs_by_alias[decision.id].observation_id
                assert observation_id not in decisions
                decisions[observation_id] = decision
    return decisions


class TestChunkedValidationRegression:
    """Partitioning preserves decisions mapped through each request's own aliases."""

    SCENARIO_TEXT = "Alice met Bob in Chicago at Acme HQ; Doe introduced Eve to the team later."

    @pytest.fixture
    def scenario(self) -> tuple[list[Observation], dict[str, dict[str, Any]]]:
        values_and_labels = [
            ("Alice", "first_name"),
            ("Bob", "first_name"),
            ("Chicago", "city"),
            ("Acme", "organization"),
            ("Doe", "last_name"),
            ("Eve", "first_name"),
        ]
        observations = [
            _lifecycle_observation(
                f"o{index}", value, label, self.SCENARIO_TEXT.index(value), self.SCENARIO_TEXT.index(value) + len(value)
            )
            for index, (value, label) in enumerate(values_and_labels, start=1)
        ]
        decisions_by_value: dict[str, dict[str, Any]] = {
            "Alice": {"decision": "keep"},
            "Bob": {"decision": "drop"},
            "Chicago": {"decision": "reclass", "proposed_label": "location"},
            "Acme": {"decision": "keep"},
            "Doe": {"decision": "reclass", "proposed_label": "surname"},
            # Eve is omitted to exercise fallback independently of chunk boundaries.
        }
        return observations, decisions_by_value

    def _run(
        self,
        *,
        observations: list[Observation],
        decisions_by_value: dict[str, dict[str, Any]],
        max_per_call: int,
    ) -> tuple[dict[str, tuple[ValidationChoice | None, str]], list[tuple[str, str]], int]:
        def respond(prompt: str) -> dict[str, Any]:
            skeleton_text = prompt.split("|SKELETON:", maxsplit=1)[1].split("|NOTATION:", maxsplit=1)[0]
            skeleton = ast.literal_eval(skeleton_text)
            return {
                "decisions": [
                    {"id": entry["id"], **decisions_by_value[entry["value"]]}
                    for entry in skeleton["decisions"]
                    if entry["value"] in decisions_by_value
                ]
            }

        row = _build_row(text=self.SCENARIO_TEXT, observations=observations)
        facade = FakeFacade("solo", response=respond)
        params = ChunkedValidationParams(
            pool=["solo"],
            max_entities_per_call=max_per_call,
            excerpt_window_chars=200,
            prompt_template=_MINIMAL_TEMPLATE,
        )
        out = chunked_validate_row(row, params, {"solo": facade})
        normalized = {
            observation_id: (decision.decision, decision.proposed_label or "")
            for observation_id, decision in _decisions_by_observation(out).items()
        }
        admit_validation_decisions(out)
        admissions = AdmissionResults.model_validate(out[COL_ADMISSION_RESULTS])
        assert len(admissions.results) == len(observations)
        assert admissions.results[-1].status is AdmissionStatus.fallback_protected
        observations_by_ref = {observation.ref: observation for observation in observations}
        retained: list[tuple[str, str]] = []
        for result in admissions.results:
            if result.status is AdmissionStatus.rejected:
                continue
            assert result.effective_label is not None
            retained.append((observations_by_ref[result.observation_ref].value, result.effective_label))
        return normalized, retained, len(facade.calls)

    def test_multi_chunk_matches_single_chunk(self, scenario) -> None:
        observations, decisions_by_value = scenario
        single_decisions, single_validated, single_calls = self._run(
            observations=observations, decisions_by_value=decisions_by_value, max_per_call=10
        )
        multi_decisions, multi_validated, multi_calls = self._run(
            observations=observations, decisions_by_value=decisions_by_value, max_per_call=2
        )

        assert single_calls == 1
        assert multi_calls == 3
        assert (
            single_decisions
            == multi_decisions
            == {
                "o1": (ValidationChoice.keep, ""),
                "o2": (ValidationChoice.drop, ""),
                "o3": (ValidationChoice.reclass, "location"),
                "o4": (ValidationChoice.keep, ""),
                "o5": (ValidationChoice.reclass, "surname"),
            }
        )
        assert "o6" not in multi_decisions
        assert (
            single_validated
            == multi_validated
            == [
                ("Alice", "first_name"),
                ("Chicago", "location"),
                ("Acme", "organization"),
                ("Doe", "surname"),
                ("Eve", "first_name"),
            ]
        )
