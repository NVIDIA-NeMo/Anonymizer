# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle
from dataclasses import FrozenInstanceError, replace
from typing import Any

import pandas as pd
import pytest

from anonymizer.config.anonymizer_config import AnonymizerConfig
from anonymizer.config.models import ModelSelection
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import (
    COL_FINAL_ENTITIES,
    COL_REPLACED_TEXT,
    COL_REPLACEMENT_APPLICATION,
    COL_TEXT,
)
from anonymizer.engine.execution.graph import (
    _AtomicGroup,
    _CoherenceScope,
    _compile_trivial_graph,
    _ContextScope,
    _DatumId,
    _DatumLink,
    _GraphLimits,
    _GraphValidationCode,
    _GraphValidationError,
    _ProtectionGraph,
    _RelationKind,
    _TextDatum,
    _trivial_graph,
)
from anonymizer.engine.execution.graph_runtime import _GraphExecutionResult, _TrivialGraphRuntime
from anonymizer.engine.execution.invocation import _CompiledInvocation
from anonymizer.engine.execution.pandas_runtime import _PandasExecutionResult
from anonymizer.engine.execution.protection_service import (
    _GraphProtectionFailed,
    _GraphProtectionResult,
    _GraphProtectionSucceeded,
    _TrivialRedactProtectionService,
)
from anonymizer.engine.private_row_verification import (
    PrivateRowVerificationError,
    _InvocationRowVerifier,
    _TerminalOutcome,
)

_LIMITS = _GraphLimits(max_datums=4, max_datum_bytes=64, max_graph_bytes=128)


def _graph(*texts: str) -> _ProtectionGraph:
    return _trivial_graph(tuple(_TextDatum(_DatumId(f"datum-{index}"), text) for index, text in enumerate(texts)))


class _SuccessfulBackend:
    def __init__(self) -> None:
        self.frame: pd.DataFrame | None = None

    def run(
        self,
        dataframe: pd.DataFrame,
        *,
        invocation: _CompiledInvocation,
        data_summary: str | None,
        preview_num_records: int | None,
        verifier: _InvocationRowVerifier,
    ) -> _PandasExecutionResult:
        del invocation, data_summary, preview_num_records
        self.frame = dataframe.copy()
        detected = dataframe.assign(**{COL_FINAL_ENTITIES: [{"entities": []} for _ in range(len(dataframe))]})
        verifier.freeze_accepted_detections(detected)
        final = verifier.finish(detected)
        return _PandasExecutionResult(
            dataframe=final,
            failed_records=[],
            terminal_outcomes=verifier.take_terminal_outcomes(),
            result_row_tokens=verifier.take_result_order(),
        )


def _protect_release_row(
    input_text: str,
    row: dict[str, object],
    model_selection: ModelSelection,
) -> tuple[_ProtectionGraph, _GraphProtectionResult]:
    graph = _graph(input_text)
    token = "token-single"
    dataframe_result = _PandasExecutionResult(
        dataframe=pd.DataFrame([row]),
        failed_records=[],
        terminal_outcomes=((token, _TerminalOutcome.SUCCESS),),
        result_row_tokens=(token,),
    )

    class _StaticRuntime:
        def run(self, *_args: Any, **_kwargs: Any) -> _GraphExecutionResult:
            return _GraphExecutionResult(
                (graph.datums[0].id,),
                (graph.datums[0].text,),
                dataframe_result,
                (token,),
            )

    result = _TrivialRedactProtectionService(_StaticRuntime()).protect(
        graph,
        limits=_LIMITS,
        invocation=_CompiledInvocation.compile(AnonymizerConfig(replace=Redact()), model_selection),
    )
    return graph, result


def test_trivial_graph_is_immutable_and_compiles_in_datum_order() -> None:
    graph = _graph("first", "second")

    compiled = _compile_trivial_graph(graph, limits=_LIMITS)

    assert [datum.text for datum in compiled.datums] == ["first", "second"]
    with pytest.raises(FrozenInstanceError):
        setattr(graph.datums[0], "text", "changed")
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(graph)
    assert "first" not in repr(graph)


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (
            lambda graph: replace(
                graph,
                links=(_DatumLink(graph.datums[0].id, graph.datums[1].id, _RelationKind.RELATED),),
            ),
            _GraphValidationCode.UNSUPPORTED_RELATIONSHIPS,
        ),
        (
            lambda graph: replace(
                graph,
                context_scopes=(_ContextScope(graph.datums[0].id, (graph.datums[1].id,)),),
            ),
            _GraphValidationCode.UNSUPPORTED_CONTEXT,
        ),
        (
            lambda graph: replace(
                graph,
                coherence_scopes=(_CoherenceScope(tuple(datum.id for datum in graph.datums)),),
            ),
            _GraphValidationCode.UNSUPPORTED_COHERENCE,
        ),
        (
            lambda graph: replace(
                graph,
                atomic_groups=(_AtomicGroup(tuple(datum.id for datum in graph.datums)),),
            ),
            _GraphValidationCode.UNSUPPORTED_ATOMICITY,
        ),
    ],
)
def test_first_compiler_rejects_related_record_semantics(
    mutation: Any,
    code: _GraphValidationCode,
) -> None:
    graph = mutation(_graph("first", "second"))

    with pytest.raises(_GraphValidationError) as exc_info:
        _compile_trivial_graph(graph, limits=_LIMITS)

    assert exc_info.value.code is code
    assert repr(exc_info.value) == "<private protection graph error>"


def test_compiler_rejects_duplicate_ids_and_forged_graphs_without_content() -> None:
    duplicate_id = _DatumId("same")
    duplicate = _trivial_graph((_TextDatum(duplicate_id, "secret-a"), _TextDatum(duplicate_id, "secret-b")))
    with pytest.raises(_GraphValidationError) as exc_info:
        _compile_trivial_graph(duplicate, limits=_LIMITS)
    assert exc_info.value.code is _GraphValidationCode.DUPLICATE_DATUM_ID
    assert "secret" not in str(exc_info.value)

    forged = object.__new__(_ProtectionGraph)
    with pytest.raises(_GraphValidationError) as exc_info:
        _compile_trivial_graph(forged, limits=_LIMITS)
    assert exc_info.value.code is _GraphValidationCode.MALFORMED_GRAPH


def test_graph_runtime_lowers_only_text_and_preserves_graph_identity(
    stub_slim_model_selection: ModelSelection,
) -> None:
    backend = _SuccessfulBackend()
    graph = _graph("first", "second")

    result = _TrivialGraphRuntime(backend).run(
        graph,
        limits=_LIMITS,
        invocation=_CompiledInvocation.compile(AnonymizerConfig(replace=Redact()), stub_slim_model_selection),
        data_summary=None,
        preview_num_records=None,
    )

    assert result.datum_ids == tuple(datum.id for datum in graph.datums)
    assert result.dataframe_result.dataframe[COL_TEXT].tolist() == ["first", "second"]
    assert backend.frame is not None
    assert list(backend.frame.columns) == [COL_TEXT, "__anonymizer_private_row_correlation__"]
    assert all(datum.id.value not in backend.frame.to_string() for datum in graph.datums)


def test_graph_runtime_sanitizes_backend_failure(stub_slim_model_selection: ModelSelection) -> None:
    secret = "backend-secret@example.test"

    class _FailingBackend:
        def run(self, *_args: Any, **_kwargs: Any) -> _PandasExecutionResult:
            raise RuntimeError(secret)

    with pytest.raises(PrivateRowVerificationError) as exc_info:
        _TrivialGraphRuntime(_FailingBackend()).run(
            _graph("input-secret@example.test"),
            limits=_LIMITS,
            invocation=_CompiledInvocation.compile(AnonymizerConfig(replace=Redact()), stub_slim_model_selection),
            data_summary=None,
            preview_num_records=None,
        )

    assert secret not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_release_reconciles_reordered_terminal_outcomes_by_private_token(
    stub_slim_model_selection: ModelSelection,
) -> None:
    graph = _graph("first", "second")
    tokens = ("token-first", "token-second")
    dataframe_result = _PandasExecutionResult(
        dataframe=pd.DataFrame(
            {
                COL_TEXT: ["first", "second"],
                COL_REPLACED_TEXT: ["first", "second"],
                COL_FINAL_ENTITIES: [{"entities": []}, {"entities": []}],
                COL_REPLACEMENT_APPLICATION: [
                    {
                        "targeted_span_count": 0,
                        "applied_span_count": 0,
                        "skipped_span_count": 0,
                        "skipped_span_label_counts": {},
                    },
                    {
                        "targeted_span_count": 0,
                        "applied_span_count": 0,
                        "skipped_span_count": 0,
                        "skipped_span_label_counts": {},
                    },
                ],
            }
        ),
        failed_records=[],
        terminal_outcomes=tuple((token, _TerminalOutcome.SUCCESS) for token in reversed(tokens)),
        result_row_tokens=tokens,
    )

    class _ReorderedRuntime:
        def run(self, *_args: Any, **_kwargs: Any) -> _GraphExecutionResult:
            return _GraphExecutionResult(
                tuple(datum.id for datum in graph.datums),
                tuple(datum.text for datum in graph.datums),
                dataframe_result,
                tokens,
            )

    result = _TrivialRedactProtectionService(_ReorderedRuntime()).protect(
        graph,
        limits=_LIMITS,
        invocation=_CompiledInvocation.compile(AnonymizerConfig(replace=Redact()), stub_slim_model_selection),
    )

    assert all(isinstance(outcome, _GraphProtectionSucceeded) for outcome in result.outcomes)
    assert [outcome.output for outcome in result.outcomes if isinstance(outcome, _GraphProtectionSucceeded)] == [
        "first",
        "second",
    ]


def test_release_rejects_unchanged_authoritative_span_when_entity_value_case_differs(
    stub_slim_model_selection: ModelSelection,
) -> None:
    graph, result = _protect_release_row(
        "Alice works here",
        {
            COL_REPLACED_TEXT: "Alice works here",
            COL_FINAL_ENTITIES: {
                "entities": [{"value": "alice", "label": "first_name", "start_position": 0, "end_position": 5}]
            },
            COL_REPLACEMENT_APPLICATION: {
                "targeted_span_count": 1,
                "applied_span_count": 0,
                "skipped_span_count": 1,
                "skipped_span_label_counts": {"first_name": 1},
            },
        },
        stub_slim_model_selection,
    )

    assert result.outcomes == (_GraphProtectionFailed(graph.datums[0].id, "release", "datum"),)


@pytest.mark.parametrize(
    ("entities", "output"),
    [
        (
            [{"value": "alice", "label": "first_name", "start_position": 0, "end_position": 5}],
            "[REDACTED_FIRST_NAME] works here",
        ),
        (
            [{"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5}],
            "Alice works here",
        ),
    ],
)
def test_release_rejects_forged_success_accounting(
    stub_slim_model_selection: ModelSelection,
    entities: list[dict[str, object]],
    output: str,
) -> None:
    graph, result = _protect_release_row(
        "Alice works here",
        {
            COL_REPLACED_TEXT: output,
            COL_FINAL_ENTITIES: {"entities": entities},
            COL_REPLACEMENT_APPLICATION: {
                "targeted_span_count": 1,
                "applied_span_count": 1,
                "skipped_span_count": 0,
                "skipped_span_label_counts": {},
            },
        },
        stub_slim_model_selection,
    )

    assert result.outcomes == (_GraphProtectionFailed(graph.datums[0].id, "release", "datum"),)


@pytest.mark.parametrize(
    "entities",
    [
        [{"value": "Alice", "label": "first_name", "start_position": 0}],
        [{"value": "Alice", "label": "first_name", "start_position": -1, "end_position": 5}],
        [{"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 99}],
        [
            {"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5},
            {"value": "lice ", "label": "alias", "start_position": 1, "end_position": 6},
        ],
    ],
)
def test_release_rejects_malformed_authoritative_spans(
    stub_slim_model_selection: ModelSelection,
    entities: list[dict[str, object]],
) -> None:
    targeted = len(entities)
    graph, result = _protect_release_row(
        "Alice works here",
        {
            COL_REPLACED_TEXT: "[REDACTED] works here",
            COL_FINAL_ENTITIES: {"entities": entities},
            COL_REPLACEMENT_APPLICATION: {
                "targeted_span_count": targeted,
                "applied_span_count": targeted,
                "skipped_span_count": 0,
                "skipped_span_label_counts": {},
            },
        },
        stub_slim_model_selection,
    )

    assert result.outcomes == (_GraphProtectionFailed(graph.datums[0].id, "release", "datum"),)


@pytest.mark.parametrize("entity_label", [None, "", 7])
def test_release_rejects_malformed_accepted_entity_labels(
    stub_slim_model_selection: ModelSelection,
    entity_label: object,
) -> None:
    graph, result = _protect_release_row(
        "Alice works here",
        {
            COL_REPLACED_TEXT: "[REDACTED] works here",
            COL_FINAL_ENTITIES: {
                "entities": [{"value": "Alice", "label": entity_label, "start_position": 0, "end_position": 5}]
            },
            COL_REPLACEMENT_APPLICATION: {
                "targeted_span_count": 1,
                "applied_span_count": 1,
                "skipped_span_count": 0,
                "skipped_span_label_counts": {},
            },
        },
        stub_slim_model_selection,
    )

    assert result.outcomes == (_GraphProtectionFailed(graph.datums[0].id, "release", "datum"),)


@pytest.mark.parametrize(
    "application",
    [
        None,
        {"targeted_span_count": 1, "applied_span_count": 1, "skipped_span_count": 0},
        {
            "targeted_span_count": True,
            "applied_span_count": 1,
            "skipped_span_count": 0,
            "skipped_span_label_counts": {},
        },
        {
            "targeted_span_count": 2,
            "applied_span_count": 2,
            "skipped_span_count": 0,
            "skipped_span_label_counts": {},
        },
        {
            "targeted_span_count": 1,
            "applied_span_count": 0,
            "skipped_span_count": 1,
            "skipped_span_label_counts": {"first_name": 1},
        },
    ],
)
def test_release_rejects_malformed_or_incomplete_replacement_accounting(
    stub_slim_model_selection: ModelSelection,
    application: object,
) -> None:
    graph, result = _protect_release_row(
        "Alice works here",
        {
            COL_REPLACED_TEXT: "[REDACTED_FIRST_NAME] works here",
            COL_FINAL_ENTITIES: {
                "entities": [{"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5}]
            },
            COL_REPLACEMENT_APPLICATION: application,
        },
        stub_slim_model_selection,
    )

    assert result.outcomes == (_GraphProtectionFailed(graph.datums[0].id, "release", "datum"),)


def test_release_accepts_complete_exact_case_redaction(stub_slim_model_selection: ModelSelection) -> None:
    graph, result = _protect_release_row(
        "Alice works here",
        {
            COL_REPLACED_TEXT: "[REDACTED_FIRST_NAME] works here",
            COL_FINAL_ENTITIES: {
                "entities": [{"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5}]
            },
            COL_REPLACEMENT_APPLICATION: {
                "targeted_span_count": 1,
                "applied_span_count": 1,
                "skipped_span_count": 0,
                "skipped_span_label_counts": {},
            },
        },
        stub_slim_model_selection,
    )

    assert result.outcomes == (_GraphProtectionSucceeded(graph.datums[0].id, "[REDACTED_FIRST_NAME] works here", True),)


def test_release_accepts_unchanged_no_detection_with_zero_accounting(
    stub_slim_model_selection: ModelSelection,
) -> None:
    graph, result = _protect_release_row(
        "plain text",
        {
            COL_REPLACED_TEXT: "plain text",
            COL_FINAL_ENTITIES: {"entities": []},
            COL_REPLACEMENT_APPLICATION: {
                "targeted_span_count": 0,
                "applied_span_count": 0,
                "skipped_span_count": 0,
                "skipped_span_label_counts": {},
            },
        },
        stub_slim_model_selection,
    )

    assert result.outcomes == (_GraphProtectionSucceeded(graph.datums[0].id, "plain text", False),)
