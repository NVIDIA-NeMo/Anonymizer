# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle
from dataclasses import FrozenInstanceError
from typing import Any, cast

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
from anonymizer.engine.execution.accounting_admission import _compile_accounting_plan
from anonymizer.engine.execution.accounting_outcomes import _GroupWithheld, _InvocationInconsistent, _InvocationLost
from anonymizer.engine.execution.accounting_plan import _AccountingLimits, _AccountingPlan
from anonymizer.engine.execution.graph import (
    _DatumId,
    _ProtectionGraph,
    _TextDatum,
    _trivial_graph,
)
from anonymizer.engine.execution.graph_runtime import _AccountingGraphRuntime
from anonymizer.engine.execution.invocation import _CompiledInvocation
from anonymizer.engine.execution.pandas_runtime import _PandasExecutionResult
from anonymizer.engine.execution.protection_service import (
    _GraphProtectionFailed,
    _GraphProtectionResult,
    _GraphProtectionSucceeded,
    _RedactProtectionService,
)
from anonymizer.engine.private_row_verification import (
    _InvocationRowVerifier,
)

_LIMITS = _AccountingLimits(max_datums=4, max_datum_bytes=64, max_graph_bytes=128)


def _graph(*texts: str) -> _ProtectionGraph:
    return _trivial_graph(tuple(_TextDatum(_DatumId(f"datum-{index}"), text) for index, text in enumerate(texts)))


def _plan(*texts: str) -> _AccountingPlan:
    compiled = _compile_accounting_plan(_graph(*texts), limits=_LIMITS)
    assert isinstance(compiled, _AccountingPlan)
    return compiled


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

    class _StaticBackend:
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
            detected = dataframe.assign(**{name: [value] for name, value in row.items()})
            verifier.freeze_accepted_detections(detected)
            final = verifier.finish(detected)
            return _PandasExecutionResult(
                dataframe=final,
                failed_records=[],
                terminal_outcomes=verifier.take_terminal_outcomes(),
                result_row_tokens=verifier.take_result_order(),
            )

    service = _RedactProtectionService(_AccountingGraphRuntime(_StaticBackend()))
    plan = service.admit(graph, limits=_LIMITS)
    assert isinstance(plan, _AccountingPlan)
    result = service.protect(
        plan,
        invocation=_CompiledInvocation.compile(AnonymizerConfig(replace=Redact()), model_selection),
    )
    return graph, result


def test_trivial_graph_is_immutable_and_preserves_datum_order() -> None:
    graph = _graph("first", "second")

    assert [datum.text for datum in graph.datums] == ["first", "second"]
    with pytest.raises(FrozenInstanceError):
        setattr(graph.datums[0], "text", "changed")
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(graph)
    assert "first" not in repr(graph)


def test_graph_runtime_lowers_only_text_and_preserves_graph_identity(
    stub_slim_model_selection: ModelSelection,
) -> None:
    backend = _SuccessfulBackend()
    graph = _graph("first", "second")
    plan = _compile_accounting_plan(graph, limits=_LIMITS)
    assert isinstance(plan, _AccountingPlan)
    result = _AccountingGraphRuntime(backend).run(
        plan,
        invocation=_CompiledInvocation.compile(AnonymizerConfig(replace=Redact()), stub_slim_model_selection),
        data_summary=None,
        preview_num_records=None,
        hydrate=lambda datum, row: (datum.id, row[COL_TEXT]),
    )

    assert tuple(datum.id.value for datum in result.plan.datums) == ("datum-0", "datum-1")
    assert backend.frame is not None
    assert list(backend.frame.columns) == [COL_TEXT, "__anonymizer_private_row_correlation__"]
    assert all(datum.id.value not in backend.frame.to_string() for datum in graph.datums)


def test_graph_runtime_rejects_rows_swapped_after_verification(
    stub_slim_model_selection: ModelSelection,
) -> None:
    class _PostVerificationSwapBackend:
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
            detected = dataframe.assign(**{COL_FINAL_ENTITIES: [{"entities": []}, {"entities": []}]})
            verifier.freeze_accepted_detections(detected)
            final = verifier.finish(detected)
            return _PandasExecutionResult(
                dataframe=final.iloc[::-1].reset_index(drop=True),
                failed_records=[],
                terminal_outcomes=verifier.take_terminal_outcomes(),
                result_row_tokens=verifier.take_result_order(),
            )

    execution = _AccountingGraphRuntime(_PostVerificationSwapBackend()).run(
        _plan("first", "second"),
        invocation=_CompiledInvocation.compile(AnonymizerConfig(replace=Redact()), stub_slim_model_selection),
        data_summary=None,
        preview_num_records=None,
        hydrate=lambda datum, row: (datum.id.value, row[COL_TEXT]),
    )

    assert isinstance(execution.accounting.invocation, _InvocationInconsistent)
    assert all(isinstance(group, _GroupWithheld) for group in execution.accounting.groups)


def test_graph_runtime_accounts_backend_failure_as_lost_without_content(
    stub_slim_model_selection: ModelSelection,
) -> None:
    secret = "backend-secret@example.test"

    class _FailingBackend:
        def run(self, *_args: Any, **_kwargs: Any) -> _PandasExecutionResult:
            raise RuntimeError(secret)

    execution = _AccountingGraphRuntime(_FailingBackend()).run(
        _plan("input-secret@example.test"),
        invocation=_CompiledInvocation.compile(AnonymizerConfig(replace=Redact()), stub_slim_model_selection),
        data_summary=None,
        preview_num_records=None,
        hydrate=lambda datum, row: (datum.id, row[COL_TEXT]),
    )

    assert isinstance(execution.accounting.invocation, _InvocationLost)
    assert secret not in repr(execution)


def test_graph_runtime_rejects_raw_graph_before_backend_effects(
    stub_slim_model_selection: ModelSelection,
) -> None:
    class _NeverRunsBackend:
        def run(self, *_args: Any, **_kwargs: Any) -> _PandasExecutionResult:
            raise AssertionError("raw graphs must not reach the backend")

    with pytest.raises(TypeError, match="private accounting plan"):
        _AccountingGraphRuntime(_NeverRunsBackend()).run(
            cast(_AccountingPlan, _graph("uncompiled")),
            invocation=_CompiledInvocation.compile(AnonymizerConfig(replace=Redact()), stub_slim_model_selection),
            data_summary=None,
            preview_num_records=None,
            hydrate=lambda datum, row: (datum.id, row[COL_TEXT]),
        )


def test_release_reconciles_reordered_terminal_outcomes_by_private_token(
    stub_slim_model_selection: ModelSelection,
) -> None:
    graph = _graph("first", "second")

    class _ReorderedBackend:
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
            detected = dataframe.assign(
                **{
                    COL_REPLACED_TEXT: dataframe[COL_TEXT],
                    COL_FINAL_ENTITIES: [{"entities": []} for _ in range(len(dataframe))],
                    COL_REPLACEMENT_APPLICATION: [
                        {
                            "targeted_span_count": 0,
                            "applied_span_count": 0,
                            "skipped_span_count": 0,
                            "skipped_span_label_counts": {},
                        }
                        for _ in range(len(dataframe))
                    ],
                }
            )
            verifier.freeze_accepted_detections(detected)
            final = verifier.finish(detected).iloc[::-1].reset_index(drop=True)
            tokens = tuple(reversed(verifier.take_result_order()))
            return _PandasExecutionResult(
                dataframe=final,
                failed_records=[],
                terminal_outcomes=tuple(reversed(verifier.take_terminal_outcomes())),
                result_row_tokens=tokens,
            )

    service = _RedactProtectionService(_AccountingGraphRuntime(_ReorderedBackend()))
    plan = service.admit(graph, limits=_LIMITS)
    assert isinstance(plan, _AccountingPlan)
    result = service.protect(
        plan,
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
