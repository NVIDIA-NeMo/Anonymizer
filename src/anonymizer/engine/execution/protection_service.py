# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-neutral release service for the first trivial-graph profile."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from anonymizer.engine.constants import COL_FINAL_ENTITIES, COL_REPLACED_TEXT
from anonymizer.engine.execution.graph import _DatumId, _GraphLimits
from anonymizer.engine.execution.graph_runtime import _GraphExecutionResult
from anonymizer.engine.execution.invocation import _CompiledInvocation
from anonymizer.engine.private_row_verification import _TerminalOutcome

if TYPE_CHECKING:
    import pandas as pd


class _PrivateProtectionValue:
    def __repr__(self) -> str:
        return f"<private {type(self).__name__.strip('_').replace('_', ' ').lower()}>"

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private protection results are not serializable")


@dataclass(frozen=True, slots=True, repr=False)
class _GraphProtectionSucceeded(_PrivateProtectionValue):
    datum_id: _DatumId
    output: str
    applied: bool


@dataclass(frozen=True, slots=True, repr=False)
class _GraphProtectionFailed(_PrivateProtectionValue):
    datum_id: _DatumId
    stage: str
    scope: str


_GraphProtectionOutcome = _GraphProtectionSucceeded | _GraphProtectionFailed


@dataclass(frozen=True, slots=True, repr=False)
class _GraphProtectionResult(_PrivateProtectionValue):
    outcomes: tuple[_GraphProtectionOutcome, ...]


class _GraphRuntimeBackend(Protocol):
    def run(
        self,
        graph: object,
        *,
        limits: _GraphLimits,
        invocation: _CompiledInvocation,
        data_summary: str | None,
        preview_num_records: int | None,
    ) -> _GraphExecutionResult: ...


class _TrivialRedactProtectionService:
    """Protect and release independently scoped text datums without source types."""

    def __init__(self, runtime: _GraphRuntimeBackend) -> None:
        self._runtime = runtime

    def protect(
        self,
        graph: object,
        *,
        limits: _GraphLimits,
        invocation: _CompiledInvocation,
    ) -> _GraphProtectionResult:
        execution = self._runtime.run(
            graph,
            limits=limits,
            invocation=invocation,
            data_summary=None,
            preview_num_records=None,
        )
        try:
            return self._release(execution)
        except Exception:
            return self._fail_all(execution.datum_ids)

    def _release(self, execution: _GraphExecutionResult) -> _GraphProtectionResult:
        dataframe_result = execution.dataframe_result
        if dataframe_result.failed_records:
            return self._fail_all(execution.datum_ids)
        terminal_by_token = _index_terminal_outcomes(execution)
        if terminal_by_token is None:
            return self._fail_all(execution.datum_ids)
        row_by_token = _index_success_rows(execution, terminal_by_token)
        if row_by_token is None:
            return self._fail_all(execution.datum_ids)
        outcomes: list[_GraphProtectionOutcome] = []
        graph_outcomes = zip(
            execution.datum_ids,
            execution.input_texts,
            execution.datum_row_tokens,
            strict=True,
        )
        for datum_id, input_text, token in graph_outcomes:
            status = terminal_by_token[token]
            if status is not _TerminalOutcome.SUCCESS:
                outcomes.append(_GraphProtectionFailed(datum_id, "pipeline", "datum"))
                continue
            row = row_by_token[token]
            output = row[COL_REPLACED_TEXT]
            if not isinstance(output, str):
                return self._fail_all(execution.datum_ids)
            valid_entities, has_detections = _accepted_detection_state(row[COL_FINAL_ENTITIES])
            if not valid_entities:
                return self._fail_all(execution.datum_ids)
            if has_detections and not _redact_release_passed(row[COL_FINAL_ENTITIES], output):
                outcomes.append(_GraphProtectionFailed(datum_id, "release", "datum"))
                continue
            if not has_detections and output != input_text:
                outcomes.append(_GraphProtectionFailed(datum_id, "release", "datum"))
                continue
            outcomes.append(_GraphProtectionSucceeded(datum_id, output, applied=has_detections))
        return _GraphProtectionResult(tuple(outcomes))

    @staticmethod
    def _fail_all(datum_ids: tuple[_DatumId, ...]) -> _GraphProtectionResult:
        return _GraphProtectionResult(
            tuple(_GraphProtectionFailed(datum_id, "pipeline", "invocation") for datum_id in datum_ids)
        )


def _index_terminal_outcomes(execution: _GraphExecutionResult) -> dict[str, _TerminalOutcome] | None:
    datum_row_tokens = execution.datum_row_tokens
    if len(datum_row_tokens) != len(execution.datum_ids) or len(set(datum_row_tokens)) != len(datum_row_tokens):
        return None
    terminal_by_token: dict[str, _TerminalOutcome] = {}
    for token, status in execution.dataframe_result.terminal_outcomes:
        if not isinstance(token, str) or not isinstance(status, _TerminalOutcome) or token in terminal_by_token:
            return None
        terminal_by_token[token] = status
    return terminal_by_token if set(terminal_by_token) == set(datum_row_tokens) else None


def _index_success_rows(
    execution: _GraphExecutionResult,
    terminal_by_token: dict[str, _TerminalOutcome],
) -> dict[str, pd.Series] | None:
    dataframe_result = execution.dataframe_result
    row_by_token = {
        token: row
        for token, (_, row) in zip(
            dataframe_result.result_row_tokens,
            dataframe_result.dataframe.iterrows(),
            strict=True,
        )
    }
    successful_tokens = {token for token, status in terminal_by_token.items() if status is _TerminalOutcome.SUCCESS}
    if len(row_by_token) != len(dataframe_result.result_row_tokens) or set(row_by_token) != successful_tokens:
        return None
    return row_by_token


def _accepted_detection_state(value: object) -> tuple[bool, bool]:
    if isinstance(value, dict):
        if "entities" not in value:
            return False, False
        entities = value["entities"]
    else:
        entities = getattr(value, "entities", None)
    if not isinstance(entities, (list, tuple)):
        return False, False
    return True, bool(entities)


def _redact_release_passed(value: object, output: str) -> bool:
    """Require every accepted entity value to be absent from released text."""
    if isinstance(value, dict):
        entities = value.get("entities", [])
    else:
        entities = getattr(value, "entities", [])
    for entity in entities:
        raw = entity.get("value") if isinstance(entity, dict) else getattr(entity, "value", None)
        if not isinstance(raw, str) or not raw or raw in output:
            return False
    return True
