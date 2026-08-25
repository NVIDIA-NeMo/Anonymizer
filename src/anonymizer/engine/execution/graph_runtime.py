# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute admitted accounting graphs through the pandas backend."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeGuard, TypeVar, assert_never

import pandas as pd

from anonymizer.engine.constants import COL_TEXT
from anonymizer.engine.execution.accounting_admission import (
    _AccountingAdmissionCode,
)
from anonymizer.engine.execution.accounting_evidence import (
    _Dispatch,
    _FailureRecord,
    _SuccessRecord,
    _TerminalRecord,
)
from anonymizer.engine.execution.accounting_ledger import _AccountingLedger
from anonymizer.engine.execution.accounting_outcomes import _AccountingResult, _CauseCode
from anonymizer.engine.execution.accounting_plan import _AccountingPlan, _is_admitted_accounting_plan
from anonymizer.engine.execution.graph import _DatumId, _TextDatum
from anonymizer.engine.execution.invocation import _CompiledInvocation
from anonymizer.engine.execution.pandas_runtime import _PandasExecutionResult
from anonymizer.engine.ndd.adapter import FailedRecord, _FailedRowEvidence
from anonymizer.engine.private_row_verification import (
    PrivateRowVerificationError,
    _InvocationRowVerifier,
    _TerminalOutcome,
)

T = TypeVar("T")


class _FrameExecutionBackend(Protocol):
    """Private effect boundary implemented by the current pandas runtime."""

    def run(
        self,
        dataframe: pd.DataFrame,
        *,
        invocation: _CompiledInvocation,
        data_summary: str | None,
        preview_num_records: int | None,
        verifier: _InvocationRowVerifier,
    ) -> _PandasExecutionResult: ...


class _AccountingGraphAdmissionError(TypeError):
    def __init__(self, code: _AccountingAdmissionCode) -> None:
        self.code = code
        super().__init__("private accounting plan required")

    def __repr__(self) -> str:
        return "<private accounting graph error>"


@dataclass(frozen=True, slots=True, repr=False)
class _AccountingGraphExecution(Generic[T]):
    plan: _AccountingPlan
    accounting: _AccountingResult[T]
    failed_records: tuple[FailedRecord, ...]

    def __repr__(self) -> str:
        return "<private accounting graph execution>"


class _AccountingGraphRuntime:
    """Schedule a compiled task DAG through bounded pandas frontiers."""

    def __init__(self, backend: _FrameExecutionBackend) -> None:
        self._backend = backend

    def run(
        self,
        plan: _AccountingPlan,
        *,
        invocation: _CompiledInvocation,
        data_summary: str | None,
        preview_num_records: int | None,
        hydrate: Callable[[_TextDatum, pd.Series], T],
        datum_release_predicate: Callable[[_DatumId, T], bool] = lambda _datum_id, _candidate: True,
        group_release_predicate: Callable[[tuple[tuple[_DatumId, T], ...]], bool] = lambda _outputs: True,
    ) -> _AccountingGraphExecution[T]:
        if preview_num_records is not None:
            raise _AccountingGraphAdmissionError(_AccountingAdmissionCode.UNSUPPORTED_TASK_CARDINALITY)
        if not _is_admitted_accounting_plan(plan):
            raise _AccountingGraphAdmissionError(_AccountingAdmissionCode.MALFORMED_GRAPH)
        ledger: _AccountingLedger[T] = _AccountingLedger(
            plan,
            datum_release_predicate=datum_release_predicate,
        )
        ledger.open()
        failed_records: list[FailedRecord] = []
        datum_by_id = {datum.id: datum for datum in plan.datums}
        while ready := ledger.ready_tasks():
            dispatches = tuple(ledger.dispatch(task) for task in ready)
            frame = pd.DataFrame({COL_TEXT: [datum_by_id[task.datum_id].text for task in ready]})
            correlations = tuple(dispatch.row_token.value for dispatch in dispatches)
            verifier = _InvocationRowVerifier(frame, correlations=correlations)
            bound = verifier.bind(frame)
            try:
                result = self._backend.run(
                    bound,
                    invocation=invocation,
                    data_summary=data_summary,
                    preview_num_records=None,
                    verifier=verifier,
                )
            except KeyboardInterrupt:
                ledger.request_cancellation()
                verifier.abort(cancelled=True)
                ledger.reconcile(dispatches, (), trusted_run_record=False)
                raise
            except PrivateRowVerificationError:
                verifier.abort(cancelled=False)
                ledger.mark_inconsistent(_CauseCode.CONTRADICTORY)
                break
            except Exception:
                verifier.abort(cancelled=False)
                ledger.reconcile(dispatches, (), trusted_run_record=False)
                break
            if not _is_well_formed_result(result):
                verifier.abort(cancelled=False)
                ledger.mark_inconsistent(_CauseCode.CONTRADICTORY)
                break
            try:
                verifier.verify_returned_rows(result.dataframe, result.result_row_tokens)
            except PrivateRowVerificationError:
                ledger.mark_inconsistent(_CauseCode.CONTRADICTORY)
                break
            failed_records.extend(result.failed_records)
            self._reconcile_frontier(
                ledger,
                plan,
                dispatches,
                result,
                hydrate,
            )
        accounting = ledger.finish(
            group_release_predicate=group_release_predicate,
        )
        return _AccountingGraphExecution(plan, accounting, tuple(failed_records))

    @staticmethod
    def _reconcile_frontier(
        ledger: _AccountingLedger[T],
        plan: _AccountingPlan,
        dispatches: tuple[_Dispatch, ...],
        result: _PandasExecutionResult,
        hydrate: Callable[[_TextDatum, pd.Series], T],
    ) -> None:
        dispatch_by_token = {dispatch.row_token.value: dispatch for dispatch in dispatches}
        terminal_tokens = tuple(token for token, _status in result.terminal_outcomes)
        result_tokens = result.result_row_tokens
        successful_tokens = tuple(
            token for token, status in result.terminal_outcomes if status is _TerminalOutcome.SUCCESS
        )
        failed_tokens = tuple(token for token, status in result.terminal_outcomes if status is _TerminalOutcome.FAILED)
        failure_evidence = result.failed_row_evidence
        failure_tokens = tuple(evidence.row_token for evidence in failure_evidence)
        trusted_stop_tokens = result.trusted_stop_tokens
        if (
            len(dispatch_by_token) != len(dispatches)
            or len(set(terminal_tokens)) != len(terminal_tokens)
            or len(set(result_tokens)) != len(result_tokens)
            or not set(terminal_tokens).issubset(dispatch_by_token)
            or not set(result_tokens).issubset(dispatch_by_token)
            or not set(result_tokens).issubset(successful_tokens)
            or len(set(failure_tokens)) != len(failure_tokens)
            or not set(failure_tokens).issubset(failed_tokens)
            or tuple(evidence.record for evidence in failure_evidence) != tuple(result.failed_records)
            or len(set(trusted_stop_tokens)) != len(trusted_stop_tokens)
            or not set(trusted_stop_tokens).issubset(
                token for token, status in result.terminal_outcomes if status is _TerminalOutcome.CANCELLED
            )
        ):
            ledger.mark_inconsistent(_CauseCode.FOREIGN)
            return
        try:
            row_by_token = {
                token: row for token, (_index, row) in zip(result_tokens, result.dataframe.iterrows(), strict=True)
            }
        except ValueError:
            ledger.mark_inconsistent(_CauseCode.CONTRADICTORY)
            return
        datum_by_id = {datum.id: datum for datum in plan.datums}
        records: list[_TerminalRecord[T]] = []
        lost_dispatches: list[_Dispatch] = []
        if trusted_stop_tokens:
            ledger.request_cancellation()
        for token, status in result.terminal_outcomes:
            dispatch = dispatch_by_token[token]
            match status:
                case _TerminalOutcome.SUCCESS:
                    try:
                        candidate = hydrate(datum_by_id[dispatch.task.datum_id], row_by_token[token])
                    except Exception:
                        records.append(_FailureRecord(dispatch))
                    else:
                        records.append(_SuccessRecord(dispatch, candidate))
                case _TerminalOutcome.FAILED:
                    records.append(_FailureRecord(dispatch))
                case _TerminalOutcome.CANCELLED:
                    if token in trusted_stop_tokens:
                        ledger.acknowledge_stop(dispatch)
                    else:
                        lost_dispatches.append(dispatch)
                case unreachable:
                    assert_never(unreachable)
        for dispatch in lost_dispatches:
            ledger.mark_transport_lost(dispatch)
        closed_tokens = {*trusted_stop_tokens, *(dispatch.row_token.value for dispatch in lost_dispatches)}
        reconciled = tuple(dispatch for dispatch in dispatches if dispatch.row_token.value not in closed_tokens)
        ledger.reconcile(reconciled, tuple(records), trusted_run_record=True)


def _is_well_formed_result(value: object) -> TypeGuard[_PandasExecutionResult]:
    return (
        isinstance(value, _PandasExecutionResult)
        and isinstance(value.dataframe, pd.DataFrame)
        and isinstance(value.failed_records, list)
        and all(isinstance(record, FailedRecord) for record in value.failed_records)
        and isinstance(value.terminal_outcomes, tuple)
        and all(
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], str)
            and bool(item[0])
            and isinstance(item[1], _TerminalOutcome)
            for item in value.terminal_outcomes
        )
        and isinstance(value.result_row_tokens, tuple)
        and all(isinstance(token, str) and token for token in value.result_row_tokens)
        and isinstance(value.failed_row_evidence, tuple)
        and all(
            isinstance(evidence, _FailedRowEvidence)
            and isinstance(evidence.row_token, str)
            and bool(evidence.row_token)
            and isinstance(evidence.record, FailedRecord)
            for evidence in value.failed_row_evidence
        )
        and isinstance(value.trusted_stop_tokens, tuple)
        and all(isinstance(token, str) and token for token in value.trusted_stop_tokens)
    )
