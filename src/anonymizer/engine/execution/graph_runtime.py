# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute admitted accounting graphs through the pandas backend."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from inspect import Signature, signature
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
from anonymizer.engine.execution.accounting_plan import (
    _AccountingPlan,
    _DatumTaskSubject,
    _is_admitted_accounting_plan,
    _TaskKey,
)
from anonymizer.engine.execution.context_admission import (
    _ContextAdmissionCode,
    _ContextPlan,
    _is_admitted_context_plan,
)
from anonymizer.engine.execution.context_contract import (
    _capability_satisfies,
    _ContextBackendCapability,
    _snapshot_context_capability,
)
from anonymizer.engine.execution.context_observations import _observe_context_boundary
from anonymizer.engine.execution.context_workframes import (
    _BackendArtifactId,
    _BackendClosureAttestation,
    _ContextBindingFault,
    _ContextCleanupStatus,
    _ContextReconciliationStatus,
    _ContextWorkframes,
    _lower_context_workframes,
    _WorkframeConstructionError,
)
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


@dataclass(frozen=True, slots=True, repr=False)
class _ContextRunnerAdapter:
    """Validated private invocation shape for context-capable backends."""

    runner: Callable[..., object]

    def run(
        self,
        dataframe: pd.DataFrame,
        *,
        context_dataframe: pd.DataFrame,
        artifact_id: _BackendArtifactId,
        invocation: _CompiledInvocation,
        data_summary: str | None,
        preview_num_records: int | None,
        verifier: _InvocationRowVerifier,
    ) -> object:
        return self.runner(
            dataframe,
            context_dataframe=context_dataframe,
            artifact_id=artifact_id,
            invocation=invocation,
            data_summary=data_summary,
            preview_num_records=preview_num_records,
            verifier=verifier,
        )


def _adapt_context_runner(backend: object) -> _ContextRunnerAdapter | None:
    """Reject missing or incompatible private context runners before execution opens."""
    try:
        runner = getattr(backend, "run_context", None)
        if not callable(runner):
            return None
        runner_signature: Signature = signature(runner)
        runner_signature.bind(
            object(),
            context_dataframe=object(),
            artifact_id=object(),
            invocation=object(),
            data_summary=None,
            preview_num_records=None,
            verifier=object(),
        )
    except Exception:
        return None
    return _ContextRunnerAdapter(runner)


class _AccountingGraphAdmissionError(TypeError):
    def __init__(self, code: _AccountingAdmissionCode) -> None:
        self.code = code
        super().__init__("private accounting plan required")

    def __repr__(self) -> str:
        return "<private accounting graph error>"


class _ContextGraphAdmissionError(TypeError):
    def __init__(self, code: _ContextAdmissionCode) -> None:
        self.code = code
        super().__init__("compatible private context plan and backend required")

    def __repr__(self) -> str:
        return "<private context graph error>"


@dataclass(frozen=True, slots=True, repr=False)
class _AccountingGraphExecution(Generic[T]):
    plan: _AccountingPlan
    accounting: _AccountingResult[T]
    failed_records: tuple[FailedRecord, ...]

    def __repr__(self) -> str:
        return "<private accounting graph execution>"


@dataclass(frozen=True, slots=True, repr=False)
class _PreparedRuntimePlan:
    accounting: _AccountingPlan
    context: _ContextPlan | None
    context_count: int
    context_runner: _ContextRunnerAdapter | None


@dataclass(slots=True, repr=False)
class _ExecutionFrontier:
    ready: tuple[_TaskKey, ...]
    dispatches: tuple[_Dispatch, ...]
    verifier: _InvocationRowVerifier
    bound: pd.DataFrame
    workframes: _ContextWorkframes | None
    context_runner: _ContextRunnerAdapter | None


class _AccountingGraphRuntime:
    """Schedule a compiled task DAG through bounded pandas frontiers."""

    def __init__(self, backend: _FrameExecutionBackend) -> None:
        self._backend = backend

    def context_capability(self) -> _ContextBackendCapability | None:
        """Take one typed preflight snapshot from the selected backend."""
        return _snapshot_context_capability(self._backend)

    def run(
        self,
        plan: _AccountingPlan | _ContextPlan,
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
        prepared = self._prepare_runtime_plan(plan)
        ledger: _AccountingLedger[T] = _AccountingLedger(
            prepared.accounting,
            datum_release_predicate=datum_release_predicate,
        )
        ledger.open()
        failed_records: list[FailedRecord] = []
        datum_by_id = {datum.id: datum for datum in prepared.accounting.datums}
        while ready := ledger.ready_tasks():
            frontier = self._build_frontier(ledger, prepared, ready, datum_by_id)
            if frontier is None:
                continue
            result = self._invoke_frontier(
                ledger,
                frontier,
                invocation=invocation,
                data_summary=data_summary,
            )
            if result is None:
                break
            if _is_well_formed_result(result):
                failed_records.extend(result.failed_records)
            if not self._accept_frontier(ledger, prepared.accounting, frontier, result, hydrate):
                break
        if prepared.context is not None:
            with _observe_context_boundary(
                "release",
                target_count=len(prepared.accounting.datums),
                context_count=prepared.context_count,
            ):
                accounting = ledger.finish(group_release_predicate=group_release_predicate)
        else:
            accounting = ledger.finish(group_release_predicate=group_release_predicate)
        return _AccountingGraphExecution(prepared.accounting, accounting, tuple(failed_records))

    def _prepare_runtime_plan(self, plan: _AccountingPlan | _ContextPlan) -> _PreparedRuntimePlan:
        if not isinstance(plan, _ContextPlan):
            if not _is_admitted_accounting_plan(plan):
                raise _AccountingGraphAdmissionError(_AccountingAdmissionCode.MALFORMED_GRAPH)
            if any(not isinstance(task.subject, _DatumTaskSubject) for task in plan.tasks):
                raise _AccountingGraphAdmissionError(_AccountingAdmissionCode.UNSUPPORTED_TASK_CARDINALITY)
            return _PreparedRuntimePlan(plan, None, 0, None)
        if not _is_admitted_context_plan(plan):
            raise _ContextGraphAdmissionError(_ContextAdmissionCode.MALFORMED_GRAPH)
        if any(not isinstance(task.subject, _DatumTaskSubject) for task in plan.accounting.tasks):
            raise _ContextGraphAdmissionError(_ContextAdmissionCode.MALFORMED_GRAPH)
        context_count = sum(len(projection.bindings) for projection in plan.projections)
        with _observe_context_boundary(
            "capability_recheck",
            target_count=len(plan.accounting.datums),
            context_count=context_count,
        ) as observation:
            context_runner = _adapt_context_runner(self._backend)
            if not _capability_satisfies(self.context_capability(), plan.contract) or context_runner is None:
                observation.outcome = "rejected"
                observation.reason = _ContextAdmissionCode.BACKEND_INCOMPATIBLE.value
                raise _ContextGraphAdmissionError(_ContextAdmissionCode.BACKEND_INCOMPATIBLE)
        return _PreparedRuntimePlan(plan.accounting, plan, context_count, context_runner)

    def _build_frontier(
        self,
        ledger: _AccountingLedger[T],
        prepared: _PreparedRuntimePlan,
        ready: tuple[_TaskKey, ...],
        datum_by_id: dict[_DatumId, _TextDatum],
    ) -> _ExecutionFrontier | None:
        datum_subjects: list[_DatumTaskSubject] = []
        for task in ready:
            if not isinstance(task.subject, _DatumTaskSubject):
                raise _AccountingGraphAdmissionError(_AccountingAdmissionCode.UNSUPPORTED_TASK_CARDINALITY)
            datum_subjects.append(task.subject)
        if prepared.context is None:
            frame = pd.DataFrame({COL_TEXT: [datum_by_id[subject.datum_id].text for subject in datum_subjects]})
            dispatches = self._dispatch_frontier(ledger, prepared, ready)
            correlations = tuple(dispatch.row_token.value for dispatch in dispatches)
            verifier = _InvocationRowVerifier(frame, correlations=correlations)
            return _ExecutionFrontier(ready, dispatches, verifier, verifier.bind(frame), None, None)
        with _observe_context_boundary(
            "workframe_construction",
            target_count=len(ready),
            context_count=prepared.context_count,
        ) as observation:
            try:
                workframes = _lower_context_workframes(prepared.context, ready)
            except _WorkframeConstructionError:
                observation.outcome = "failed"
                observation.reason = "binding_construction_failed"
                for task in ready:
                    ledger.mark_task_failed(task)
                return None
        correlations = tuple(work_id.value for work_id in workframes.target_work_ids())
        try:
            dispatches = self._dispatch_frontier(ledger, prepared, ready, correlations=correlations)
        except Exception:
            self._discard_context_workframes(ledger, workframes, target_count=len(ready))
            for task in ready:
                ledger.mark_task_failed(task)
            return None
        try:
            workframes.bind_dispatches(dispatches)
        except Exception:
            ledger.reconcile(dispatches, (), trusted_run_record=False)
            self._close_context_workframes(
                ledger,
                workframes,
                (),
                target_count=len(ready),
            )
            return None
        frame = workframes.target_frame.loc[:, [COL_TEXT]]
        verifier = _InvocationRowVerifier(frame, correlations=correlations)
        return _ExecutionFrontier(
            ready,
            dispatches,
            verifier,
            workframes.target_frame,
            workframes,
            prepared.context_runner,
        )

    @staticmethod
    def _dispatch_frontier(
        ledger: _AccountingLedger[T],
        prepared: _PreparedRuntimePlan,
        ready: tuple[_TaskKey, ...],
        *,
        correlations: tuple[str, ...] | None = None,
    ) -> tuple[_Dispatch, ...]:
        if prepared.context is None:
            return tuple(ledger.dispatch(task) for task in ready)
        if correlations is None or len(correlations) != len(ready):
            raise _WorkframeConstructionError
        with _observe_context_boundary(
            "dispatch",
            target_count=len(ready),
            context_count=prepared.context_count,
        ):
            return ledger.dispatch_batch(ready, row_token_values=correlations)

    def _invoke_frontier(
        self,
        ledger: _AccountingLedger[T],
        frontier: _ExecutionFrontier,
        *,
        invocation: _CompiledInvocation,
        data_summary: str | None,
    ) -> object | None:
        try:
            return self._call_frontier_backend(frontier, invocation=invocation, data_summary=data_summary)
        except KeyboardInterrupt:
            ledger.request_cancellation()
            frontier.verifier.abort(cancelled=True)
            ledger.reconcile(frontier.dispatches, (), trusted_run_record=False)
            self._close_frontier_without_evidence(ledger, frontier)
            raise
        except PrivateRowVerificationError:
            frontier.verifier.abort(cancelled=False)
            ledger.mark_inconsistent(_CauseCode.CONTRADICTORY)
            self._close_frontier_without_evidence(ledger, frontier)
            return None
        except Exception:
            frontier.verifier.abort(cancelled=False)
            ledger.reconcile(frontier.dispatches, (), trusted_run_record=False)
            self._close_frontier_without_evidence(ledger, frontier)
            return None

    def _call_frontier_backend(
        self,
        frontier: _ExecutionFrontier,
        *,
        invocation: _CompiledInvocation,
        data_summary: str | None,
    ) -> object:
        if frontier.workframes is None:
            return self._backend.run(
                frontier.bound,
                invocation=invocation,
                data_summary=data_summary,
                preview_num_records=None,
                verifier=frontier.verifier,
            )
        if frontier.context_runner is None:
            raise TypeError("private context backend is unavailable")
        return self._run_context_backend(
            frontier.context_runner,
            frontier.workframes,
            frontier.bound,
            invocation=invocation,
            data_summary=data_summary,
            verifier=frontier.verifier,
        )

    def _accept_frontier(
        self,
        ledger: _AccountingLedger[T],
        plan: _AccountingPlan,
        frontier: _ExecutionFrontier,
        value: object,
        hydrate: Callable[[_TextDatum, pd.Series], T],
    ) -> bool:
        if not _is_well_formed_result(value):
            frontier.verifier.abort(cancelled=False)
            ledger.mark_inconsistent(_CauseCode.CONTRADICTORY)
            self._close_frontier_without_evidence(ledger, frontier)
            return False
        try:
            frontier.verifier.verify_returned_rows(value.dataframe, value.result_row_tokens)
        except PrivateRowVerificationError:
            ledger.mark_inconsistent(_CauseCode.CONTRADICTORY)
            self._close_frontier(ledger, frontier, value.closure_attestations)
            return False
        context_status = self._accept_context_evidence(ledger, frontier, value)
        if context_status is not _ContextReconciliationStatus.GLOBAL_INVALID:
            self._reconcile_frontier(ledger, plan, frontier.dispatches, value, hydrate)
        cleanup = self._close_frontier(ledger, frontier, value.closure_attestations)
        return cleanup is _ContextCleanupStatus.VERIFIED

    def _accept_context_evidence(
        self,
        ledger: _AccountingLedger[T],
        frontier: _ExecutionFrontier,
        result: _PandasExecutionResult,
    ) -> _ContextReconciliationStatus:
        if frontier.workframes is None:
            return _ContextReconciliationStatus.VERIFIED
        return self._reconcile_context_bindings(
            ledger,
            frontier.workframes,
            result,
            target_count=len(frontier.ready),
        )

    def _close_frontier_without_evidence(
        self,
        ledger: _AccountingLedger[T],
        frontier: _ExecutionFrontier,
    ) -> None:
        self._close_frontier(ledger, frontier, ())

    def _close_frontier(
        self,
        ledger: _AccountingLedger[T],
        frontier: _ExecutionFrontier,
        attestations: tuple[_BackendClosureAttestation, ...],
    ) -> _ContextCleanupStatus:
        if frontier.workframes is None:
            return _ContextCleanupStatus.VERIFIED
        return self._close_context_workframes(
            ledger,
            frontier.workframes,
            attestations,
            target_count=len(frontier.ready),
        )

    def _run_context_backend(
        self,
        context_runner: _ContextRunnerAdapter,
        workframes: _ContextWorkframes,
        target_frame: pd.DataFrame,
        *,
        invocation: _CompiledInvocation,
        data_summary: str | None,
        verifier: _InvocationRowVerifier,
    ) -> object:
        context_frame = workframes.context_frame
        with _observe_context_boundary(
            "backend_execution",
            target_count=len(target_frame),
            context_count=len(context_frame),
        ):
            return context_runner.run(
                target_frame,
                context_dataframe=context_frame,
                artifact_id=workframes.artifact_id,
                invocation=invocation,
                data_summary=data_summary,
                preview_num_records=None,
                verifier=verifier,
            )

    @staticmethod
    def _reconcile_context_bindings(
        ledger: _AccountingLedger[T],
        workframes: _ContextWorkframes,
        result: _PandasExecutionResult,
        *,
        target_count: int,
    ) -> _ContextReconciliationStatus:
        with _observe_context_boundary(
            "reconciliation",
            target_count=target_count,
            context_count=len(result.context_binding_evidence),
        ) as observation:
            context_reconciliation = workframes.reconcile(result.context_binding_evidence)
            observation.reconciliation = context_reconciliation.status.value
            if context_reconciliation.status is _ContextReconciliationStatus.GLOBAL_INVALID:
                observation.outcome = "inconsistent"
                observation.reason = "binding_attribution_invalid"
                ledger.mark_inconsistent(_CauseCode.CONTRADICTORY)
            elif context_reconciliation.status is _ContextReconciliationStatus.LOCAL_INVALID:
                observation.outcome = "localized"
                observation.reason = "binding_evidence_invalid"
                cause_by_fault = {
                    _ContextBindingFault.MISSING: _CauseCode.MISSING,
                    _ContextBindingFault.DUPLICATE: _CauseCode.DUPLICATE,
                    _ContextBindingFault.CONTRADICTORY: _CauseCode.CONTRADICTORY,
                }
                for task, fault in context_reconciliation.faults:
                    ledger.mark_task_inconsistent(task, cause_by_fault[fault])
            return context_reconciliation.status

    @staticmethod
    def _close_context_workframes(
        ledger: _AccountingLedger[T],
        workframes: _ContextWorkframes,
        attestations: tuple[_BackendClosureAttestation, ...],
        *,
        target_count: int,
    ) -> _ContextCleanupStatus:
        with _observe_context_boundary(
            "cleanup",
            target_count=target_count,
            context_count=len(workframes.context_frame),
        ) as observation:
            try:
                cleanup = workframes.close(attestations)
            except Exception:
                observation.outcome = "failed"
                observation.reason = _CauseCode.CLEANUP_FAILED.value
                observation.cleanup = _ContextCleanupStatus.FAILED.value
                ledger.mark_cleanup_failed()
                return _ContextCleanupStatus.FAILED
            observation.cleanup = cleanup.status.value
            if cleanup.status is _ContextCleanupStatus.FAILED:
                observation.outcome = "failed"
                observation.reason = _CauseCode.CLEANUP_FAILED.value
                ledger.mark_cleanup_failed()
            elif cleanup.status is _ContextCleanupStatus.UNCONFIRMED:
                observation.outcome = "inconsistent"
                observation.reason = _CauseCode.CLEANUP_UNCONFIRMED.value
                ledger.mark_cleanup_unconfirmed()
            return cleanup.status

    @staticmethod
    def _discard_context_workframes(
        ledger: _AccountingLedger[T],
        workframes: _ContextWorkframes,
        *,
        target_count: int,
    ) -> None:
        """Discard unopened owned frames when atomic dispatch never committed."""
        with _observe_context_boundary(
            "cleanup",
            target_count=target_count,
            context_count=len(workframes.context_frame),
        ) as observation:
            try:
                workframes.discard_before_dispatch()
            except Exception:
                observation.outcome = "failed"
                observation.reason = _CauseCode.CLEANUP_FAILED.value
                observation.cleanup = _ContextCleanupStatus.FAILED.value
                try:
                    workframes.contain_discard_failure()
                except Exception:
                    pass
                ledger.mark_cleanup_failed()
                return
            observation.cleanup = _ContextCleanupStatus.VERIFIED.value

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
                        if not isinstance(dispatch.task.subject, _DatumTaskSubject):
                            raise TypeError
                        candidate = hydrate(datum_by_id[dispatch.task.subject.datum_id], row_by_token[token])
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
        and isinstance(value.context_binding_evidence, tuple)
        and isinstance(value.closure_attestations, tuple)
    )
