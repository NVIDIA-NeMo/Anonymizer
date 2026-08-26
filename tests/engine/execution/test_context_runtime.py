# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, cast

import pandas as pd
import pytest

import anonymizer.engine.constants as execution_constants
import anonymizer.engine.execution.graph_runtime as graph_runtime
from anonymizer.config.anonymizer_config import AnonymizerConfig
from anonymizer.config.models import ModelSelection
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import (
    COL_CONTEXT_BINDING_ID,
    COL_CONTEXT_ORDINAL,
    COL_CONTEXT_OWNER_WORK_ID,
    COL_CONTEXT_TEXT,
    COL_FINAL_ENTITIES,
    COL_TARGET_WORK_ID,
    COL_TEXT,
)
from anonymizer.engine.execution.accounting_ledger import _AccountingLedger
from anonymizer.engine.execution.accounting_outcomes import (
    _GroupReleased,
    _GroupWithheld,
    _InvocationCancelled,
    _InvocationCompleted,
    _InvocationFailed,
    _InvocationInconsistent,
    _InvocationLost,
    _TaskCancelled,
    _TaskInconsistent,
    _TaskLost,
    _TaskSucceeded,
)
from anonymizer.engine.execution.accounting_plan import _AccountingLimits, _AccountingPlan, _TaskKey
from anonymizer.engine.execution.context_admission import (
    _compile_context_plan,
    _ContextAdmissionCode,
    _ContextPlan,
    _ContextRejected,
)
from anonymizer.engine.execution.context_contract import (
    _BackendArtifactClass,
    _ContextBackendCapability,
    _ContextExecutionContract,
    _ContextLimits,
    _ContextOrdering,
    _ContextProfile,
    _ContextSchemaVersion,
    _RetentionPosture,
)
from anonymizer.engine.execution.context_workframes import (
    _BackendArtifactId,
    _BackendClosureAttestation,
    _ContextBindingEvidence,
    _ContextWorkframes,
    _lower_context_workframes,
    _make_context_binding_evidence,
    _WorkframeClosedError,
    _WorkframeConstructionError,
)
from anonymizer.engine.execution.graph import (
    _AtomicGroup,
    _CoherenceScope,
    _ContextScope,
    _DatumDependency,
    _DatumId,
    _DatumPurpose,
    _ProtectionGraph,
    _TextDatum,
)
from anonymizer.engine.execution.graph_runtime import (
    _AccountingGraphExecution,
    _AccountingGraphRuntime,
    _ContextGraphAdmissionError,
    _ExecutionFrontier,
    _PreparedRuntimePlan,
)
from anonymizer.engine.execution.invocation import _CompiledInvocation
from anonymizer.engine.execution.pandas_runtime import _PandasExecutionResult
from anonymizer.engine.execution.protection_service import _RedactProtectionService
from anonymizer.engine.private_row_verification import _InvocationRowVerifier, _TerminalOutcome
from tests.engine.execution.phase5_reference_model import (
    ReferenceAdmission,
    ReferenceCase,
    ReferenceEventKind,
    ReferenceInvocation,
    ReferenceLimits,
    ReferenceResult,
    ReferenceScope,
    evaluate,
    reference_cases,
    schedule_for,
)


@dataclass(frozen=True)
class _BackendSchedule:
    """Translate one frozen oracle trace into private backend evidence."""

    binding_evidence: str = "exact"
    cleanup: str = "verified"
    task_schedule: str = "success"
    terminal_evidence: _TerminalEvidence | None = None

    @classmethod
    def from_reference(cls, reference: ReferenceCase) -> _BackendSchedule:
        corruption = next(
            (event.outcome for event in reference.events if event.kind is ReferenceEventKind.BINDING_CORRUPTION),
            "exact",
        )
        constructed = {
            event.subject for event in reference.events if event.kind is ReferenceEventKind.BINDING_CONSTRUCTION
        }
        consumed = {event.subject for event in reference.events if event.kind is ReferenceEventKind.BINDING_CONSUMPTION}
        if constructed - consumed:
            corruption = "missing"
        cleanup = tuple(
            event
            for event in reference.events
            if event.kind in {ReferenceEventKind.CLEANUP_PRIMARY, ReferenceEventKind.CLEANUP_COMPETING}
        )
        cleanup_evidence = cleanup[0].outcome if len(cleanup) == 1 else "unconfirmed"
        if any(event.kind is ReferenceEventKind.TRUSTED_STOP for event in reference.events):
            task_schedule = "trusted_stop"
        elif any(event.kind is ReferenceEventKind.TRANSPORT_LOSS for event in reference.events):
            task_schedule = "transport_loss"
        elif any(event.kind is ReferenceEventKind.CANCELLATION for event in reference.events):
            task_schedule = (
                "cancel_after_terminal"
                if any(event.kind is ReferenceEventKind.TASK_TERMINAL for event in reference.events)
                else "cancel_before_dispatch"
            )
        elif any(event.kind is ReferenceEventKind.TASK_CORRUPTION for event in reference.events):
            task_schedule = "corrupt"
        elif {event.subject for event in reference.events if event.kind is ReferenceEventKind.TASK_TERMINAL} != {
            identifier for identifier, _text in reference.targets
        }:
            task_schedule = "missing"
        elif any(event.kind is ReferenceEventKind.TASK_TERMINAL for event in reference.events):
            task_schedule = "success"
        else:
            task_schedule = "missing"
        terminal_evidence = next(
            (
                _TerminalEvidence(event.outcome)
                for event in reference.events
                if event.kind is ReferenceEventKind.TASK_CORRUPTION
            ),
            None,
        )
        return cls(corruption, cleanup_evidence, task_schedule, terminal_evidence)

    @classmethod
    def from_legacy_mode(cls, mode: str) -> _BackendSchedule:
        """Keep focused fixture modes on the same evidence boundary."""
        match mode:
            case "missing" | "duplicate" | "wrong_ordinal":
                return cls(binding_evidence=mode)
            case "cross":
                return cls(binding_evidence="cross_target")
            case "unconfirmed_cleanup":
                return cls(cleanup="missing")
            case "failed_cleanup":
                return cls(cleanup="failed")
            case "trusted_stop" | "transport_loss":
                return cls(task_schedule=mode)
            case _:
                return cls()


class _ContextBackend:
    def __init__(
        self,
        capability: _ContextBackendCapability | None,
        *,
        evidence_mode: str = "exact",
        schedule: _BackendSchedule | None = None,
    ) -> None:
        self._capability = capability
        self.evidence_mode = evidence_mode
        self.schedule = schedule
        self.calls = 0
        self.target_frame: pd.DataFrame | None = None
        self.context_frame: pd.DataFrame | None = None

    def context_capability(self) -> _ContextBackendCapability | None:
        return self._capability

    def run_context(
        self,
        dataframe: pd.DataFrame,
        *,
        context_dataframe: pd.DataFrame,
        artifact_id: _BackendArtifactId,
        invocation: _CompiledInvocation,
        data_summary: str | None,
        preview_num_records: int | None,
        verifier: _InvocationRowVerifier,
    ) -> _PandasExecutionResult:
        del invocation, data_summary, preview_num_records
        self.calls += 1
        if self.evidence_mode == "permuted":
            dataframe = dataframe.iloc[::-1].copy()
            dataframe.index = pd.Index([None] * len(dataframe))
            context_dataframe = context_dataframe.iloc[::-1].copy()
            context_dataframe.index = pd.Index(["duplicate"] * len(context_dataframe))
        self.target_frame = dataframe.copy()
        self.context_frame = context_dataframe.copy()
        schedule = self.schedule or _BackendSchedule.from_legacy_mode(self.evidence_mode)
        if schedule.task_schedule in {"trusted_stop", "transport_loss"}:
            verifier.abort(cancelled=True)
            final = dataframe.iloc[0:0].drop(columns=[COL_TARGET_WORK_ID], errors="ignore")
        else:
            detected = dataframe.assign(**{COL_FINAL_ENTITIES: [{"entities": []} for _ in range(len(dataframe))]})
            verifier.freeze_accepted_detections(detected)
            final = verifier.finish(detected.iloc[1:] if schedule.task_schedule == "missing" else detected)
        evidence = tuple(
            _make_context_binding_evidence(
                row[COL_CONTEXT_BINDING_ID],
                row[COL_CONTEXT_OWNER_WORK_ID],
                row[COL_CONTEXT_ORDINAL],
                row[COL_CONTEXT_TEXT],
            )
            for _index, row in context_dataframe.iterrows()
        )
        if schedule.binding_evidence == "missing" and evidence:
            evidence = evidence[1:]
        elif schedule.binding_evidence == "duplicate" and evidence:
            evidence = (evidence[0], *evidence)
        elif schedule.binding_evidence == "wrong_ordinal" and evidence:
            evidence = (replace(evidence[0], ordinal=evidence[0].ordinal + 1), *evidence[1:])
        elif schedule.binding_evidence in {"cross_target", "foreign", "contradictory"}:
            evidence = cast(tuple[_ContextBindingEvidence, ...], (object(),))
        terminal_outcomes = verifier.take_terminal_outcomes()
        result_row_tokens = verifier.take_result_order()
        if schedule.task_schedule == "missing":
            terminal_outcomes = terminal_outcomes[1:]
        elif schedule.terminal_evidence is not None:
            terminal_outcomes = _corrupt_terminal_evidence(terminal_outcomes, schedule.terminal_evidence)
        closures = (
            ()
            if schedule.cleanup != "verified"
            else (
                _BackendClosureAttestation(
                    artifact_id,
                    _BackendArtifactClass.CONTEXT_REQUEST,
                    True,
                ),
            )
        )
        if schedule.cleanup == "failed":
            closures = (_BackendClosureAttestation(artifact_id, _BackendArtifactClass.CONTEXT_REQUEST, False),)
        return _PandasExecutionResult(
            dataframe=final,
            failed_records=[],
            terminal_outcomes=terminal_outcomes,
            result_row_tokens=result_row_tokens,
            trusted_stop_tokens=(
                tuple(dataframe[COL_TARGET_WORK_ID]) if schedule.task_schedule == "trusted_stop" else ()
            ),
            context_binding_evidence=evidence,
            closure_attestations=closures,
        )

    def run(self, *_args: object, **_kwargs: object) -> _PandasExecutionResult:
        raise AssertionError("context plans must not fall back to independent-row execution")


def _corrupt_terminal_evidence(
    terminal_outcomes: tuple[tuple[str, _TerminalOutcome], ...],
    evidence: _TerminalEvidence,
) -> tuple[tuple[str, _TerminalOutcome], ...]:
    """Emit one frozen terminal-evidence class at the private backend boundary."""
    first = terminal_outcomes[0]
    match evidence:
        case _TerminalEvidence.DUPLICATE | _TerminalEvidence.CONTRADICTORY:
            return (*terminal_outcomes, first)
        case _TerminalEvidence.FOREIGN:
            return (*terminal_outcomes, ("foreign-terminal", first[1]))
        case _TerminalEvidence.STALE:
            return (*terminal_outcomes, ("stale-terminal", first[1]))
        case _TerminalEvidence.CROSS_TARGET:
            return (*terminal_outcomes, ("cross-target-terminal", first[1]))
        case _TerminalEvidence.PLAN_MISMATCH:
            return (*terminal_outcomes, ("mismatched-plan-terminal", first[1]))


class _TerminalEvidence(str, Enum):
    """Frozen malformed terminal-evidence classes accepted by the corpus adapter."""

    DUPLICATE = "duplicate"
    FOREIGN = "foreign"
    STALE = "stale"
    CROSS_TARGET = "cross_target"
    PLAN_MISMATCH = "plan_mismatch"
    CONTRADICTORY = "contradictory"


class _ScheduledContextRuntime(_AccountingGraphRuntime):
    """Inject cancellation linearization at the test runtime boundary only."""

    def __init__(self, backend: _ContextBackend, schedule: _BackendSchedule | None) -> None:
        super().__init__(backend)
        self._schedule = schedule

    def _build_frontier(
        self,
        ledger: _AccountingLedger[tuple[str, str]],
        prepared: _PreparedRuntimePlan,
        ready: tuple[_TaskKey, ...],
        datum_by_id: dict[_DatumId, _TextDatum],
    ) -> _ExecutionFrontier | None:
        if self._schedule is not None and self._schedule.task_schedule == "cancel_before_dispatch":
            ledger.request_cancellation()
        return super()._build_frontier(ledger, prepared, ready, datum_by_id)

    def _accept_frontier(
        self,
        ledger: _AccountingLedger[tuple[str, str]],
        plan: _AccountingPlan,
        frontier: _ExecutionFrontier,
        value: object,
        hydrate: Callable[[_TextDatum, pd.Series], tuple[str, str]],
    ) -> bool:
        accepted = super()._accept_frontier(ledger, plan, frontier, value, hydrate)
        if self._schedule is not None and self._schedule.task_schedule == "cancel_after_terminal":
            ledger.request_cancellation()
        return accepted


@pytest.mark.parametrize(
    "mutation",
    [
        lambda capability: replace(capability, retention=_RetentionPosture.ENABLED),
        lambda capability: replace(capability, profile=cast(_ContextProfile, "future")),
        lambda capability: replace(capability, schema_version=cast(_ContextSchemaVersion, "future")),
        lambda capability: replace(capability, ordering=cast(_ContextOrdering, "implicit")),
        lambda capability: replace(capability, artifact_classes=()),
        lambda capability: replace(
            capability,
            limits=replace(capability.limits, max_context_bytes_per_target=1),
        ),
    ],
)
def test_runtime_rechecks_complete_capability_before_open_or_dispatch(
    mutation: Callable[[_ContextBackendCapability], _ContextBackendCapability],
    stub_slim_model_selection: ModelSelection,
) -> None:
    plan, capability = _plan()
    backend = _ContextBackend(mutation(capability))

    with pytest.raises(_ContextGraphAdmissionError) as raised:
        _run(plan, backend, stub_slim_model_selection)

    assert raised.value.code is _ContextAdmissionCode.BACKEND_INCOMPATIBLE
    assert backend.calls == 0


def test_runtime_rejects_a_capable_backend_without_context_execution_before_open(
    stub_slim_model_selection: ModelSelection,
) -> None:
    plan, capability = _plan()

    class _CapabilityOnlyBackend:
        def context_capability(self) -> _ContextBackendCapability:
            return capability

        def run(self, *_args: object, **_kwargs: object) -> _PandasExecutionResult:
            raise AssertionError("context plans must not use the row backend")

    with pytest.raises(_ContextGraphAdmissionError) as raised:
        _run(plan, cast(_ContextBackend, _CapabilityOnlyBackend()), stub_slim_model_selection)

    assert raised.value.code is _ContextAdmissionCode.BACKEND_INCOMPATIBLE


@pytest.mark.parametrize(
    "runner",
    [None, object(), lambda dataframe: dataframe],
)
def test_runtime_rejects_unusable_context_runner_before_ledger_or_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    stub_slim_model_selection: ModelSelection,
    runner: object,
) -> None:
    plan, capability = _plan()
    backend = _ContextBackend(capability)
    monkeypatch.setattr(backend, "run_context", runner, raising=False)

    def reject_open(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("ledger must not open")

    def reject_lowering(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("workframes must not be constructed")

    monkeypatch.setattr(_AccountingLedger, "open", reject_open)
    monkeypatch.setattr(graph_runtime, "_lower_context_workframes", reject_lowering)

    with pytest.raises(_ContextGraphAdmissionError) as raised:
        _run(plan, backend, stub_slim_model_selection)

    assert raised.value.code is _ContextAdmissionCode.BACKEND_INCOMPATIBLE


def test_runtime_recheck_handles_a_raising_capability_snapshot_before_open(
    monkeypatch: pytest.MonkeyPatch,
    stub_slim_model_selection: ModelSelection,
) -> None:
    plan, capability = _plan()
    backend = _ContextBackend(capability)

    def raising_snapshot() -> _ContextBackendCapability:
        raise RuntimeError("backend unavailable")

    def reject_open(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("ledger must not open")

    monkeypatch.setattr(backend, "context_capability", raising_snapshot)
    monkeypatch.setattr(_AccountingLedger, "open", reject_open)

    with pytest.raises(_ContextGraphAdmissionError) as raised:
        _run(plan, backend, stub_slim_model_selection)

    assert raised.value.code is _ContextAdmissionCode.BACKEND_INCOMPATIBLE


def test_context_target_frame_carries_exact_private_task_and_attempt_identities(
    monkeypatch: pytest.MonkeyPatch,
    stub_slim_model_selection: ModelSelection,
) -> None:
    plan, capability = _plan()
    backend = _ContextBackend(capability)
    dispatched = []
    real_dispatch = _AccountingLedger.dispatch_batch

    def capture_dispatch(
        ledger: _AccountingLedger[object],
        tasks: tuple[_TaskKey, ...],
        *,
        row_token_values: tuple[str, ...],
    ):
        values = real_dispatch(ledger, tasks, row_token_values=row_token_values)
        dispatched.extend(values)
        return values

    monkeypatch.setattr(_AccountingLedger, "dispatch_batch", capture_dispatch)

    execution = _run(plan, backend, stub_slim_model_selection)

    assert isinstance(execution.accounting.invocation, _InvocationCompleted)
    assert backend.target_frame is not None
    task_column = getattr(execution_constants, "COL_TASK_ID")
    attempt_column = getattr(execution_constants, "COL_ATTEMPT_ID")
    assert {task_column, attempt_column}.issubset(backend.target_frame.columns)
    assert tuple(backend.target_frame[task_column]) == tuple(dispatch.task for dispatch in dispatched)
    assert tuple(backend.target_frame[attempt_column]) == tuple(dispatch.attempt_id for dispatch in dispatched)
    rendered = backend.target_frame.to_string()
    assert "target-a" not in rendered
    assert "target-b" not in rendered


def test_context_construction_failure_occurs_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    stub_slim_model_selection: ModelSelection,
) -> None:
    plan, capability = _plan()
    backend = _ContextBackend(capability)

    def fail_lowering(*_args: object, **_kwargs: object) -> None:
        raise _WorkframeConstructionError

    def reject_dispatch(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dispatch must not occur after construction failed")

    monkeypatch.setattr(graph_runtime, "_lower_context_workframes", fail_lowering)
    monkeypatch.setattr(_AccountingLedger, "dispatch_batch", reject_dispatch)

    execution = _run(plan, backend, stub_slim_model_selection)

    assert all(not isinstance(task, _TaskSucceeded) for task in execution.accounting.tasks)
    assert backend.calls == 0


def test_dispatch_failure_closes_constructed_workframes(
    monkeypatch: pytest.MonkeyPatch,
    stub_slim_model_selection: ModelSelection,
) -> None:
    plan, capability = _plan()
    backend = _ContextBackend(capability)
    lowered: list[_ContextWorkframes] = []
    real_lowering = graph_runtime._lower_context_workframes

    def capture_lowering(*args: Any, **kwargs: Any) -> _ContextWorkframes:
        frames = real_lowering(*args, **kwargs)
        lowered.append(frames)
        return frames

    def fail_dispatch(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("dispatch unavailable")

    monkeypatch.setattr(graph_runtime, "_lower_context_workframes", capture_lowering)
    monkeypatch.setattr(_AccountingLedger, "dispatch_batch", fail_dispatch)

    execution = _run(plan, backend, stub_slim_model_selection)

    assert all(not isinstance(task, _TaskSucceeded) for task in execution.accounting.tasks)
    assert backend.calls == 0
    assert len(lowered) == 1
    assert lowered[0].target_frame.empty
    assert lowered[0].context_frame.empty
    rendered = repr(vars(lowered[0]))
    assert all(value not in rendered for value in ("alpha", "beta", "gamma", "target-a", "context-c"))
    with pytest.raises(_WorkframeClosedError):
        lowered[0].artifact_id


def test_discard_failure_after_uncommitted_dispatch_is_contained_and_embargoed(
    monkeypatch: pytest.MonkeyPatch,
    stub_slim_model_selection: ModelSelection,
) -> None:
    plan, capability = _plan()
    backend = _ContextBackend(capability)
    lowered: list[_ContextWorkframes] = []
    real_lowering = graph_runtime._lower_context_workframes

    def capture_lowering(*args: Any, **kwargs: Any) -> _ContextWorkframes:
        frames = real_lowering(*args, **kwargs)
        lowered.append(frames)
        return frames

    def fail_dispatch(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("dispatch unavailable")

    def fail_discard(self: _ContextWorkframes) -> None:
        raise RuntimeError("discard unavailable")

    monkeypatch.setattr(graph_runtime, "_lower_context_workframes", capture_lowering)
    monkeypatch.setattr(_AccountingLedger, "dispatch_batch", fail_dispatch)
    monkeypatch.setattr(_ContextWorkframes, "discard_before_dispatch", fail_discard)

    execution = _run(plan, backend, stub_slim_model_selection)

    assert isinstance(execution.accounting.invocation, _InvocationFailed)
    assert "cleanup_failed" in _invocation_causes(execution)
    assert all(isinstance(group, _GroupWithheld) for group in execution.accounting.groups)
    assert backend.calls == 0
    assert len(lowered) == 1
    assert lowered[0].target_frame.empty
    assert lowered[0].context_frame.empty
    with pytest.raises(_WorkframeClosedError):
        lowered[0].artifact_id


def test_cancellation_winning_before_atomic_dispatch_closes_frames(
    monkeypatch: pytest.MonkeyPatch,
    stub_slim_model_selection: ModelSelection,
) -> None:
    plan, capability = _plan()
    backend = _ContextBackend(capability)
    lowered: list[_ContextWorkframes] = []
    real_lowering = graph_runtime._lower_context_workframes
    real_dispatch = _AccountingLedger.dispatch_batch

    def capture_lowering(*args: Any, **kwargs: Any) -> _ContextWorkframes:
        frames = real_lowering(*args, **kwargs)
        lowered.append(frames)
        return frames

    def cancel_before_dispatch(
        ledger: _AccountingLedger[object],
        tasks: tuple[_TaskKey, ...],
        *,
        row_token_values: tuple[str, ...],
    ):
        ledger.request_cancellation()
        return real_dispatch(ledger, tasks, row_token_values=row_token_values)

    monkeypatch.setattr(graph_runtime, "_lower_context_workframes", capture_lowering)
    monkeypatch.setattr(_AccountingLedger, "dispatch_batch", cancel_before_dispatch)

    execution = _run(plan, backend, stub_slim_model_selection)

    assert isinstance(execution.accounting.invocation, _InvocationCancelled)
    assert backend.calls == 0
    assert len(lowered) == 1
    assert lowered[0].target_frame.empty
    assert lowered[0].context_frame.empty


def test_late_cancellation_preserves_private_success_but_embargos_release() -> None:
    plan, _capability = _plan()
    ledger: _AccountingLedger[str] = _AccountingLedger(plan.accounting)
    ledger.open()
    ready = ledger.ready_tasks()
    frames = _lower_context_workframes(plan, ready)
    correlations = tuple(work_id.value for work_id in frames.target_work_ids())
    dispatches = ledger.dispatch_batch(ready, row_token_values=correlations)
    frames.bind_dispatches(dispatches)
    for dispatch in dispatches:
        ledger.accept_success(dispatch, dispatch.task.datum_id.value)
    evidence = tuple(
        _make_context_binding_evidence(
            row[COL_CONTEXT_BINDING_ID],
            row[COL_CONTEXT_OWNER_WORK_ID],
            row[COL_CONTEXT_ORDINAL],
            row[COL_CONTEXT_TEXT],
        )
        for _index, row in frames.context_frame.iterrows()
    )
    assert frames.reconcile(evidence).status.value == "verified"
    artifact_id = frames.artifact_id
    assert (
        frames.close(
            (_BackendClosureAttestation(artifact_id, _BackendArtifactClass.CONTEXT_REQUEST, True),)
        ).status.value
        == "verified"
    )

    ledger.request_cancellation()
    result = ledger.finish()

    assert all(isinstance(task, _TaskSucceeded) for task in result.tasks)
    assert isinstance(result.invocation, _InvocationCancelled)
    assert all(isinstance(group, _GroupWithheld) for group in result.groups)


def test_protection_service_preflights_context_with_the_selected_backend() -> None:
    plan, capability = _plan()
    backend = _ContextBackend(capability)
    service = _RedactProtectionService(_AccountingGraphRuntime(backend))

    admitted = service.admit_context(
        _graph(),
        accounting_limits=_AccountingLimits(8, 64, 256),
        contract=plan.contract,
    )

    assert isinstance(admitted, _ContextPlan)


def test_context_runtime_reconciles_exact_bindings_and_releases_targets_only(
    stub_slim_model_selection: ModelSelection,
) -> None:
    plan, capability = _plan()
    backend = _ContextBackend(capability)

    execution = _run(plan, backend, stub_slim_model_selection)

    assert isinstance(execution.accounting.invocation, _InvocationCompleted)
    assert all(isinstance(group, _GroupReleased) for group in execution.accounting.groups)
    assert backend.calls == 1
    assert backend.target_frame is not None
    assert backend.context_frame is not None
    assert COL_TEXT in backend.target_frame
    assert COL_TEXT not in backend.context_frame
    assert set(backend.context_frame[COL_CONTEXT_ORDINAL]) == {0, 1}
    assert "context-c" not in backend.context_frame.to_string()


def test_context_runtime_is_invariant_to_equal_text_row_order_and_duplicate_indices(
    stub_slim_model_selection: ModelSelection,
) -> None:
    plan, capability = _plan(equal_text=True)
    backend = _ContextBackend(capability, evidence_mode="permuted")

    execution = _run(plan, backend, stub_slim_model_selection)

    assert all(isinstance(group, _GroupReleased) for group in execution.accounting.groups)
    assert backend.target_frame is not None
    assert backend.context_frame is not None
    assert backend.target_frame.index.tolist() == [None, None]
    assert backend.context_frame.index.tolist() == ["duplicate", "duplicate", "duplicate"]
    assert backend.target_frame[COL_TEXT].tolist() == ["same-text", "same-text"]
    assert backend.context_frame[COL_CONTEXT_TEXT].tolist() == ["same-text", "same-text", "same-text"]


def test_missing_binding_evidence_withholds_only_its_owner_group(
    stub_slim_model_selection: ModelSelection,
) -> None:
    plan, capability = _plan()

    execution = _run(plan, _ContextBackend(capability, evidence_mode="missing"), stub_slim_model_selection)

    assert isinstance(execution.accounting.groups[0], _GroupWithheld)
    assert isinstance(execution.accounting.groups[1], _GroupReleased)


@pytest.mark.parametrize("mode", ["cross", "unconfirmed_cleanup"])
def test_global_binding_or_cleanup_uncertainty_embargos_every_group(
    mode: str,
    stub_slim_model_selection: ModelSelection,
) -> None:
    plan, capability = _plan()

    execution = _run(plan, _ContextBackend(capability, evidence_mode=mode), stub_slim_model_selection)

    assert isinstance(execution.accounting.invocation, _InvocationInconsistent)
    assert all(isinstance(group, _GroupWithheld) for group in execution.accounting.groups)


def test_definitive_cleanup_failure_preserves_private_tasks_but_fails_public_release(
    stub_slim_model_selection: ModelSelection,
) -> None:
    plan, capability = _plan()

    execution = _run(plan, _ContextBackend(capability, evidence_mode="failed_cleanup"), stub_slim_model_selection)

    assert isinstance(execution.accounting.invocation, _InvocationFailed)
    assert all(isinstance(task, _TaskSucceeded) for task in execution.accounting.tasks)
    assert all(isinstance(group, _GroupWithheld) for group in execution.accounting.groups)


def test_production_admission_matches_every_frozen_reference_case() -> None:
    """Pair every frozen oracle dimension with the production admission boundary."""
    for reference in reference_cases():
        expected = evaluate(reference)
        actual = _compile_reference_case(reference)

        assert isinstance(actual, _ContextPlan) is (expected.admission is ReferenceAdmission.ADMITTED), (
            reference.case_id
        )


def _runtime_reference_cases() -> tuple[ReferenceCase, ...]:
    """Cover every frozen graph, schedule, and runtime-capability class."""
    return tuple(reference_cases())


@pytest.mark.parametrize("reference", _runtime_reference_cases(), ids=lambda reference: reference.case_id)
def test_production_runtime_and_release_match_every_admitted_frozen_graph_and_runtime_capability(
    reference: ReferenceCase,
    stub_slim_model_selection: ModelSelection,
) -> None:
    """Pair every admitted frozen graph and runtime capability with runtime release."""
    expected = evaluate(reference)
    if expected.admission is not ReferenceAdmission.ADMITTED:
        return

    plan = _compile_reference_case(reference)
    assert isinstance(plan, _ContextPlan)
    backend = _ContextBackend(
        _reference_runtime_capability(reference, plan.contract),
        schedule=_BackendSchedule.from_reference(reference),
    )
    if expected.invocation is ReferenceInvocation.NOT_OPENED:
        with pytest.raises(_ContextGraphAdmissionError) as raised:
            _run(plan, backend, stub_slim_model_selection)
        assert raised.value.code is _ContextAdmissionCode.BACKEND_INCOMPATIBLE
        assert backend.calls == 0
        return

    execution = _run(plan, backend, stub_slim_model_selection, schedule=backend.schedule)

    _assert_execution_matches_reference(execution, plan, expected, reference=reference, backend=backend)


def _compile_reference_case(reference: ReferenceCase) -> _ContextPlan | _ContextRejected:
    graph = _reference_graph(reference)
    contract = _reference_contract(reference)
    capability = _reference_capability(reference, contract)
    return _compile_context_plan(
        graph,
        accounting_limits=_AccountingLimits(
            len(graph.datums),
            reference.limits.datum_bytes,
            sum(len(datum.text.encode()) for datum in graph.datums),
            reference.limits.id_bytes,
        ),
        contract=contract,
        capability=capability,
    )


def _reference_graph(reference: ReferenceCase) -> _ProtectionGraph:
    target_datums = tuple(
        _TextDatum(_DatumId(identifier), text, _DatumPurpose.TARGET) for identifier, text in reference.targets
    )
    context_datums = tuple(
        _TextDatum(_DatumId(identifier), text, _DatumPurpose.CONTEXT_ONLY)
        for identifier, text in reference.context_only
    )
    return _ProtectionGraph(
        datums=(*target_datums, *context_datums),
        links=(),
        context_scopes=tuple(
            _ContextScope(_DatumId(scope.target), tuple(_DatumId(member) for member in scope.context))
            for scope in reference.scopes
        ),
        coherence_scopes=tuple(_CoherenceScope((_DatumId(identifier),)) for identifier, _text in reference.targets),
        atomic_groups=tuple(_AtomicGroup(tuple(_DatumId(member) for member in group)) for group in reference.groups),
        dependencies=tuple(
            _DatumDependency(_DatumId(before), _DatumId(after)) for before, after in reference.dependencies
        ),
    )


def _reference_contract(reference: ReferenceCase) -> _ContextExecutionContract:
    return _ContextExecutionContract(
        _ContextProfile.TARGET_CONTEXT_V1
        if reference.profile == "target-context-v1" and reference.relation == "bounded_context"
        else cast(_ContextProfile, "unsupported"),
        _ContextSchemaVersion.V1
        if reference.schema == "context-workframe-v1"
        else cast(_ContextSchemaVersion, "unsupported"),
        _ContextLimits(
            reference.limits.members,
            reference.limits.context_bytes,
            reference.limits.references,
            reference.limits.expanded_bytes,
        ),
        reference.allow_target_as_context,
        _ContextOrdering.DECLARED if reference.ordering == "declared" else cast(_ContextOrdering, "unsupported"),
        (_BackendArtifactClass.CONTEXT_REQUEST,),
    )


def _reference_capability(reference: ReferenceCase, contract: _ContextExecutionContract) -> _ContextBackendCapability:
    return _ContextBackendCapability(
        _ContextProfile.TARGET_CONTEXT_V1,
        _ContextSchemaVersion.V1,
        contract.limits,
        reference.allow_target_as_context,
        _ContextOrdering.DECLARED,
        (_BackendArtifactClass.CONTEXT_REQUEST,),
        _RetentionPosture.DISABLED if reference.preflight_capability == "compatible" else _RetentionPosture.ENABLED,
    )


def _reference_runtime_capability(
    reference: ReferenceCase,
    contract: _ContextExecutionContract,
) -> _ContextBackendCapability | None:
    capability = _reference_capability(reference, contract)
    match reference.runtime_capability:
        case "compatible":
            return capability
        case "missing":
            return None
        case "incompatible":
            return replace(capability, retention=_RetentionPosture.ENABLED)
        case "weakened":
            return replace(capability, limits=replace(capability.limits, max_expanded_frame_bytes=0))
        case "retention_enabled":
            return replace(capability, retention=_RetentionPosture.ENABLED)
        case "profile":
            return replace(capability, profile=cast(_ContextProfile, "future"))
        case "schema":
            return replace(capability, schema_version=cast(_ContextSchemaVersion, "future"))
        case "ordering":
            return replace(capability, ordering=cast(_ContextOrdering, "implicit"))
        case unexpected:
            raise AssertionError(f"unexpected frozen runtime capability: {unexpected}")


@pytest.mark.parametrize(
    ("backend_mode", "reference_evidence", "reference_cleanup", "task_schedule"),
    [
        ("exact", "exact", "verified", "success"),
        ("missing", "missing", "verified", "success"),
        ("duplicate", "duplicate", "verified", "success"),
        ("wrong_ordinal", "wrong_ordinal", "verified", "success"),
        ("cross", "cross_target", "verified", "success"),
        ("unconfirmed_cleanup", "exact", "missing", "success"),
        ("failed_cleanup", "exact", "failed", "success"),
        ("trusted_stop", "exact", "verified", "trusted_stop"),
        ("transport_loss", "exact", "verified", "transport_loss"),
    ],
)
def test_production_release_matches_the_independent_reference_model(
    backend_mode: str,
    reference_evidence: str,
    reference_cleanup: str,
    task_schedule: str,
    stub_slim_model_selection: ModelSelection,
) -> None:
    plan, capability = _plan()
    reference = ReferenceCase(
        case_id="production-pair",
        targets=(("target-a", "alpha"), ("target-b", "beta")),
        context_only=(("context-c", "gamma"),),
        scopes=(
            ReferenceScope("target-a", ("context-c", "target-b")),
            ReferenceScope("target-b", ("target-a",)),
        ),
        limits=ReferenceLimits(5, 9, 2, 9, 3, 23),
        events=(),
        groups=(("target-a",), ("target-b",)),
    )
    reference = replace(
        reference,
        events=schedule_for(
            reference,
            binding_evidence=reference_evidence,
            cleanup=reference_cleanup,
            task_schedule=task_schedule,
        ),
    )

    expected = evaluate(reference)
    actual = _run(plan, _ContextBackend(capability, evidence_mode=backend_mode), stub_slim_model_selection)

    assert expected.admission is ReferenceAdmission.ADMITTED
    assert expected.binding_count == sum(len(projection.bindings) for projection in plan.projections)
    assert expected.event_count <= expected.event_max
    _assert_execution_matches_reference(actual, plan, expected)


def _run(
    plan: _ContextPlan,
    backend: _ContextBackend,
    model_selection: ModelSelection,
    *,
    schedule: _BackendSchedule | None = None,
) -> _AccountingGraphExecution[tuple[str, str]]:
    runtime = _AccountingGraphRuntime(backend) if schedule is None else _ScheduledContextRuntime(backend, schedule)
    return runtime.run(
        plan,
        invocation=_CompiledInvocation.compile(AnonymizerConfig(replace=Redact()), model_selection),
        data_summary=None,
        preview_num_records=None,
        hydrate=lambda datum, row: (datum.id.value, row[COL_TEXT]),
    )


def _assert_execution_matches_reference(
    execution: _AccountingGraphExecution[tuple[str, str]],
    plan: _ContextPlan,
    expected: ReferenceResult,
    *,
    reference: ReferenceCase | None = None,
    backend: _ContextBackend | None = None,
) -> None:
    released = tuple(
        datum_id.value
        for group in execution.accounting.groups
        if isinstance(group, _GroupReleased)
        for datum_id, _candidate in group.outputs
    )

    assert released == expected.released
    _assert_invocation_matches_reference(execution, expected, reference)
    assert _task_outcomes(execution, plan) == _runtime_task_outcomes(expected, reference)
    if backend is not None:
        assert _runtime_cleanup_outcome(execution, backend) == expected.cleanup


def _assert_invocation_matches_reference(
    execution: _AccountingGraphExecution[tuple[str, str]],
    expected: ReferenceResult,
    reference: ReferenceCase | None,
) -> None:
    assert isinstance(execution.accounting.invocation, _InvocationCompleted) == (
        expected.invocation is ReferenceInvocation.COMPLETED
    )
    assert isinstance(execution.accounting.invocation, _InvocationInconsistent) == (
        expected.invocation is ReferenceInvocation.INCONSISTENT
    )
    assert isinstance(execution.accounting.invocation, _InvocationFailed) == (
        expected.invocation is ReferenceInvocation.FAILED
    )
    assert isinstance(execution.accounting.invocation, _InvocationCancelled) == (
        expected.invocation is ReferenceInvocation.CANCELLED
    )
    assert isinstance(execution.accounting.invocation, _InvocationLost) == (
        expected.invocation is ReferenceInvocation.LOST
    )
    expected_cause = _runtime_invocation_cause(expected, reference)
    if expected_cause != "none":
        assert expected_cause in _invocation_causes(execution)


def _runtime_task_outcomes(
    expected: ReferenceResult,
    reference: ReferenceCase | None,
) -> tuple[tuple[str, str, str], ...]:
    """Project oracle schedules onto the private runtime's terminal evidence vocabulary."""
    if reference is None:
        return expected.task_outcomes
    schedule = _BackendSchedule.from_reference(reference)
    if schedule.terminal_evidence is not None:
        return tuple((target, "inconsistent", "foreign") for target, _state, _reason in expected.task_outcomes)
    if schedule.task_schedule == "missing":
        return tuple(
            (target, state, "missing" if reason == "terminal_missing" else reason)
            for target, state, reason in expected.task_outcomes
        )
    if schedule.task_schedule == "cancel_before_dispatch":
        return tuple((target, state, "stop_acknowledged") for target, state, _reason in expected.task_outcomes)
    if schedule.binding_evidence == "wrong_ordinal":
        return tuple(
            (target, state, "contradictory" if reason == "wrong_ordinal" else reason)
            for target, state, reason in expected.task_outcomes
        )
    return expected.task_outcomes


def _runtime_invocation_cause(expected: ReferenceResult, reference: ReferenceCase | None) -> str:
    if reference is not None and _BackendSchedule.from_reference(reference).terminal_evidence is not None:
        return "foreign"
    return {"terminal_attribution_invalid": "foreign"}.get(expected.reason, expected.reason)


def _runtime_cleanup_outcome(
    execution: _AccountingGraphExecution[tuple[str, str]],
    backend: _ContextBackend,
) -> str:
    if backend.calls == 0:
        return "not_entered"
    causes = _invocation_causes(execution)
    if "cleanup_failed" in causes:
        return "failed"
    if "cleanup_unconfirmed" in causes:
        return "unconfirmed"
    return "verified"


def _invocation_causes(execution: _AccountingGraphExecution[tuple[str, str]]) -> tuple[str, ...]:
    causes = getattr(execution.accounting.invocation, "causes", ())
    return tuple(cause.code.value for cause in causes)


def _task_outcomes(
    execution: _AccountingGraphExecution[tuple[str, str]],
    plan: _ContextPlan,
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (
            datum.id.value,
            "succeeded"
            if isinstance(task, _TaskSucceeded)
            else "inconsistent"
            if isinstance(task, _TaskInconsistent)
            else "cancelled"
            if isinstance(task, _TaskCancelled)
            else "lost",
            "none"
            if isinstance(task, _TaskSucceeded)
            else "stop_acknowledged"
            if isinstance(task, _TaskCancelled)
            else next(cause.code.value for cause in task.causes)
            if isinstance(task, (_TaskInconsistent, _TaskLost))
            else "unexpected",
        )
        for datum, task in zip(plan.accounting.datums, execution.accounting.tasks, strict=True)
    )


def _plan(*, equal_text: bool = False) -> tuple[_ContextPlan, _ContextBackendCapability]:
    graph = _graph()
    if equal_text:
        graph = replace(graph, datums=tuple(replace(datum, text="same-text") for datum in graph.datums))
    limits = _ContextLimits(2, 32, 4, 128)
    contract = _ContextExecutionContract(
        _ContextProfile.TARGET_CONTEXT_V1,
        _ContextSchemaVersion.V1,
        limits,
        True,
        _ContextOrdering.DECLARED,
        (_BackendArtifactClass.CONTEXT_REQUEST,),
    )
    capability = _ContextBackendCapability(
        contract.profile,
        contract.schema_version,
        limits,
        True,
        contract.ordering,
        contract.required_artifacts,
        _RetentionPosture.DISABLED,
    )
    compiled = _compile_context_plan(
        graph,
        accounting_limits=_AccountingLimits(8, 64, 256),
        contract=contract,
        capability=capability,
    )
    assert isinstance(compiled, _ContextPlan)
    return compiled, capability


def _graph() -> _ProtectionGraph:
    target_a = _TextDatum(_DatumId("target-a"), "alpha", _DatumPurpose.TARGET)
    target_b = _TextDatum(_DatumId("target-b"), "beta", _DatumPurpose.TARGET)
    context = _TextDatum(_DatumId("context-c"), "gamma", _DatumPurpose.CONTEXT_ONLY)
    return _ProtectionGraph(
        datums=(target_a, target_b, context),
        links=(),
        context_scopes=(
            _ContextScope(target_a.id, (context.id, target_b.id)),
            _ContextScope(target_b.id, (target_a.id,)),
        ),
        coherence_scopes=(_CoherenceScope((target_a.id,)), _CoherenceScope((target_b.id,))),
        atomic_groups=(_AtomicGroup((target_a.id,)), _AtomicGroup((target_b.id,))),
    )
