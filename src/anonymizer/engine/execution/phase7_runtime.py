# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private Phase 7 lifecycle coordinator and release-qualified projection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol, cast

from anonymizer.engine.execution.accounting_ledger import _AccountingLedger, _EvidenceAcceptance
from anonymizer.engine.execution.accounting_outcomes import (
    _AccountingResult,
    _CauseCode,
    _GroupReleased,
    _TaskBlocked,
    _TaskCancelled,
    _TaskFailed,
    _TaskInconsistent,
    _TaskLost,
    _TaskSucceeded,
)
from anonymizer.engine.execution.accounting_plan import (
    _TaskKey,
)
from anonymizer.engine.execution.phase6_plan import _is_admitted_phase6_plan, _Phase6Plan
from anonymizer.engine.execution.phase6_runtime import (
    _is_admitted_phase6_execution,
    _is_admitted_substitute_handoff,
    _Phase6Execution,
)
from anonymizer.engine.execution.phase7_admission import _is_admitted_phase7_plan, _Phase7Plan, _ScopeManifest
from anonymizer.engine.execution.phase7_application import (
    _AppliedDatum,
    _AppliedScope,
    _apply_substitute_patches,
    _materialize_substitute_patches,
)
from anonymizer.engine.execution.phase7_contract import _is_admitted_phase7_contract, _Phase7StableSubstituteContract
from anonymizer.engine.execution.phase7_ndd_backend import _Phase7NddResult, _Phase7NddStatus
from anonymizer.engine.execution.phase7_validation import _validate_scope_bundle, _ValidatedBundle


class _PrivatePhase7RuntimeValue:
    def __repr__(self) -> str:
        return f"<private {type(self).__name__.strip('_').replace('_', ' ').lower()}>"

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private Phase 7 runtime values are not serializable")


class _ScopePlanState(str, Enum):
    PLANNED = "planned"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"
    LOST = "lost"
    INCONSISTENT = "inconsistent"


@dataclass(frozen=True, slots=True, repr=False)
class _Phase7PlanReceipt(_PrivatePhase7RuntimeValue):
    """Content-free proof that an immutable bundle passed private validation."""


@dataclass(frozen=True, slots=True, repr=False)
class _Phase7ScopeOutcome(_PrivatePhase7RuntimeValue):
    """Content-free terminal reduction for exactly one declared scope."""

    state: _ScopePlanState


@dataclass(frozen=True, slots=True, repr=False)
class _Phase7CleanupAttestation(_PrivatePhase7RuntimeValue):
    """The frozen pre-release cleanup evidence required by the P7 contract."""

    version: str
    verified: bool
    active_reservation_count: int
    backend_workframe_reference_count: int
    ledger_mutation_closed: bool
    provisional_bundle_reference_count: int
    provisional_values_observable: bool
    # Issued by this runtime immediately before finalization.  It is an
    # invocation-private capability, not a printable/backend-derived ID.
    cleanup_identity: object | None = None


@dataclass(frozen=True, slots=True, repr=False)
class _Phase7Phase4Evidence(_PrivatePhase7RuntimeValue):
    """Private evidence handoff; Phase 4 remains the release authority."""

    scopes: tuple[_Phase7ScopeOutcome, ...]
    accounting: _AccountingResult[object]
    cleanup: _Phase7CleanupAttestation
    global_embargo: bool


@dataclass(frozen=True, slots=True, repr=False)
class _Phase7Execution(_PrivatePhase7RuntimeValue):
    scopes: tuple[_Phase7ScopeOutcome, ...]
    cleanup: _Phase7CleanupAttestation
    phase4: _Phase7Phase4Evidence
    released: tuple[_AppliedDatum, ...] = ()


class _Phase7EffectBackend(Protocol):
    def propose_scope(
        self, manifest: object, handoffs: object, contract: object, dispatch: object
    ) -> _Phase7NddResult: ...

    def close(self) -> None: ...

    def discard_values(self) -> None: ...

    def cleanup_attestation(self, cleanup_identity: object) -> object: ...


class _Phase7RuntimeAdmissionError(TypeError):
    def __init__(self) -> None:
        super().__init__("admitted private Phase 6 and Phase 7 plans and compatible backend required")

    def __repr__(self) -> str:
        return "<private Phase 7 runtime admission error>"


class _Phase7Runtime:
    """Plan one immutable bundle per ready scope and attest cleanup before handoff."""

    def __init__(
        self,
        backend: _Phase7EffectBackend,
        *,
        cancellation_requested: Callable[[], bool] | None = None,
        trusted_stop_receipt_verified: Callable[[object, object], bool] | None = None,
    ) -> None:
        self._backend = backend
        self._cancellation_requested = cancellation_requested
        # The execution host owns this verifier.  It is deliberately separate
        # from the backend that can only report an aborted candidate attempt.
        self._trusted_stop_receipt_verified = trusted_stop_receipt_verified

    def run(
        self,
        phase6: _Phase6Plan,
        phase6_execution: _Phase6Execution,
        plan: _Phase7Plan,
        contract: _Phase7StableSubstituteContract,
    ) -> _Phase7Execution:
        self._preflight(phase6, phase6_execution, plan, contract)
        # The compiler, not this coordinator, owns the single Phase 4 task
        # plan.  In particular scope capabilities are never reconstructed
        # from a manifest or declaration order at execution time.
        task_by_scope = dict(zip((manifest.id for manifest in plan.manifests), plan.scope_tasks, strict=True))
        if not _has_exact_phase6_prefix(phase6, phase6_execution, plan):
            raise _Phase7RuntimeAdmissionError
        ledger: _AccountingLedger[object] = _AccountingLedger(plan.accounting)
        ledger.open()
        # Phase 7 extends, rather than replaces, Phase 6's compiler-expanded
        # Phase 4 plan.  These exact terminal records are immutable input.
        ledger.import_terminal_outcomes(phase6_execution.accounting.tasks)
        applied_by_scope: dict[object, tuple[_AppliedDatum, ...]] = {}
        try:
            for manifest in plan.manifests:
                task = task_by_scope[manifest.id]
                # Cancellation is an invocation event, not backend status.
                # Observe it before a scope becomes dispatchable so it cannot
                # manufacture an attempt; observe it again after a synchronous
                # backend return before accepting a candidate.
                if self._cancelled():
                    ledger.request_cancellation()
                    break
                if not _scope_has_terminal_phase6_evidence(manifest, phase6, phase6_execution):
                    ledger.mark_task_blocked(task)
                    continue
                applied = self._plan_scope(ledger, task, manifest, phase6_execution.handoffs, contract)
                if applied is not None:
                    applied_by_scope[manifest.id] = applied
        finally:
            ledger.seal_mutation()
            cleanup = self._cleanup()
            if not cleanup.verified:
                ledger.record_cleanup_unconfirmed_after_seal()
        applied_by_datum = {datum.datum_id: datum for datums in applied_by_scope.values() for datum in datums}
        accounting = ledger.finish(datum_release_predicate=lambda datum_id, _candidate: datum_id in applied_by_datum)
        frozen = tuple(_scope_outcome(accounting, task_by_scope[manifest.id]) for manifest in plan.manifests)
        embargo = not cleanup.verified or any(
            outcome.state in {_ScopePlanState.INCONSISTENT, _ScopePlanState.LOST, _ScopePlanState.CANCELLED}
            for outcome in frozen
        )
        phase4 = _Phase7Phase4Evidence(frozen, accounting, cleanup, embargo)
        released_ids = {
            datum_id
            for group in accounting.groups
            if isinstance(group, _GroupReleased)
            for datum_id, _candidate in group.outputs
        }
        released = tuple(
            applied_by_datum[datum.id]
            for datum in plan.accounting.datums
            if datum.id in released_ids and datum.id in applied_by_datum
        )
        # Cleanup retired every provisional bundle before this result crosses
        # the owner boundary. Only release-qualified protected text remains.
        return _Phase7Execution(frozen, cleanup, phase4, released)

    def _preflight(
        self,
        phase6: object,
        phase6_execution: object,
        plan: object,
        contract: object,
    ) -> None:
        if (
            not isinstance(phase6, _Phase6Plan)
            or not _is_admitted_phase6_plan(phase6)
            or not _is_admitted_phase6_execution(phase6_execution, phase6)
            or not isinstance(plan, _Phase7Plan)
            or not _is_admitted_phase7_plan(plan)
            or not isinstance(contract, _Phase7StableSubstituteContract)
            or not _is_admitted_phase7_contract(contract)
            or not callable(getattr(self._backend, "propose_scope", None))
            or not callable(getattr(self._backend, "close", None))
            or not callable(getattr(self._backend, "discard_values", None))
            or not callable(getattr(self._backend, "cleanup_attestation", None))
        ):
            raise _Phase7RuntimeAdmissionError

    def _plan_scope(
        self,
        ledger: _AccountingLedger[object],
        task: _TaskKey,
        manifest: _ScopeManifest,
        handoffs: tuple[object, ...],
        contract: _Phase7StableSubstituteContract,
    ) -> tuple[_AppliedDatum, ...] | None:
        if not manifest.slots:
            bundle = _validate_scope_bundle(manifest, handoffs, (), contract)
            if isinstance(bundle, _ValidatedBundle):
                applied = _apply_bundle(bundle)
                if applied is not None:
                    ledger.mark_task_succeeded(task, _Phase7PlanReceipt())
                    return applied
            ledger.mark_task_failed(task)
            return None
        dispatch = ledger.dispatch(task)
        try:
            result = self._backend.propose_scope(manifest, handoffs, contract, dispatch)
        except Exception:
            ledger.mark_transport_lost(dispatch)
            return None
        if self._cancelled():
            # Cancellation is only a request.  Once dispatch occurred, a
            # returned candidate does not independently prove that execution
            # stopped, so it is stale and the attempt must remain lost rather
            # than fabricating a trusted stop acknowledgement.
            ledger.request_cancellation()
            ledger.mark_transport_lost(dispatch)
            return None
        if not isinstance(result, _Phase7NddResult):
            ledger.mark_inconsistent(_CauseCode.CONTRADICTORY)
            return None
        if result.status is _Phase7NddStatus.CANDIDATE:
            bundle = _validate_scope_bundle(manifest, handoffs, result.assignments, contract)
            if not isinstance(bundle, _ValidatedBundle):
                ledger.accept_failure(dispatch)
                return None
            applied = _apply_bundle(bundle)
            if applied is None:
                ledger.accept_failure(dispatch)
                return None
            if ledger.accept_success(dispatch, _Phase7PlanReceipt()) is _EvidenceAcceptance.ACCEPTED:
                return applied
            ledger.mark_inconsistent(_CauseCode.CONTRADICTORY)
            return None
        if result.status is _Phase7NddStatus.TASK_FAILED:
            ledger.accept_failure(dispatch)
        elif result.status is _Phase7NddStatus.ABORTED:
            # An abort report and a dispatch echo are both backend assertions,
            # not proof that work stopped.  Only an independently verified
            # receipt bound to this exact dispatch can acknowledge stop.
            ledger.request_cancellation()
            if self._trusted_stop_verified(result.trusted_stop_receipt, dispatch):
                ledger.acknowledge_stop(dispatch)
            else:
                ledger.mark_transport_lost(dispatch)
        elif result.status is _Phase7NddStatus.POISONED:
            ledger.mark_transport_lost(dispatch)
        else:
            ledger.mark_inconsistent(_CauseCode.CONTRADICTORY)
        return None

    def _cleanup(self) -> _Phase7CleanupAttestation:
        """Perform and attest the lifecycle actions this owner actually observed.

        The backend supplies closure evidence only after the coordinator has
        observed both mandatory retirement operations complete.
        """
        try:
            self._backend.close()
            self._backend.discard_values()
        except Exception:
            return _Phase7CleanupAttestation("phase7-cleanup-attestation/v1", False, 0, 0, False, 0, False)
        cleanup_identity = object()
        try:
            evidence = self._backend.cleanup_attestation(cleanup_identity)
        except Exception:
            evidence = None
        if not _is_verified_cleanup_attestation(evidence, cleanup_identity):
            return _Phase7CleanupAttestation("phase7-cleanup-attestation/v1", False, 0, 0, False, 0, False)
        return cast(_Phase7CleanupAttestation, evidence)

    def _cancelled(self) -> bool:
        callback = self._cancellation_requested
        if callback is None:
            return False
        try:
            return callback() is True
        except Exception:
            # An unreadable cancellation source is ambiguous lifecycle
            # evidence, so fail closed through the same cancellation embargo.
            return True

    def _trusted_stop_verified(self, receipt: object | None, dispatch: object) -> bool:
        verifier = self._trusted_stop_receipt_verified
        if receipt is None or verifier is None:
            return False
        try:
            return verifier(receipt, dispatch) is True
        except Exception:
            return False


def _is_verified_cleanup_attestation(value: object, cleanup_identity: object) -> bool:
    """Reject missing or contradictory backend closure evidence before release."""
    return (
        isinstance(value, _Phase7CleanupAttestation)
        and value.version == "phase7-cleanup-attestation/v1"
        and value.verified
        and value.active_reservation_count == 0
        and value.backend_workframe_reference_count == 0
        and value.ledger_mutation_closed
        and value.provisional_bundle_reference_count == 0
        and not value.provisional_values_observable
        and value.cleanup_identity is cleanup_identity
    )


def _apply_bundle(bundle: _ValidatedBundle) -> tuple[_AppliedDatum, ...] | None:
    """Apply once, then retain only protected datum outputs for release."""
    patches = _materialize_substitute_patches(bundle)
    if not isinstance(patches, tuple):
        return None
    applied = _apply_substitute_patches(bundle, patches)
    if not isinstance(applied, _AppliedScope):
        return None
    return tuple(applied.datums)


def _scope_has_terminal_phase6_evidence(
    manifest: _ScopeManifest,
    phase6: _Phase6Plan,
    execution: _Phase6Execution,
) -> bool:
    handoffs = execution.handoffs
    if not isinstance(handoffs, tuple) or not all(_is_admitted_substitute_handoff(item, phase6) for item in handoffs):
        return False
    task_by_key = {outcome.task: outcome for outcome in execution.accounting.tasks}
    expected = tuple(
        task for task in phase6.accounting.tasks if getattr(task.subject, "datum_id", None) in manifest.members
    )
    # A non-success Phase 6 terminal record is a local prerequisite failure;
    # it never becomes a new planner attempt.  Missing records are equally
    # non-admissible and remain withheld by the Phase 4 reduction.
    if len(task_by_key) != len(execution.accounting.tasks) or any(
        not isinstance(task_by_key.get(task), _TaskSucceeded) for task in expected
    ):
        return False
    covered = {datum_id for handoff in handoffs for datum_id in handoff.terminal_evidence.datum_ids}
    return set(manifest.members) <= covered


def _has_exact_phase6_prefix(
    phase6: _Phase6Plan,
    execution: _Phase6Execution,
    plan: _Phase7Plan,
) -> bool:
    """Bind imported terminals to the compiler-expanded Phase 4 prefix.

    Task-key membership is insufficient: a forged/reordered expansion could
    otherwise import a plausible subset of Phase 6 while retaining Phase 7
    scope capabilities.  The admitted execution is already sealed to
    ``phase6``; this check binds that exact terminal sequence to the prefix of
    the later compiler-issued plan before the ledger is opened.
    """
    outcomes = execution.accounting.tasks
    prefix = plan.accounting.tasks[: len(outcomes)]
    return (
        len(outcomes) == len(phase6.accounting.tasks)
        and tuple(outcome.task for outcome in outcomes) == phase6.accounting.tasks
        and prefix == phase6.accounting.tasks
        and plan.accounting.tasks[len(outcomes) :] == plan.scope_tasks
    )


def _scope_outcome(accounting: _AccountingResult[object], task: _TaskKey) -> _Phase7ScopeOutcome:
    outcome = next(item for item in accounting.tasks if item.task == task)
    if isinstance(outcome, _TaskSucceeded):
        return _Phase7ScopeOutcome(_ScopePlanState.PLANNED)
    if isinstance(outcome, _TaskBlocked):
        return _Phase7ScopeOutcome(_ScopePlanState.BLOCKED)
    if isinstance(outcome, _TaskFailed):
        return _Phase7ScopeOutcome(_ScopePlanState.FAILED)
    if isinstance(outcome, _TaskCancelled):
        return _Phase7ScopeOutcome(_ScopePlanState.CANCELLED)
    if isinstance(outcome, _TaskLost):
        return _Phase7ScopeOutcome(_ScopePlanState.LOST)
    if isinstance(outcome, _TaskInconsistent):
        return _Phase7ScopeOutcome(_ScopePlanState.INCONSISTENT)
    return _Phase7ScopeOutcome(_ScopePlanState.INCONSISTENT)
