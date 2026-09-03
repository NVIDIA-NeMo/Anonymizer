# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused lifecycle coverage for the private Phase 7 coordinator."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from anonymizer.engine.execution.accounting_outcomes import (
    _GroupReleased,
    _TaskCancelled,
    _TaskFailed,
    _TaskLost,
    _TaskSucceeded,
)
from anonymizer.engine.execution.accounting_plan import _DatumTaskSubject
from anonymizer.engine.execution.phase7_admission import _Phase7Plan
from anonymizer.engine.execution.phase7_application import _AppliedDatum
from anonymizer.engine.execution.phase7_contract import _load_phase7_contract, _Phase7StableSubstituteContract
from anonymizer.engine.execution.phase7_ndd_backend import _Phase7NddResult, _Phase7NddStatus
from anonymizer.engine.execution.phase7_runtime import (
    _Phase7CleanupAttestation,
    _Phase7Runtime,
    _ScopePlanState,
)
from tests.engine.execution.test_phase7_admission import _compile_phase7, _Proposal, _qualified_phase6


@dataclass
class _Backend:
    calls: int = 0
    closed: int = 0
    discarded: int = 0
    attest_cleanup: bool = True
    discard_fails: bool = False
    stale_cleanup_identity: bool = False
    trusted_stop_receipt: object | None = None
    echo_dispatch_as_stop_receipt: bool = False
    result: _Phase7NddStatus = _Phase7NddStatus.TASK_FAILED

    def propose_scope(self, manifest: object, handoffs: object, contract: object, dispatch: object) -> _Phase7NddResult:
        del manifest, handoffs, contract
        assert dispatch is not None
        self.calls += 1
        return _Phase7NddResult(
            self.result,
            trusted_stop_receipt=dispatch if self.echo_dispatch_as_stop_receipt else self.trusted_stop_receipt,
        )

    def close(self) -> None:
        self.closed += 1

    def discard_values(self) -> None:
        if self.discard_fails:
            raise RuntimeError
        self.discarded += 1

    def cleanup_attestation(self, cleanup_identity: object) -> object:
        if not self.attest_cleanup:
            return None
        identity = object() if self.stale_cleanup_identity else cleanup_identity
        return _Phase7CleanupAttestation("phase7-cleanup-attestation/v1", True, 0, 0, True, 0, False, identity)


def test_empty_manifest_is_verified_no_work_then_cleaned_without_backend_dispatch() -> None:
    phase6, _phase6_backend, execution = _qualified_phase6(("plain",), (("target-0",),), {})
    plan = _compile_phase7(phase6, execution, phase6.coherence_scopes)
    assert isinstance(plan, _Phase7Plan)
    backend = _Backend()

    contract = _load_phase7_contract()
    assert isinstance(contract, _Phase7StableSubstituteContract)
    result = _Phase7Runtime(backend).run(phase6, execution, plan, contract)

    assert tuple(outcome.state for outcome in result.scopes) == (_ScopePlanState.PLANNED,)
    assert backend.calls == 0
    assert backend.closed == 1
    assert backend.discarded == 1
    assert result.cleanup.verified
    assert result.cleanup.active_reservation_count == 0
    assert result.cleanup.provisional_bundle_reference_count == 0
    # The runtime consumes the exact compiler-issued Phase 4 task plan; it
    # does not fabricate a detached scope-only accounting shell.
    assert len(result.phase4.accounting.tasks) == len(plan.accounting.tasks)
    assert result.phase4.accounting.tasks[:-2] == execution.accounting.tasks
    assert not hasattr(result, "bundles")
    planned, applied = result.phase4.accounting.tasks[-2:]
    assert isinstance(planned, _TaskSucceeded)
    assert not hasattr(planned.candidate, "assignments")
    assert plan.scope_tasks == (planned.task,)
    assert plan.application_tasks == (applied.task,)
    assert isinstance(applied.task.subject, _DatumTaskSubject)
    assert isinstance(applied, _TaskSucceeded)
    assert isinstance(applied.candidate, _AppliedDatum)
    assert applied.candidate.output == "plain"
    assert result.released == (applied.candidate,)
    released_group = result.phase4.accounting.groups[0]
    assert isinstance(released_group, _GroupReleased)
    assert released_group.outputs == ((applied.candidate.datum_id, applied.candidate),)


def test_application_exception_fails_owned_tasks_and_still_cleans_up(monkeypatch: pytest.MonkeyPatch) -> None:
    phase6, _phase6_backend, execution = _qualified_phase6(("plain",), (("target-0",),), {})
    plan = _compile_phase7(phase6, execution, phase6.coherence_scopes)
    assert isinstance(plan, _Phase7Plan)
    backend = _Backend()
    contract = _load_phase7_contract()
    assert isinstance(contract, _Phase7StableSubstituteContract)

    def fail_application(_bundle: object) -> None:
        raise RuntimeError("private canary")

    monkeypatch.setattr("anonymizer.engine.execution.phase7_runtime._apply_bundle", fail_application)

    result = _Phase7Runtime(backend).run(phase6, execution, plan, contract)

    application = next(
        outcome for outcome in result.phase4.accounting.tasks if outcome.task == plan.application_tasks[0]
    )
    assert isinstance(application, _TaskFailed)
    assert result.released == ()
    assert backend.closed == 1
    assert backend.discarded == 1
    assert result.cleanup.verified


def test_reconstructed_phase6_execution_is_rejected_before_any_planner_dispatch() -> None:
    phase6, _phase6_backend, execution = _qualified_phase6(
        ("Alice",), (("target-0",),), {"target-0": (_Proposal("Alice", "first_name", "person"),)}
    )
    plan = _compile_phase7(phase6, execution, phase6.coherence_scopes)
    assert isinstance(plan, _Phase7Plan)
    backend = _Backend()
    incomplete = type(execution)(execution.accounting, execution.released, ())

    contract = _load_phase7_contract()
    assert isinstance(contract, _Phase7StableSubstituteContract)
    with pytest.raises(TypeError):
        _Phase7Runtime(backend).run(phase6, incomplete, plan, contract)
    assert backend.calls == 0


def test_phase6_terminals_must_match_the_exact_compiler_expanded_prefix() -> None:
    phase6, _phase6_backend, execution = _qualified_phase6(("plain",), (("target-0",),), {})
    plan = _compile_phase7(phase6, execution, phase6.coherence_scopes)
    assert isinstance(plan, _Phase7Plan)
    # This is still a sealed Phase 7-shaped value, but its Phase 4 expansion
    # no longer carries the admitted Phase 6 prefix at its front.
    malformed = replace(plan, accounting=replace(plan.accounting, tasks=plan.accounting.tasks[::-1]))
    backend = _Backend()
    contract = _load_phase7_contract()
    assert isinstance(contract, _Phase7StableSubstituteContract)

    with pytest.raises(TypeError):
        _Phase7Runtime(backend).run(phase6, execution, malformed, contract)
    assert backend.calls == 0


def test_missing_cleanup_evidence_embargoes_the_private_phase4_handoff() -> None:
    phase6, _phase6_backend, execution = _qualified_phase6(("plain",), (("target-0",),), {})
    plan = _compile_phase7(phase6, execution, phase6.coherence_scopes)
    assert isinstance(plan, _Phase7Plan)
    backend = _Backend()

    backend.discard_fails = True
    contract = _load_phase7_contract()
    assert isinstance(contract, _Phase7StableSubstituteContract)

    result = _Phase7Runtime(backend).run(phase6, execution, plan, contract)

    assert not result.cleanup.verified
    assert result.phase4.global_embargo


@pytest.mark.parametrize(
    ("backend_status", "expected", "embargo"),
    [
        (_Phase7NddStatus.ABORTED, _ScopePlanState.LOST, True),
        (_Phase7NddStatus.POISONED, _ScopePlanState.LOST, True),
        (_Phase7NddStatus.INVOCATION_INCONSISTENT, _ScopePlanState.INCONSISTENT, True),
    ],
)
def test_terminal_backend_lifecycle_evidence_is_absorbing_and_embargoed(
    backend_status: _Phase7NddStatus, expected: _ScopePlanState, embargo: bool
) -> None:
    phase6, _phase6_backend, execution = _qualified_phase6(
        ("Alice",), (("target-0",),), {"target-0": (_Proposal("Alice", "first_name", "person"),)}
    )
    plan = _compile_phase7(phase6, execution, phase6.coherence_scopes)
    assert isinstance(plan, _Phase7Plan)
    contract = _load_phase7_contract()
    assert isinstance(contract, _Phase7StableSubstituteContract)
    backend = _Backend(result=backend_status)

    result = _Phase7Runtime(backend).run(phase6, execution, plan, contract)

    assert result.scopes[0].state is expected
    assert backend.calls == 1
    assert result.phase4.global_embargo is embargo


def test_unverified_abort_after_dispatch_is_lost_without_stop_acknowledgement() -> None:
    phase6, _phase6_backend, execution = _qualified_phase6(
        ("Alice",), (("target-0",),), {"target-0": (_Proposal("Alice", "first_name", "person"),)}
    )
    plan = _compile_phase7(phase6, execution, phase6.coherence_scopes)
    assert isinstance(plan, _Phase7Plan)
    contract = _load_phase7_contract()
    assert isinstance(contract, _Phase7StableSubstituteContract)
    result = _Phase7Runtime(_Backend(result=_Phase7NddStatus.ABORTED)).run(phase6, execution, plan, contract)

    lost = next(outcome for outcome in result.phase4.accounting.tasks if outcome.task == plan.scope_tasks[0])
    assert isinstance(lost, _TaskLost)
    assert {cause.code.value for cause in lost.causes} == {"transport_lost"}
    assert result.phase4.global_embargo


def test_backend_echoed_dispatch_is_not_trusted_stop_evidence() -> None:
    phase6, _phase6_backend, execution = _qualified_phase6(
        ("Alice",), (("target-0",),), {"target-0": (_Proposal("Alice", "first_name", "person"),)}
    )
    plan = _compile_phase7(phase6, execution, phase6.coherence_scopes)
    assert isinstance(plan, _Phase7Plan)
    contract = _load_phase7_contract()
    assert isinstance(contract, _Phase7StableSubstituteContract)

    result = _Phase7Runtime(_Backend(result=_Phase7NddStatus.ABORTED, echo_dispatch_as_stop_receipt=True)).run(
        phase6, execution, plan, contract
    )

    assert result.scopes[0].state is _ScopePlanState.LOST
    scope_outcome = next(outcome for outcome in result.phase4.accounting.tasks if outcome.task == plan.scope_tasks[0])
    assert isinstance(scope_outcome, _TaskLost)
    assert result.phase4.global_embargo


def test_independently_verified_stop_receipt_acknowledges_cancellation() -> None:
    phase6, _phase6_backend, execution = _qualified_phase6(
        ("Alice",), (("target-0",),), {"target-0": (_Proposal("Alice", "first_name", "person"),)}
    )
    plan = _compile_phase7(phase6, execution, phase6.coherence_scopes)
    assert isinstance(plan, _Phase7Plan)
    contract = _load_phase7_contract()
    assert isinstance(contract, _Phase7StableSubstituteContract)

    receipt = object()
    observed: list[object] = []

    def verify_stop(candidate: object, dispatch: object) -> bool:
        observed.extend((candidate, dispatch))
        return candidate is receipt and getattr(dispatch, "task", None) is plan.scope_tasks[0]

    result = _Phase7Runtime(
        _Backend(result=_Phase7NddStatus.ABORTED, trusted_stop_receipt=receipt),
        trusted_stop_receipt_verified=verify_stop,
    ).run(phase6, execution, plan, contract)

    assert len(observed) == 2
    cancelled = next(outcome for outcome in result.phase4.accounting.tasks if outcome.task == plan.scope_tasks[0])
    assert isinstance(cancelled, _TaskCancelled)
    assert {cause.code.value for cause in cancelled.causes} == {"cancellation", "stop_acknowledged"}
    assert result.phase4.global_embargo


def test_cleanup_attestation_must_be_complete_and_verified() -> None:
    phase6, _phase6_backend, execution = _qualified_phase6(("plain",), (("target-0",),), {})
    plan = _compile_phase7(phase6, execution, phase6.coherence_scopes)
    assert isinstance(plan, _Phase7Plan)
    contract = _load_phase7_contract()
    assert isinstance(contract, _Phase7StableSubstituteContract)
    backend = _Backend(attest_cleanup=False)

    result = _Phase7Runtime(backend).run(phase6, execution, plan, contract)

    assert not result.cleanup.verified
    assert result.phase4.global_embargo


def test_cleanup_attestation_must_bind_the_current_finalization_identity() -> None:
    phase6, _phase6_backend, execution = _qualified_phase6(("plain",), (("target-0",),), {})
    plan = _compile_phase7(phase6, execution, phase6.coherence_scopes)
    assert isinstance(plan, _Phase7Plan)
    contract = _load_phase7_contract()
    assert isinstance(contract, _Phase7StableSubstituteContract)

    result = _Phase7Runtime(_Backend(stale_cleanup_identity=True)).run(phase6, execution, plan, contract)

    assert not result.cleanup.verified
    assert result.phase4.global_embargo


def test_accepted_cancellation_before_dispatch_has_zero_planner_attempts() -> None:
    phase6, _phase6_backend, execution = _qualified_phase6(
        ("Alice",), (("target-0",),), {"target-0": (_Proposal("Alice", "first_name", "person"),)}
    )
    plan = _compile_phase7(phase6, execution, phase6.coherence_scopes)
    assert isinstance(plan, _Phase7Plan)
    contract = _load_phase7_contract()
    assert isinstance(contract, _Phase7StableSubstituteContract)
    backend = _Backend()

    result = _Phase7Runtime(backend, cancellation_requested=lambda: True).run(phase6, execution, plan, contract)

    assert backend.calls == 0
    assert result.scopes[0].state is _ScopePlanState.CANCELLED
    assert result.phase4.global_embargo


def test_accepted_cancellation_after_dispatch_without_verified_stop_is_lost() -> None:
    phase6, _phase6_backend, execution = _qualified_phase6(
        ("Alice",), (("target-0",),), {"target-0": (_Proposal("Alice", "first_name", "person"),)}
    )
    plan = _compile_phase7(phase6, execution, phase6.coherence_scopes)
    assert isinstance(plan, _Phase7Plan)
    contract = _load_phase7_contract()
    assert isinstance(contract, _Phase7StableSubstituteContract)
    observed = [False]

    class _CancellingBackend(_Backend):
        def propose_scope(
            self, manifest: object, handoffs: object, contract: object, dispatch: object
        ) -> _Phase7NddResult:
            observed[0] = True
            return super().propose_scope(manifest, handoffs, contract, dispatch)

    backend = _CancellingBackend()
    result = _Phase7Runtime(backend, cancellation_requested=lambda: observed[0]).run(phase6, execution, plan, contract)

    assert backend.calls == 1
    assert result.scopes[0].state is _ScopePlanState.LOST
    scope_outcome = next(outcome for outcome in result.phase4.accounting.tasks if outcome.task == plan.scope_tasks[0])
    assert isinstance(scope_outcome, _TaskLost)
    assert result.phase4.global_embargo
