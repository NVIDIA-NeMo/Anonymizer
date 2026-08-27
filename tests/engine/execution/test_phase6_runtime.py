# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import pickle
from dataclasses import replace

from anonymizer.engine.execution.accounting_ledger import _AccountingLedger
from anonymizer.engine.execution.accounting_outcomes import _InvocationFailed, _InvocationInconsistent
from anonymizer.engine.execution.accounting_plan import _AccountingLimits
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
from anonymizer.engine.execution.graph import (
    _AtomicGroup,
    _CoherenceScope,
    _ContextScope,
    _DatumId,
    _DatumPurpose,
    _ProtectionGraph,
    _TextDatum,
)
from anonymizer.engine.execution.mention_admission import (
    _MentionId,
    _MentionLimits,
    _ValidationDecision,
    _ValidationDecisionKind,
)
from anonymizer.engine.execution.mention_resolution import (
    _EvidenceVersion,
    _SameSubjectEvidence,
)
from anonymizer.engine.execution.phase6_plan import (
    _compile_phase6_plan,
    _is_admitted_phase6_plan,
    _Phase6Plan,
)
from anonymizer.engine.execution.phase6_runtime import (
    _CandidateProposal,
    _Phase6CandidateWork,
    _Phase6Execution,
    _Phase6ResolverWork,
    _Phase6Runtime,
    _Phase6RuntimeAdmissionError,
    _Phase6ValidationWork,
)


def test_phase6_runtime_test_infrastructure() -> None:
    assert importlib.util.find_spec("anonymizer.engine.execution.phase6_plan") is not None


def test_phase6_plan_freezes_stages_target_only_resolver_scopes_and_predecessors() -> None:
    contract, capability = _contract_and_capability()

    result = _compile_phase6_plan(
        _context_graph(),
        accounting_limits=_ACCOUNTING_LIMITS,
        context_contract=contract,
        capability=capability,
        mention_limits=_MENTION_LIMITS,
    )

    assert isinstance(result, _Phase6Plan)
    assert _is_admitted_phase6_plan(result)
    assert tuple(stage.value for stage in result.accounting.stages) == (
        "detect",
        "augment",
        "validate",
        "finalize",
        "resolve",
        "classify",
        "transform",
        "verify",
    )
    target_by_token = {target.token: target.datum_id.value for target in result.targets}
    assert tuple(
        (
            target_by_token[scope.owner],
            tuple(target_by_token[token] for token in scope.eligible_targets),
        )
        for scope in result.resolver_scopes
    ) == (
        ("target-a", ("target-a", "target-b")),
        ("target-b", ("target-b", "target-a")),
    )
    assert all(
        "context-c" not in members
        for _owner, members in (
            (
                target_by_token[scope.owner],
                tuple(target_by_token[token] for token in scope.eligible_targets),
            )
            for scope in result.resolver_scopes
        )
    )

    tasks = {(task.stage.value, task.datum_id.value): task for task in result.accounting.tasks}
    ledger: _AccountingLedger[str] = _AccountingLedger(result.accounting)
    ledger.open()
    for stage in ("detect", "augment", "validate"):
        ready = ledger.ready_tasks()
        assert {task.stage.value for task in ready} == {stage}
        for task in ready:
            ledger.accept_success(ledger.dispatch(task), stage)
    finalize = ledger.ready_tasks()
    by_datum = {task.datum_id.value: task for task in finalize}
    ledger.accept_success(ledger.dispatch(by_datum["target-a"]), "finalized-a")
    assert ledger.ready_tasks() == (by_datum["target-b"],)
    ledger.accept_success(ledger.dispatch(by_datum["target-b"]), "finalized-b")
    assert set(ledger.ready_tasks()) == {
        tasks[("resolve", "target-a")],
        tasks[("resolve", "target-b")],
    }

    with_pickle_error = result
    try:
        pickle.dumps(with_pickle_error)
    except TypeError as error:
        assert "not serializable" in str(error)
    else:
        raise AssertionError("Phase 6 plans must remain private")


def test_phase6_runtime_accounts_effects_and_releases_exact_local_redact_outputs() -> None:
    plan = _plan(_context_graph())

    class _Backend:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def context_capability(self) -> _ContextBackendCapability:
            return _contract_and_capability()[1]

        def detect(self, work: _Phase6CandidateWork) -> tuple[_CandidateProposal, ...]:
            self.calls.append(("detect", work.target.datum_id.value))
            return (_CandidateProposal(0, len(work.target.text), work.target.text, "name"),)

        def augment(self, work: _Phase6CandidateWork) -> tuple[_CandidateProposal, ...]:
            self.calls.append(("augment", work.target.datum_id.value))
            return ()

        def validate(self, work: _Phase6ValidationWork) -> tuple[_ValidationDecision, ...]:
            self.calls.append(("validate", work.target.datum_id.value))
            return tuple(
                _ValidationDecision(candidate.token, _ValidationDecisionKind.KEEP) for candidate in work.candidates
            )

        def resolve(self, work: _Phase6ResolverWork) -> tuple[_SameSubjectEvidence, ...]:
            self.calls.append(("resolve", work.owner.datum_id.value))
            if work.owner.datum_id.value != "target-a":
                return ()
            owned = tuple(
                mention for mention in work.eligible_mentions if mention.target_datum_id == work.owner.datum_id
            )
            other = tuple(
                mention for mention in work.eligible_mentions if mention.target_datum_id != work.owner.datum_id
            )
            return (
                _SameSubjectEvidence(
                    work.owner.token,
                    owned[0].id,
                    other[0].id,
                    _EvidenceVersion.V1,
                ),
            )

        def close_phase6(self) -> bool:
            return True

    backend = _Backend()
    result = _Phase6Runtime(backend).run(plan)

    assert isinstance(result, _Phase6Execution)
    assert tuple((datum.datum_id.value, datum.output) for datum in result.released) == (
        ("target-a", "[REDACTED]"),
        ("target-b", "[REDACTED]"),
    )
    assert len(backend.calls) == 8
    assert {type(task).__name__ for task in result.accounting.tasks} == {"_TaskSucceeded"}
    assert "Alice" not in repr(result)
    assert "[REDACTED]" not in repr(result)
    try:
        pickle.dumps(result)
    except TypeError as error:
        assert "not serializable" in str(error)
    else:
        raise AssertionError("Phase 6 executions must remain private")


def test_phase6_runtime_localizes_known_target_failure_without_raw_fallback() -> None:
    plan = _plan(_independent_graph())

    class _Backend:
        def context_capability(self) -> _ContextBackendCapability:
            return _contract_and_capability()[1]

        def detect(self, work: _Phase6CandidateWork) -> tuple[_CandidateProposal, ...]:
            if work.target.datum_id.value == "target-a":
                raise RuntimeError("known detector failure")
            return ()

        def augment(self, work: _Phase6CandidateWork) -> tuple[_CandidateProposal, ...]:
            return ()

        def validate(self, work: _Phase6ValidationWork) -> tuple[_ValidationDecision, ...]:
            return ()

        def resolve(self, work: _Phase6ResolverWork) -> tuple[_SameSubjectEvidence, ...]:
            return ()

        def close_phase6(self) -> bool:
            return True

    result = _Phase6Runtime(_Backend()).run(plan)

    assert tuple((datum.datum_id.value, datum.output) for datum in result.released) == (("target-b", "public"),)
    assert all(datum.datum_id.value != "target-a" for datum in result.released)


def test_phase6_runtime_rechecks_capability_before_opening_effects() -> None:
    plan = _plan(_independent_graph())

    class _Backend(_NoMentionBackend):
        def context_capability(self) -> _ContextBackendCapability:
            capability = _contract_and_capability()[1]
            return replace(capability, retention=_RetentionPosture.ENABLED)

    backend = _Backend()

    try:
        _Phase6Runtime(backend).run(plan)
    except _Phase6RuntimeAdmissionError:
        pass
    else:
        raise AssertionError("runtime capability drift must reject before effects")
    assert backend.effect_count == 0
    assert backend.close_count == 0


def test_phase6_runtime_cleanup_failure_embargoes_verified_outputs() -> None:
    plan = _plan(_independent_graph())
    backend = _NoMentionBackend(clean=False)

    result = _Phase6Runtime(backend).run(plan)

    assert isinstance(result.accounting.invocation, _InvocationFailed)
    assert result.released == ()
    assert backend.close_count == 1


def test_phase6_runtime_foreign_resolver_endpoint_causes_global_embargo() -> None:
    plan = _plan(_context_graph())

    class _Backend(_NoMentionBackend):
        def detect(self, work: _Phase6CandidateWork) -> tuple[_CandidateProposal, ...]:
            self.effect_count += 1
            return (_CandidateProposal(0, len(work.target.text), work.target.text, "name"),)

        def validate(self, work: _Phase6ValidationWork) -> tuple[_ValidationDecision, ...]:
            self.effect_count += 1
            return tuple(
                _ValidationDecision(candidate.token, _ValidationDecisionKind.KEEP) for candidate in work.candidates
            )

        def resolve(self, work: _Phase6ResolverWork) -> tuple[_SameSubjectEvidence, ...]:
            self.effect_count += 1
            return (
                _SameSubjectEvidence(
                    work.owner.token,
                    _MentionId(),
                    work.eligible_mentions[0].id,
                    _EvidenceVersion.V1,
                ),
            )

    result = _Phase6Runtime(_Backend()).run(plan)

    assert isinstance(result.accounting.invocation, _InvocationInconsistent)
    assert result.released == ()


def _plan(graph: _ProtectionGraph) -> _Phase6Plan:
    contract, capability = _contract_and_capability()
    result = _compile_phase6_plan(
        graph,
        accounting_limits=_ACCOUNTING_LIMITS,
        context_contract=contract,
        capability=capability,
        mention_limits=_MENTION_LIMITS,
    )
    assert isinstance(result, _Phase6Plan)
    return result


def _contract_and_capability() -> tuple[_ContextExecutionContract, _ContextBackendCapability]:
    limits = _ContextLimits(4, 128, 8, 512)
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
    return contract, capability


def _context_graph() -> _ProtectionGraph:
    target_a = _TextDatum(_DatumId("target-a"), "Alice", _DatumPurpose.TARGET)
    target_b = _TextDatum(_DatumId("target-b"), "A. Example", _DatumPurpose.TARGET)
    context = _TextDatum(_DatumId("context-c"), "private context", _DatumPurpose.CONTEXT_ONLY)
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


def _independent_graph() -> _ProtectionGraph:
    target_a = _TextDatum(_DatumId("target-a"), "private", _DatumPurpose.TARGET)
    target_b = _TextDatum(_DatumId("target-b"), "public", _DatumPurpose.TARGET)
    return _ProtectionGraph(
        datums=(target_a, target_b),
        links=(),
        context_scopes=(_ContextScope(target_a.id), _ContextScope(target_b.id)),
        coherence_scopes=(_CoherenceScope((target_a.id,)), _CoherenceScope((target_b.id,))),
        atomic_groups=(_AtomicGroup((target_a.id,)), _AtomicGroup((target_b.id,))),
    )


class _NoMentionBackend:
    def __init__(self, *, clean: bool = True) -> None:
        self.clean = clean
        self.effect_count = 0
        self.close_count = 0

    def context_capability(self) -> _ContextBackendCapability:
        return _contract_and_capability()[1]

    def detect(self, work: _Phase6CandidateWork) -> tuple[_CandidateProposal, ...]:
        del work
        self.effect_count += 1
        return ()

    def augment(self, work: _Phase6CandidateWork) -> tuple[_CandidateProposal, ...]:
        del work
        self.effect_count += 1
        return ()

    def validate(self, work: _Phase6ValidationWork) -> tuple[_ValidationDecision, ...]:
        del work
        self.effect_count += 1
        return ()

    def resolve(self, work: _Phase6ResolverWork) -> tuple[_SameSubjectEvidence, ...]:
        del work
        self.effect_count += 1
        return ()

    def close_phase6(self) -> bool:
        self.close_count += 1
        return self.clean


_ACCOUNTING_LIMITS = _AccountingLimits(
    max_datums=8,
    max_datum_bytes=128,
    max_graph_bytes=512,
    max_stages=8,
)
_MENTION_LIMITS = _MentionLimits(8, 8, 64, 128)
