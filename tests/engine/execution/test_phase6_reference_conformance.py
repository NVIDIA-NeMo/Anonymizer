# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import anonymizer.engine.execution.phase6_runtime as phase6_runtime
from anonymizer.engine.execution.accounting_admission import _compile_accounting_plan
from anonymizer.engine.execution.accounting_evidence import _Dispatch, _SuccessRecord
from anonymizer.engine.execution.accounting_ledger import _AccountingLedger
from anonymizer.engine.execution.accounting_outcomes import _GroupReleased
from anonymizer.engine.execution.accounting_plan import (
    _AccountingLimits,
    _AccountingPlan,
    _DatumTaskSubject,
    _TaskKey,
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
from anonymizer.engine.execution.graph import (
    _AtomicGroup,
    _CoherenceScope,
    _ContextScope,
    _DatumDependency,
    _DatumId,
    _DatumPurpose,
    _ProtectionGraph,
    _TextDatum,
    _trivial_graph,
)
from anonymizer.engine.execution.mention_admission import (
    _AnchoredMention,
    _MentionLimits,
    _ValidationDecision,
    _ValidationDecisionKind,
)
from anonymizer.engine.execution.mention_resolution import (
    _ClusteredGraph,
    _DistinctSubjectEvidence,
    _EvidenceVersion,
    _SameSubjectEvidence,
    _SubjectEvidence,
)
from anonymizer.engine.execution.phase6_plan import _compile_phase6_plan, _Phase6Plan
from anonymizer.engine.execution.phase6_runtime import (
    _CandidateProposal,
    _Phase6AugmentationWork,
    _Phase6CandidateWork,
    _Phase6ResolverWork,
    _Phase6Runtime,
    _Phase6ValidationWork,
)
from anonymizer.engine.execution.role_policy import _RolePolicy
from tests.engine.execution.phase6_reference_model import (
    ReferenceCandidate,
    ReferenceCase,
    ReferenceEventKind,
    finite_reference_cases,
    reduce_reference,
    schedule_reference_cases,
)


def _datum_id(task: _TaskKey) -> _DatumId:
    assert isinstance(task.subject, _DatumTaskSubject)
    return task.subject.datum_id


@pytest.mark.parametrize("case", finite_reference_cases(), ids=lambda case: case.name)
def test_phase6_runtime_matches_every_frozen_reference_schedule(
    case: ReferenceCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = reduce_reference(case)
    observed_clusters: list[_ClusteredGraph] = []
    classify_roles = phase6_runtime._classify_roles

    def _observe_clusters(clustered: _ClusteredGraph, policy: _RolePolicy) -> object:
        observed_clusters.append(clustered)
        return classify_roles(clustered, policy)

    monkeypatch.setattr(phase6_runtime, "_classify_roles", _observe_clusters)
    backend = _ReferenceBackend(case)
    execution = _Phase6Runtime(backend).run(_plan(case))

    released = tuple(
        group_index
        for group_index, group in enumerate(execution.accounting.groups)
        if isinstance(group, _GroupReleased)
    )
    expected_released_outputs = tuple(
        (target_index, expected.outputs[target_index])
        for group_index in expected.released_groups
        for target_index in _groups(case)[group_index]
    )
    actual_released_outputs = tuple(
        (int(datum.datum_id.value.removeprefix("target-")), datum.output) for datum in execution.released
    )

    assert released == expected.released_groups
    assert actual_released_outputs == expected_released_outputs
    assert (
        type(execution.accounting.invocation).__name__
        == {
            "completed": "_InvocationCompleted",
            "cancelled": "_InvocationCancelled",
            "failed": "_InvocationFailed",
            "lost": "_InvocationLost",
            "inconsistent": "_InvocationInconsistent",
        }[expected.schedule.invocation]
    )
    assert backend.closed is (expected.schedule.cleanup == "accepted")
    assert _normalize_mentions(backend.observed_mentions) == tuple(
        (mention.target, mention.start, mention.end, mention.source_slice, mention.label)
        for mention in expected.mentions
    )
    assert _normalize_clusters(observed_clusters) == expected.clusters


@pytest.mark.parametrize("case", schedule_reference_cases(), ids=lambda case: case.name)
def test_production_accounting_matches_every_frozen_lifecycle_schedule(case: ReferenceCase) -> None:
    expected = reduce_reference(case)
    actual = _execute_accounting_schedule(case)

    assert actual == (
        expected.schedule.invocation,
        expected.schedule.task_outcomes,
        expected.schedule.cancellation,
        expected.schedule.released_subjects,
    )


def _execute_accounting_schedule(
    case: ReferenceCase,
) -> tuple[str, tuple[tuple[str, str], ...], str, tuple[str, ...]]:
    plan = _accounting_plan(case)
    ledger: _AccountingLedger[str] = _AccountingLedger(plan)
    ledger.open()
    dispatches: dict[str, _Dispatch] = {}
    terminal_subjects: set[str] = set()
    finalized = False
    verified = False
    cleanup = "unconfirmed"
    teardown = "unconfirmed"
    immutable_result = False
    cancellation = "none"
    result = None

    for event in case.events:
        subject = event.subject if event.subject != "invocation" else "target-0"
        match event.kind:
            case ReferenceEventKind.DISPATCH:
                ready = {_datum_id(task).value: task for task in ledger.ready_tasks()}
                if subject in ready:
                    dispatches[subject] = ledger.dispatch(ready[subject])
            case ReferenceEventKind.TERMINAL:
                dispatch = dispatches.get(subject)
                if dispatch is None:
                    continue
                if event.outcome == "contradictory":
                    records = (
                        _SuccessRecord(dispatch, "first"),
                        _SuccessRecord(dispatch, "second"),
                    )
                    ledger.reconcile((dispatch,), records, trusted_run_record=True)
                elif event.outcome == "failed":
                    ledger.accept_failure(dispatch)
                else:
                    ledger.accept_success(dispatch, f"verified-{subject}")
                terminal_subjects.add(subject)
            case ReferenceEventKind.CANCEL:
                cancellation = (
                    "before_dispatch" if not dispatches else "after_terminal" if terminal_subjects else "after_dispatch"
                )
                ledger.request_cancellation()
                for dispatched_subject, dispatch in dispatches.items():
                    if dispatched_subject not in terminal_subjects:
                        ledger.acknowledge_stop(dispatch)
            case ReferenceEventKind.LOSS:
                dispatch = dispatches.get(subject)
                if dispatch is not None:
                    ledger.mark_transport_lost(dispatch)
            case ReferenceEventKind.CANDIDATE_DECISION:
                dispatch = dispatches.get(subject)
                if dispatch is not None:
                    ledger.accept_success(dispatch, f"candidate-{subject}")
            case ReferenceEventKind.EVIDENCE:
                dispatch = dispatches.get(subject)
                if dispatch is not None:
                    if event.outcome == "duplicate":
                        record = _SuccessRecord(dispatch, f"evidence-{subject}")
                        ledger.reconcile((dispatch,), (record, record), trusted_run_record=True)
                    else:
                        ledger.accept_success(dispatch, f"evidence-{subject}")
            case ReferenceEventKind.FINALIZE:
                finalized = True
            case ReferenceEventKind.VERIFY:
                verified = True
            case ReferenceEventKind.CLEANUP:
                cleanup = event.outcome
                if cleanup == "unconfirmed":
                    ledger.mark_cleanup_unconfirmed()
                elif cleanup == "failed":
                    ledger.mark_cleanup_failed()
            case ReferenceEventKind.IMMUTABLE_ACCEPT:
                immutable_result = event.outcome == "accepted"
                result = ledger.finish(
                    group_release_predicate=lambda _outputs: (
                        finalized and verified and cleanup == "accepted" and teardown != "failed" and immutable_result
                    )
                )
            case ReferenceEventKind.TEARDOWN:
                teardown = event.outcome
                if teardown == "failed" and result is None:
                    ledger.mark_cleanup_failed()
            case ReferenceEventKind.RELEASE:
                if result is None:
                    result = ledger.finish(group_release_predicate=lambda _outputs: False)
                break
            case _:
                pass

    if result is None:
        if cleanup == "unconfirmed":
            ledger.mark_cleanup_unconfirmed()
        result = ledger.finish(group_release_predicate=lambda _outputs: False)
    invocation = type(result.invocation).__name__.removeprefix("_Invocation").lower()
    tasks = tuple(
        (_datum_id(outcome.task).value, type(outcome).__name__.removeprefix("_Task").lower())
        for outcome in result.tasks
    )
    released = tuple(
        datum_id.value
        for group in result.groups
        if isinstance(group, _GroupReleased)
        for datum_id, _candidate in group.outputs
    )
    return invocation, tasks, cancellation, released


def _accounting_plan(case: ReferenceCase) -> _AccountingPlan:
    graph = _trivial_graph(
        tuple(
            _TextDatum(_DatumId(f"target-{index}"), text, _DatumPurpose.TARGET) for index, text in enumerate(case.texts)
        )
    )
    compiled = _compile_accounting_plan(
        graph,
        limits=_AccountingLimits(max_datums=4, max_datum_bytes=128, max_graph_bytes=512),
    )
    assert isinstance(compiled, _AccountingPlan)
    return compiled


class _ReferenceBackend:
    """Translate declarations, never reduced expected results, into runtime effects."""

    def __init__(self, case: ReferenceCase) -> None:
        self._case = case
        self.observed_mentions: list[_AnchoredMention] = []
        self.closed = False

    def context_capability(self) -> _ContextBackendCapability:
        return _contract_and_capability()[1]

    def detect(self, work: _Phase6CandidateWork) -> tuple[_CandidateProposal, ...]:
        target = _target_index(work.target.datum_id)
        if not _group_passes(self._case)[_group_for_target(self._case, target)]:
            raise RuntimeError("declared local target failure")
        return tuple(
            _CandidateProposal(candidate.start, candidate.end, candidate.source_slice, candidate.label)
            for candidate in self._case.candidates
            if candidate.target == target
        )

    def augment(self, work: _Phase6AugmentationWork) -> tuple[_CandidateProposal, ...]:
        del work
        return ()

    def validate(self, work: _Phase6ValidationWork) -> tuple[_ValidationDecision, ...]:
        target = _target_index(work.target.datum_id)
        declarations = tuple(candidate for candidate in self._case.candidates if candidate.target == target)
        decisions: list[_ValidationDecision] = []
        for candidate, declaration in zip(work.candidates, declarations, strict=True):
            if declaration.decision == "missing":
                continue
            kind = {
                "keep": _ValidationDecisionKind.KEEP,
                "drop": _ValidationDecisionKind.DROP,
                "reclass": _ValidationDecisionKind.RECLASS,
            }[declaration.decision]
            decisions.append(_ValidationDecision(candidate.token, kind, declaration.reclassified_label))
        return tuple(decisions)

    def resolve(self, work: _Phase6ResolverWork) -> tuple[_SubjectEvidence, ...]:
        self.observed_mentions.extend(work.eligible_mentions)
        owner = _target_index(work.owner.datum_id)
        mentions = {
            (candidate.target, candidate.start, candidate.end, candidate.source_slice): mention
            for candidate in self._case.candidates
            for mention in work.eligible_mentions
            if _mention_matches(candidate, mention)
        }
        evidence: list[_SubjectEvidence] = []
        for declaration in self._case.evidence:
            left = self._case.candidates[declaration.left_candidate]
            right = self._case.candidates[declaration.right_candidate]
            if owner != left.target:
                continue
            evidence_type = _SameSubjectEvidence if declaration.kind == "same_subject" else _DistinctSubjectEvidence
            evidence.append(
                evidence_type(
                    work.owner.token,
                    mentions[(left.target, left.start, left.end, left.source_slice)].id,
                    mentions[(right.target, right.start, right.end, right.source_slice)].id,
                    _EvidenceVersion.V1,
                )
            )
        return tuple(evidence)

    def close_phase6(self) -> bool:
        self.closed = True
        return True


def _plan(case: ReferenceCase) -> _Phase6Plan:
    datums = tuple(
        _TextDatum(_DatumId(f"target-{index}"), text, _DatumPurpose.TARGET) for index, text in enumerate(case.texts)
    )
    ids = tuple(datum.id for datum in datums)
    context_scopes = tuple(
        _ContextScope(datum_id, tuple(candidate for candidate in ids if candidate != datum_id)) for datum_id in ids
    )
    groups = _groups(case)
    graph = _ProtectionGraph(
        datums,
        (),
        context_scopes,
        tuple(_CoherenceScope(tuple(ids[index] for index in group)) for group in groups),
        tuple(_AtomicGroup(tuple(ids[index] for index in group)) for group in groups),
        tuple(_DatumDependency(ids[left], ids[right]) for left, right in case.dependencies),
    )
    contract, capability = _contract_and_capability()
    compiled = _compile_phase6_plan(
        graph,
        accounting_limits=_AccountingLimits(
            max_datums=8,
            max_datum_bytes=128,
            max_graph_bytes=512,
            max_stages=8,
        ),
        context_contract=contract,
        capability=capability,
        mention_limits=_MentionLimits(16, 16, 64, 128),
    )
    assert isinstance(compiled, _Phase6Plan)
    return compiled


def _contract_and_capability() -> tuple[_ContextExecutionContract, _ContextBackendCapability]:
    limits = _ContextLimits(8, 128, 16, 1024)
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


def _groups(case: ReferenceCase) -> tuple[tuple[int, ...], ...]:
    return case.groups or tuple((index,) for index in range(len(case.texts)))


def _group_passes(case: ReferenceCase) -> tuple[bool, ...]:
    return case.group_passes or tuple(True for _group in _groups(case))


def _group_for_target(case: ReferenceCase, target: int) -> int:
    return next(index for index, group in enumerate(_groups(case)) if target in group)


def _target_index(datum_id: _DatumId) -> int:
    return int(datum_id.value.removeprefix("target-"))


def _mention_matches(candidate: ReferenceCandidate, mention: _AnchoredMention) -> bool:
    return (
        _target_index(mention.target_datum_id) == candidate.target
        and mention.start == candidate.start
        and mention.end == candidate.end
        and mention.source_slice == candidate.source_slice
    )


def _ordered_mentions(graphs: list[_ClusteredGraph]) -> tuple[_AnchoredMention, ...]:
    mentions = {mention.id: mention for graph in graphs for mention in graph.detected.mentions}
    return tuple(
        sorted(
            mentions.values(),
            key=lambda mention: (_target_index(mention.target_datum_id), mention.start, mention.end),
        )
    )


def _normalize_mentions(mentions: list[_AnchoredMention]) -> tuple[tuple[int, int, int, str, str], ...]:
    unique = {mention.id: mention for mention in mentions}
    return tuple(
        (
            _target_index(mention.target_datum_id),
            mention.start,
            mention.end,
            mention.source_slice,
            mention.detector_label,
        )
        for mention in sorted(
            unique.values(),
            key=lambda mention: (_target_index(mention.target_datum_id), mention.start, mention.end),
        )
    )


def _normalize_clusters(graphs: list[_ClusteredGraph]) -> tuple[tuple[int, ...], ...]:
    mentions = _ordered_mentions(graphs)
    position = {mention.id: index for index, mention in enumerate(mentions)}
    clusters = {
        tuple(sorted(position[mention_id] for mention_id in cluster.ordered_mention_ids))
        for graph in graphs
        for cluster in graph.clusters
    }
    return tuple(sorted(clusters))
