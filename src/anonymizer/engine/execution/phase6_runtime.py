# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Invocation-private coordinator for the Phase 6 local Redact profile."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, TypeAlias

from anonymizer.engine.execution.accounting_ledger import _AccountingLedger
from anonymizer.engine.execution.accounting_outcomes import _AccountingResult, _CauseCode, _GroupReleased
from anonymizer.engine.execution.accounting_plan import _TaskKey
from anonymizer.engine.execution.context_contract import _capability_satisfies, _snapshot_context_capability
from anonymizer.engine.execution.graph import _DatumId, _TextDatum
from anonymizer.engine.execution.mention_admission import (
    _AnchoredMention,
    _CandidateToken,
    _DetectedGraph,
    _finalize_mentions,
    _MentionProvenance,
    _MentionRejected,
    _MentionTarget,
    _ProvisionalCandidate,
    _ValidationDecision,
)
from anonymizer.engine.execution.mention_resolution import (
    _ClusteredGraph,
    _ResolutionRejected,
    _ResolutionRejectionCode,
    _resolve_mentions,
    _ResolverScope,
    _SubjectEvidence,
)
from anonymizer.engine.execution.phase6_plan import (
    _is_admitted_phase6_plan,
    _Phase6Component,
    _Phase6ComponentKey,
    _Phase6Plan,
)
from anonymizer.engine.execution.redact_patches import (
    _apply_redact_patches,
    _bind_patch_manifest,
    _BoundPatchManifest,
    _build_patch_manifest,
    _materialize_redact_patches,
    _PatchRejected,
    _RedactPatch,
    _ReturnedRedact,
    _VerifiedDatum,
    _VerifiedGraph,
    _verify_redact_patches,
)
from anonymizer.engine.execution.role_policy import (
    _classify_roles,
    _ResolvedGraph,
    _RolePolicyRejected,
)


class _PrivatePhase6RuntimeValue:
    def __repr__(self) -> str:
        return f"<private {type(self).__name__.strip('_').replace('_', ' ').lower()}>"

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private Phase 6 runtime values are not serializable")


@dataclass(frozen=True, slots=True, repr=False)
class _CandidateProposal(_PrivatePhase6RuntimeValue):
    start: int
    end: int
    source_slice: str
    detector_label: str


@dataclass(frozen=True, slots=True, repr=False)
class _Phase6CandidateWork(_PrivatePhase6RuntimeValue):
    target: _MentionTarget
    context: tuple[_TextDatum, ...]


@dataclass(frozen=True, slots=True, repr=False)
class _Phase6ValidationWork(_PrivatePhase6RuntimeValue):
    target: _MentionTarget
    candidates: tuple[_ProvisionalCandidate, ...]


@dataclass(frozen=True, slots=True, repr=False)
class _Phase6ResolverWork(_PrivatePhase6RuntimeValue):
    owner: _MentionTarget
    eligible_mentions: tuple[_AnchoredMention, ...]


class _Phase6EffectBackend(Protocol):
    def context_capability(self) -> object: ...

    def detect(self, work: _Phase6CandidateWork) -> tuple[_CandidateProposal, ...]: ...

    def augment(self, work: _Phase6CandidateWork) -> tuple[_CandidateProposal, ...]: ...

    def validate(self, work: _Phase6ValidationWork) -> tuple[_ValidationDecision, ...]: ...

    def resolve(self, work: _Phase6ResolverWork) -> tuple[_SubjectEvidence, ...]: ...

    def close_phase6(self) -> bool: ...


@dataclass(frozen=True, slots=True, repr=False)
class _Phase6StageReceipt(_PrivatePhase6RuntimeValue):
    task: _TaskKey


_Phase6Candidate: TypeAlias = _Phase6StageReceipt | _VerifiedDatum


@dataclass(frozen=True, slots=True, repr=False)
class _Phase6Execution(_PrivatePhase6RuntimeValue):
    accounting: _AccountingResult[_Phase6Candidate]
    released: tuple[_VerifiedDatum, ...]


class _Phase6RuntimeAdmissionError(TypeError):
    def __init__(self) -> None:
        super().__init__("admitted private Phase 6 plan and compatible backend required")

    def __repr__(self) -> str:
        return "<private Phase 6 runtime admission error>"


class _Phase6TransportLost(RuntimeError):
    """Backend signal that a dispatched attempt has no trusted terminal evidence."""


class _GlobalPhase6Fault(Exception):
    pass


class _ComponentPhase6Fault(Exception):
    pass


@dataclass(slots=True, repr=False)
class _RuntimeStore(_PrivatePhase6RuntimeValue):
    candidates: dict[_DatumId, list[_ProvisionalCandidate]] = field(default_factory=dict)
    decisions: dict[_DatumId, tuple[_ValidationDecision, ...]] = field(default_factory=dict)
    detected: dict[_DatumId, _DetectedGraph] = field(default_factory=dict)
    evidence: dict[_DatumId, tuple[_SubjectEvidence, ...]] = field(default_factory=dict)
    clustered: dict[_Phase6ComponentKey, _ClusteredGraph] = field(default_factory=dict)
    resolved: dict[_Phase6ComponentKey, _ResolvedGraph] = field(default_factory=dict)
    bound: dict[_Phase6ComponentKey, _BoundPatchManifest] = field(default_factory=dict)
    patches: dict[_Phase6ComponentKey, tuple[_RedactPatch, ...]] = field(default_factory=dict)
    returned: dict[_Phase6ComponentKey, tuple[_ReturnedRedact, ...]] = field(default_factory=dict)
    verified: dict[_Phase6ComponentKey, _VerifiedGraph] = field(default_factory=dict)

    def close(self) -> None:
        self.candidates.clear()
        self.decisions.clear()
        self.detected.clear()
        self.evidence.clear()
        self.clustered.clear()
        self.resolved.clear()
        self.bound.clear()
        self.patches.clear()
        self.returned.clear()
        self.verified.clear()


class _Phase6Runtime:
    """Account every Phase 6 semantic task and expose only released Redact results."""

    def __init__(self, backend: _Phase6EffectBackend) -> None:
        self._backend = backend

    def run(self, plan: _Phase6Plan) -> _Phase6Execution:
        self._preflight(plan)
        ledger: _AccountingLedger[_Phase6Candidate] = _AccountingLedger(plan.accounting)
        store = _RuntimeStore()
        ledger.open()
        try:
            self._drive_ledger(plan, ledger, store)
        finally:
            self._close_before_release(ledger, store)
        accounting = ledger.finish(
            datum_release_predicate=_verified_datum_predicate,
            group_release_predicate=_verified_group_predicate,
        )
        return _Phase6Execution(accounting, _collect_released(plan, accounting))

    def _drive_ledger(
        self,
        plan: _Phase6Plan,
        ledger: _AccountingLedger[_Phase6Candidate],
        store: _RuntimeStore,
    ) -> None:
        while ready := ledger.ready_tasks():
            task = ready[0]
            dispatch = ledger.dispatch(task)
            try:
                candidate = self._execute_task(plan, task, store)
            except _Phase6TransportLost:
                ledger.mark_transport_lost(dispatch)
            except _GlobalPhase6Fault:
                ledger.mark_inconsistent(_CauseCode.CONTRADICTORY)
            except _ComponentPhase6Fault:
                ledger.accept_failure(dispatch)
                self._fail_component_stage(plan, ledger, task)
            except KeyboardInterrupt:
                ledger.request_cancellation()
                raise
            except Exception:
                ledger.accept_failure(dispatch)
            else:
                ledger.accept_success(dispatch, candidate)

    def _preflight(self, plan: object) -> None:
        methods = ("detect", "augment", "validate", "resolve", "close_phase6")
        if (
            not isinstance(plan, _Phase6Plan)
            or not _is_admitted_phase6_plan(plan)
            or not _capability_satisfies(_snapshot_context_capability(self._backend), plan.context.contract)
            or not all(callable(getattr(self._backend, method, None)) for method in methods)
        ):
            raise _Phase6RuntimeAdmissionError

    def _execute_task(
        self,
        plan: _Phase6Plan,
        task: _TaskKey,
        store: _RuntimeStore,
    ) -> _Phase6Candidate:
        target = _target_for_datum(plan, task.datum_id)
        if task.stage.value in {"detect", "augment", "validate", "finalize"}:
            self._execute_mention_task(plan, task, target, store)
            return _Phase6StageReceipt(task)
        match task.stage.value:
            case "resolve":
                self._resolve_target(plan, target, store)
            case "classify":
                self._classify_component(plan, target, store)
            case "transform":
                self._transform_component(plan, target, store)
            case "verify":
                return self._verify_component(plan, target, store)
            case _:
                raise _GlobalPhase6Fault
        return _Phase6StageReceipt(task)

    def _execute_mention_task(
        self,
        plan: _Phase6Plan,
        task: _TaskKey,
        target: _MentionTarget,
        store: _RuntimeStore,
    ) -> None:
        match task.stage.value:
            case "detect":
                self._add_proposals(plan, target, self._backend.detect(_candidate_work(plan, target)), store, False)
            case "augment":
                self._add_proposals(plan, target, self._backend.augment(_candidate_work(plan, target)), store, True)
            case "validate":
                candidates = tuple(store.candidates.get(target.datum_id, ()))
                decisions = self._backend.validate(_Phase6ValidationWork(target, candidates))
                if not isinstance(decisions, tuple) or not all(
                    isinstance(decision, _ValidationDecision) for decision in decisions
                ):
                    raise _ComponentPhase6Fault
                store.decisions[target.datum_id] = decisions
            case "finalize":
                self._finalize_target(plan, target, store)
            case _:
                raise _GlobalPhase6Fault

    @staticmethod
    def _finalize_target(plan: _Phase6Plan, target: _MentionTarget, store: _RuntimeStore) -> None:
        result = _finalize_mentions(
            (target,),
            tuple(store.candidates.get(target.datum_id, ())),
            store.decisions.get(target.datum_id, ()),
            limits=plan.mention_limits,
        )
        if isinstance(result, _MentionRejected):
            if result.owner is None:
                raise _GlobalPhase6Fault
            raise _ComponentPhase6Fault
        store.detected[target.datum_id] = result

    def _add_proposals(
        self,
        plan: _Phase6Plan,
        target: _MentionTarget,
        proposals: object,
        store: _RuntimeStore,
        augmented: bool,
    ) -> None:
        if not isinstance(proposals, tuple) or not all(isinstance(item, _CandidateProposal) for item in proposals):
            raise _ComponentPhase6Fault
        current = store.candidates.setdefault(target.datum_id, [])
        provenance = _MentionProvenance.EXACT_AUGMENTER if augmented else _MentionProvenance.SPAN_DETECTOR
        for proposal in proposals:
            candidate = _candidate_from_proposal(target, proposal, provenance, plan)
            current.append(candidate)
        if len(current) > plan.mention_limits.max_candidates_per_target:
            raise _ComponentPhase6Fault

    def _resolve_target(
        self,
        plan: _Phase6Plan,
        target: _MentionTarget,
        store: _RuntimeStore,
    ) -> None:
        scope = next(scope for scope in plan.resolver_scopes if scope.owner is target.token)
        detected = _combine_detected(plan, scope.eligible_targets, store)
        work = _Phase6ResolverWork(target, detected.mentions)
        evidence = self._backend.resolve(work)
        if not isinstance(evidence, tuple):
            raise _ComponentPhase6Fault
        validation_scopes = tuple(
            scope if candidate.token is target.token else _ResolverScope(candidate.token, (candidate.token,))
            for candidate in detected.targets
        )
        validated = _resolve_mentions(detected, validation_scopes, evidence)
        if isinstance(validated, _ResolutionRejected):
            if validated.code in {
                _ResolutionRejectionCode.FOREIGN_TOKEN,
                _ResolutionRejectionCode.STALE_TOKEN,
            }:
                raise _GlobalPhase6Fault
            raise _ComponentPhase6Fault
        store.evidence[target.datum_id] = evidence

    @staticmethod
    def _classify_component(plan: _Phase6Plan, target: _MentionTarget, store: _RuntimeStore) -> None:
        component = _component_for_target(plan, target)
        if component.key in store.resolved:
            return
        detected = _combine_detected(plan, component.target_tokens, store)
        scopes = tuple(scope for scope in plan.resolver_scopes if scope.owner in component.target_tokens)
        evidence = tuple(
            item
            for member in _targets_for_component(plan, component)
            for item in store.evidence.get(member.datum_id, ())
        )
        clustered = _resolve_mentions(detected, scopes, evidence)
        if isinstance(clustered, _ResolutionRejected):
            raise _ComponentPhase6Fault
        resolved = _classify_roles(clustered, plan.role_policy)
        if isinstance(resolved, _RolePolicyRejected):
            raise _ComponentPhase6Fault
        store.clustered[component.key] = clustered
        store.resolved[component.key] = resolved

    @staticmethod
    def _transform_component(plan: _Phase6Plan, target: _MentionTarget, store: _RuntimeStore) -> None:
        component = _component_for_target(plan, target)
        if component.key in store.bound:
            return
        resolved = store.resolved.get(component.key)
        if resolved is None:
            raise _ComponentPhase6Fault
        manifest = _build_patch_manifest(resolved)
        if isinstance(manifest, _PatchRejected):
            raise _ComponentPhase6Fault
        bound = _bind_patch_manifest(manifest)
        if isinstance(bound, _PatchRejected):
            raise _ComponentPhase6Fault
        patches = _materialize_redact_patches(bound)
        if isinstance(patches, _PatchRejected):
            raise _ComponentPhase6Fault
        store.bound[component.key] = bound
        store.patches[component.key] = patches
        returned = _apply_redact_patches(resolved, patches)
        if isinstance(returned, _PatchRejected):
            raise _ComponentPhase6Fault
        store.returned[component.key] = returned

    @staticmethod
    def _verify_component(
        plan: _Phase6Plan,
        target: _MentionTarget,
        store: _RuntimeStore,
    ) -> _VerifiedDatum:
        component = _component_for_target(plan, target)
        verified = store.verified.get(component.key)
        if verified is None:
            bound = store.bound.get(component.key)
            if bound is None:
                raise _ComponentPhase6Fault
            result = _verify_redact_patches(
                bound,
                store.patches.get(component.key, ()),
                store.returned.get(component.key, ()),
            )
            if isinstance(result, _PatchRejected):
                raise _ComponentPhase6Fault
            store.verified[component.key] = result
            verified = result
        candidate = next((datum for datum in verified.datums if datum.datum_id == target.datum_id), None)
        if candidate is None:
            raise _ComponentPhase6Fault
        return candidate

    @staticmethod
    def _fail_component_stage(
        plan: _Phase6Plan,
        ledger: _AccountingLedger[_Phase6Candidate],
        task: _TaskKey,
    ) -> None:
        target = _target_for_datum(plan, task.datum_id)
        component = _component_for_target(plan, target)
        member_ids = {member.datum_id for member in _targets_for_component(plan, component)}
        for candidate in plan.accounting.tasks:
            if candidate.stage == task.stage and candidate.datum_id in member_ids and candidate != task:
                ledger.mark_task_failed(candidate)

    def _close_before_release(self, ledger: _AccountingLedger[_Phase6Candidate], store: _RuntimeStore) -> None:
        try:
            closed = self._backend.close_phase6()
        except Exception:
            ledger.mark_cleanup_unconfirmed()
        else:
            if type(closed) is not bool or not closed:
                ledger.mark_cleanup_failed()
        finally:
            store.close()


def _candidate_work(plan: _Phase6Plan, target: _MentionTarget) -> _Phase6CandidateWork:
    projection = next(item for item in plan.context.projections if item.target_datum_id == target.datum_id)
    datum_by_id = {datum.id: datum for datum in (*plan.context.accounting.datums, *plan.context.context_only_datums)}
    return _Phase6CandidateWork(target, tuple(datum_by_id[datum_id] for datum_id in projection.context_datum_ids))


def _candidate_from_proposal(
    target: _MentionTarget,
    proposal: _CandidateProposal,
    provenance: _MentionProvenance,
    plan: _Phase6Plan,
) -> _ProvisionalCandidate:
    values = (proposal.start, proposal.end)
    if (
        any(type(value) is not int for value in values)
        or proposal.start < 0
        or proposal.end <= proposal.start
        or proposal.end > len(target.text)
        or target.text[proposal.start : proposal.end] != proposal.source_slice
        or not _bounded_text(proposal.source_slice, plan.mention_limits.max_source_slice_bytes)
        or not _bounded_text(proposal.detector_label, plan.mention_limits.max_label_bytes)
    ):
        raise _ComponentPhase6Fault
    return _ProvisionalCandidate(
        _CandidateToken(),
        target.token,
        proposal.start,
        proposal.end,
        proposal.source_slice,
        proposal.detector_label,
        provenance,
    )


def _combine_detected(
    plan: _Phase6Plan,
    tokens: tuple[object, ...],
    store: _RuntimeStore,
) -> _DetectedGraph:
    selected = tuple(target for target in plan.targets if target.token in tokens)
    graphs = tuple(store.detected.get(target.datum_id) for target in selected)
    if any(graph is None for graph in graphs):
        raise _ComponentPhase6Fault
    mentions = tuple(mention for graph in graphs if graph is not None for mention in graph.mentions)
    return _DetectedGraph(selected, mentions)


def _target_for_datum(plan: _Phase6Plan, datum_id: _DatumId) -> _MentionTarget:
    target = next((candidate for candidate in plan.targets if candidate.datum_id == datum_id), None)
    if target is None:
        raise _GlobalPhase6Fault
    return target


def _component_for_target(plan: _Phase6Plan, target: _MentionTarget) -> _Phase6Component:
    component = next(
        (candidate for candidate in plan.components if target.token in candidate.target_tokens),
        None,
    )
    if component is None:
        raise _GlobalPhase6Fault
    return component


def _targets_for_component(
    plan: _Phase6Plan,
    component: _Phase6Component,
) -> tuple[_MentionTarget, ...]:
    return tuple(target for target in plan.targets if target.token in component.target_tokens)


def _verified_datum_predicate(datum_id: _DatumId, candidate: _Phase6Candidate) -> bool:
    return isinstance(candidate, _VerifiedDatum) and candidate.datum_id == datum_id


def _verified_group_predicate(outputs: tuple[tuple[_DatumId, _Phase6Candidate], ...]) -> bool:
    ids = tuple(datum_id for datum_id, _candidate in outputs)
    return len(set(ids)) == len(ids) and all(
        isinstance(candidate, _VerifiedDatum) and candidate.datum_id == datum_id for datum_id, candidate in outputs
    )


def _collect_released(
    plan: _Phase6Plan,
    accounting: _AccountingResult[_Phase6Candidate],
) -> tuple[_VerifiedDatum, ...]:
    released_by_id = {
        datum_id: candidate
        for group in accounting.groups
        if isinstance(group, _GroupReleased)
        for datum_id, candidate in group.outputs
        if isinstance(candidate, _VerifiedDatum)
    }
    return tuple(released_by_id[datum.id] for datum in plan.accounting.datums if datum.id in released_by_id)


def _bounded_text(value: object, limit: int) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        return len(value.encode("utf-8")) <= limit
    except UnicodeEncodeError:
        return False
