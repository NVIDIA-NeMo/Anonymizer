# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit bounded histories and support-preserving qualified commits."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from anonymizer.engine.graph_sdk._continuity_codec import _encode_state, _stable_ref, _text, _text_bound, _value_wire
from anonymizer.engine.graph_sdk._continuity_codec import encode_snapshot as encode_snapshot
from anonymizer.engine.graph_sdk._continuity_codec import load_snapshot as load_snapshot
from anonymizer.engine.graph_sdk._continuity_codec import write_snapshot as write_snapshot
from anonymizer.engine.graph_sdk._continuity_values import (
    _CONTINUITY_KEY,
    SNAPSHOT_FORMAT,
    SNAPSHOT_VERSION,
    AdmittedContinuity,
    ContinuityCommit,
    ContinuityDecisionDecl,
    ContinuityHandoff,
    ContinuityLimits,
    ContinuityScope,
    ContinuitySession,
    ContinuitySnapshot,
    ContinuitySupportDecl,
    ContinuityView,
    DecisionConflict,
    RetainedDecision,
    StateDecisionKey,
    StateDecisionRef,
    StateGroupRef,
    StateIdentity,
    StateNamespace,
    StateSubjectRef,
    StateSupportRef,
    TargetStateGroup,
    _key_order,
)
from anonymizer.engine.graph_sdk._effect_values import EffectCode, PrivateValue, reject, require_count, require_instance
from anonymizer.engine.graph_sdk._workflow import reachable_workflows
from anonymizer.engine.graph_sdk.data import ValidatedDataGraph
from anonymizer.engine.graph_sdk.executor import ExecutionResult, OperationOutputKey
from anonymizer.engine.graph_sdk.preparation import PreparedPlan
from anonymizer.engine.graph_sdk.qualification import QualificationResult, QualifiedOutput
from anonymizer.engine.graph_sdk.records import DecisionRef
from anonymizer.engine.graph_sdk.requests import ArtifactValue
from anonymizer.graph._values import DatumId
from anonymizer.graph.workflow import StateEffect

__all__ = [
    "AdmittedContinuity",
    "ContinuityCommit",
    "ContinuityDecisionDecl",
    "ContinuityHandoff",
    "ContinuityLimits",
    "ContinuityScope",
    "ContinuitySession",
    "ContinuitySnapshot",
    "ContinuityState",
    "ContinuitySupportDecl",
    "ContinuityView",
    "DecisionConflict",
    "RetainedDecision",
    "StateDecisionKey",
    "StateDecisionRef",
    "StateGroupRef",
    "StateIdentity",
    "StateNamespace",
    "StateSubjectRef",
    "StateSupportRef",
    "TargetStateGroup",
    "accept_handoff",
    "admit_continuity",
    "begin_continuity",
    "capture_snapshot",
    "commit_qualified",
    "create_continuity_state",
    "create_handoff",
    "encode_snapshot",
    "fork_continuity",
    "fresh_continuity",
    "load_snapshot",
    "restore_snapshot",
    "write_snapshot",
]


class ContinuityState(PrivateValue):
    """One explicitly shared state; equal namespaces never share history."""

    __slots__ = ("_identity", "_namespace", "_revision", "_decisions", "_lock")

    def __init__(
        self,
        *,
        _key: object,
        namespace: StateNamespace,
        revision: int = 0,
        decisions: tuple[RetainedDecision, ...] = (),
    ) -> None:
        if _key is not _CONTINUITY_KEY:
            raise TypeError("continuity state requires its factory")
        self._identity = StateIdentity._new()
        self._namespace = namespace
        self._revision = revision
        self._decisions = decisions
        self._lock = RLock()

    @property
    def identity(self) -> StateIdentity:
        return self._identity

    @property
    def namespace(self) -> StateNamespace:
        return self._namespace

    def view(self, *, limits: ContinuityLimits) -> ContinuityView:
        with self._lock:
            _encode_state(self.namespace, self._revision, self._decisions, limits)
            return ContinuityView(
                _key=_CONTINUITY_KEY,
                identity=self.identity,
                namespace=self.namespace,
                revision=self._revision,
                decisions=self._decisions,
            )

    def _capture(self, expected_revision: int, limits: ContinuityLimits) -> ContinuityView:
        require_count(expected_revision)
        with self._lock:
            view = self.view(limits=limits)
            if view.revision != expected_revision:
                reject(EffectCode.CONTRADICTORY)
            return view


def create_continuity_state(*, namespace: StateNamespace, limits: ContinuityLimits) -> ContinuityState:
    _encode_state(namespace, 0, (), limits)
    return ContinuityState(_key=_CONTINUITY_KEY, namespace=namespace)


def begin_continuity(
    *, data: ValidatedDataGraph, scope: ContinuityScope | None, limits: ContinuityLimits
) -> ContinuitySession:
    require_instance(data, ValidatedDataGraph)
    require_instance(limits, ContinuityLimits)
    if scope is not None:
        require_instance(scope, ContinuityScope)
        require_instance(scope.groups, tuple)
    if len(data.targets) > limits.max_targets or (scope is not None and len(scope.groups) > limits.max_targets):
        reject(EffectCode.LIMIT_EXCEEDED)
    groups = (
        tuple(TargetStateGroup(target=t, group=StateGroupRef(name=f"target-{i}")) for i, t in enumerate(data.targets))
        if scope is None
        else scope.groups
    )
    for group in groups:
        require_instance(group, TargetStateGroup)
        _text(group.group.name, limits.max_name_bytes)
    if len({group.group for group in groups}) > limits.max_groups:
        reject(EffectCode.LIMIT_EXCEEDED)
    if any(group.target.graph != data.graph or group.target not in data.targets for group in groups):
        reject(EffectCode.FOREIGN_OWNER)
    if len({group.target for group in groups}) != len(groups):
        reject(EffectCode.DUPLICATE)
    if {group.target for group in groups} != data.targets:
        reject(EffectCode.MISSING)
    lifetime = "per_call" if scope is None else scope.lifetime
    if lifetime not in {"per_call", "across_calls"}:
        reject(EffectCode.UNSUPPORTED)
    supplied = None if scope is None else scope.state
    if supplied is not None:
        require_instance(supplied, ContinuityState)
    if lifetime == "across_calls" and supplied is None:
        reject(EffectCode.MISSING)
    if lifetime == "per_call" and supplied is not None:
        reject(EffectCode.CONTRADICTORY)
    state = (
        supplied
        if supplied is not None
        else create_continuity_state(namespace=StateNamespace(name="per-call"), limits=limits)
    )
    base = state.view(limits=limits)
    return ContinuitySession(
        _key=_CONTINUITY_KEY, _state=state, data=data, lifetime=lifetime, base=base, groups=groups, limits=limits
    )


def admit_continuity(
    *,
    prepared: PreparedPlan,
    session: ContinuitySession,
    declarations: tuple[ContinuityDecisionDecl, ...],
    limits: ContinuityLimits,
) -> AdmittedContinuity:
    require_instance(prepared, PreparedPlan)
    require_instance(session, ContinuitySession)
    require_instance(limits, ContinuityLimits)
    require_instance(declarations, tuple)
    if len(declarations) > limits.max_decisions:
        reject(EffectCode.LIMIT_EXCEEDED)
    total = 0
    for declaration in declarations:
        require_instance(declaration, ContinuityDecisionDecl)
        total += len(declaration.supports)
        if len(declaration.supports) > limits.max_supports_per_decision or total > limits.max_total_supports:
            reject(EffectCode.LIMIT_EXCEEDED)
        _text_bound(declaration.key.group.name, limits.max_name_bytes)
        _text_bound(declaration.key.slot, limits.max_slot_bytes)
        _text_bound(declaration.key.subject.kind, limits.max_name_bytes)
        _text_bound(declaration.key.subject.name, limits.max_name_bytes)
        _text_bound(declaration.outcome, limits.max_name_bytes)
        _text_bound(declaration.port, limits.max_name_bytes)
        for support in declaration.supports:
            _text_bound(support.reference.kind, limits.max_name_bytes)
            _text_bound(support.reference.name, limits.max_name_bytes)
            _text_bound(support.outcome, limits.max_name_bytes)
            _text_bound(support.promise, limits.max_name_bytes)
    for declaration in declarations:
        _text(declaration.key.group.name, limits.max_name_bytes)
        _text(declaration.key.slot, limits.max_slot_bytes)
        _stable_ref(declaration.key.subject, limits)
        _text(declaration.outcome, limits.max_name_bytes)
        _text(declaration.port, limits.max_name_bytes)
        for support in declaration.supports:
            _stable_ref(support.reference, limits)
            _text(support.outcome, limits.max_name_bytes)
            _text(support.promise, limits.max_name_bytes)
    if prepared.data != session.data or prepared.data.graph != session.data.graph:
        reject(EffectCode.FOREIGN_OWNER)
    groups = {row.target: row.group for row in session.groups}
    nodes = {node.id: node for body in reachable_workflows(prepared.workflow) for node in body.nodes}
    for declaration in declarations:
        if declaration.target not in groups or declaration.node not in nodes:
            reject(EffectCode.FOREIGN_OWNER)
        if declaration.key.group != groups[declaration.target]:
            reject(EffectCode.FOREIGN_OWNER)
        if any(s.node not in nodes for s in declaration.supports):
            reject(EffectCode.FOREIGN_OWNER)
    locations = [(d.target, d.node, d.outcome, d.port) for d in declarations]
    if len(set(locations)) != len(locations) or len({d.key for d in declarations}) != len(declarations):
        reject(EffectCode.DUPLICATE)
    for declaration in declarations:
        refs = [(s.reference.kind, s.reference.name) for s in declaration.supports]
        bindings = [(s.node, s.outcome, s.promise) for s in declaration.supports]
        if len(set(refs)) != len(refs) or len(set(bindings)) != len(bindings):
            reject(EffectCode.DUPLICATE)
    for declaration in declarations:
        if not declaration.supports:
            reject(EffectCode.MISSING)
        outcomes = {o.name: o for o in nodes[declaration.node].operation.outcomes}
        outcome = outcomes.get(declaration.outcome)
        if outcome is None or declaration.port not in outcome.produced_ports:
            reject(EffectCode.MISSING)
        read = StateEffect(kind="read", name=session.base.namespace.name)
        required_effects = {read}
        if declaration.change != "reuse":
            required_effects.add(StateEffect(kind="write", name=session.base.namespace.name))
        if not required_effects <= outcome.state_effects:
            reject(EffectCode.MISSING)
        for support in declaration.supports:
            assessor = next((o for o in nodes[support.node].operation.outcomes if o.name == support.outcome), None)
            if assessor is None:
                reject(EffectCode.MISSING)
            promise = next((p for p in assessor.evidence if p.name == support.promise), None)
            if promise is None:
                reject(EffectCode.MISSING)
        if declaration.change in {"reuse", "supersede"} and declaration.prior is None:
            reject(EffectCode.MISSING)
    read_effect = StateEffect(kind="read", name=session.base.namespace.name)
    actual = [r.revision for r in prepared.state.revisions if r.effect == read_effect]
    if declarations and not actual:
        reject(EffectCode.MISSING)
    for declaration in declarations:
        if declaration.change not in {"create", "reuse", "supersede"}:
            reject(EffectCode.UNSUPPORTED)
        for support in declaration.supports:
            assessor = next(o for o in nodes[support.node].operation.outcomes if o.name == support.outcome)
            promise = next(p for p in assessor.evidence if p.name == support.promise)
            if assessor.category != "success" or not support.required_coverage <= promise.coverage:
                reject(EffectCode.UNSUPPORTED)
    for declaration in declarations:
        if declaration.change == "create" and declaration.prior is not None:
            reject(EffectCode.CONTRADICTORY)
        if declaration.prior is not None and declaration.prior.key != declaration.key:
            reject(EffectCode.CONTRADICTORY)
    if actual and actual != [session.base.revision + 1]:
        reject(EffectCode.CONTRADICTORY)
    if limits != session.limits:
        reject(EffectCode.CONTRADICTORY)
    return AdmittedContinuity(
        _key=_CONTINUITY_KEY, prepared=prepared, session=session, declarations=declarations, limits=limits
    )


@dataclass(frozen=True)
class _Proposal:
    runtime: DecisionRef
    declaration: ContinuityDecisionDecl
    value: ArtifactValue
    support_targets: tuple[frozenset[DatumId], ...]


def _proposals(
    admitted: AdmittedContinuity, execution: ExecutionResult, qualification: QualificationResult
) -> tuple[_Proposal, ...]:
    artifacts = dict(execution.artifacts)
    entries = {entry.activation: entry for state in execution.states for entry in state.entries}
    required = qualification.required_decisions
    if required != frozenset(d for target in qualification.qualified for d in target.required_decisions):
        reject(EffectCode.CONTRADICTORY)
    proposals = []
    for declaration in admitted.declarations:
        ports = [
            port
            for port in execution.ports
            if port.target == declaration.target
            and port.node == declaration.node
            and port.port == declaration.port
            and port.activation in entries
            and entries[port.activation].outcome == declaration.outcome
        ]
        matches = [port for port in ports if DecisionRef(artifact=port.artifact) in required]
        if not matches:
            continue  # An unrelated admitted provisional decision is not retained.
        if len(matches) != 1:
            reject(EffectCode.DUPLICATE)
        port = matches[0]
        provenance = [
            fact
            for fact in execution.provenance
            if fact.key == OperationOutputKey(activation=port.activation, target=port.target, port=port.port)
        ]
        if len(provenance) != 1:
            reject(EffectCode.MISSING if not provenance else EffectCode.DUPLICATE)
        if not provenance[0].decision or provenance[0].artifact != port.artifact or port.role != "decision":
            reject(EffectCode.CONTRADICTORY)
        decision = DecisionRef(artifact=port.artifact)
        value = artifacts.get(port.artifact)
        if value is None:
            reject(EffectCode.MISSING)
        _value_wire(value, admitted.limits)
        witnesses = []
        for support in declaration.supports:
            facts = [
                fact
                for fact in qualification.verified
                if fact.node == support.node
                and fact.outcome == support.outcome
                and fact.promise.name == support.promise
                and fact.finding.status == "satisfied"
                and support.required_coverage <= fact.coverage
                and decision in fact.reference.consumed
            ]
            targets = frozenset(
                target.target
                for target in qualification.qualified
                if decision in target.required_decisions
                and any(
                    fact.reference in target.evidence
                    and all(
                        not isinstance(ref, DecisionRef) or ref in target.required_decisions
                        for ref in fact.reference.consumed
                    )
                    for fact in facts
                )
            )
            if not targets:
                reject(EffectCode.MISSING)
            witnesses.append(targets)
        proposals.append(_Proposal(decision, declaration, value, tuple(witnesses)))
    refs = [p.runtime for p in proposals]
    if len(set(refs)) != len(refs):
        reject(EffectCode.DUPLICATE)
    if set(refs) != required:
        reject(EffectCode.MISSING)
    return tuple(proposals)


def _survivors(
    *,
    qualified: tuple[QualifiedOutput, ...],
    data: ValidatedDataGraph,
    supports: dict[DecisionRef, tuple[frozenset[DatumId], ...]],
    conflicts: frozenset[DecisionRef],
    maximum: int,
) -> tuple[frozenset[DatumId], int]:
    current = frozenset(t.target for t in qualified)
    steps = 0
    prerequisites = {
        t.target: frozenset(e.prerequisite for e in data.dependencies if e.dependent == t.target)
        | frozenset(peer for group in data.atomic if t.target in group for peer in group)
        for t in qualified
    }
    while current:
        following = set()
        for target in qualified:
            if target.target not in current:
                continue
            if steps == maximum:
                reject(EffectCode.LIMIT_EXCEEDED)
            steps += 1
            if (
                not target.required_decisions & conflicts
                and prerequisites[target.target] <= current
                and all(
                    all(witnesses & current for witnesses in supports[decision])
                    for decision in target.required_decisions
                )
            ):
                following.add(target.target)
        if following == current:
            break
        current = frozenset(following)
    return current, steps


def commit_qualified(
    *, admitted: AdmittedContinuity, execution: ExecutionResult, qualification: QualificationResult
) -> ContinuityCommit:
    require_instance(admitted, AdmittedContinuity)
    require_instance(execution, ExecutionResult)
    require_instance(qualification, QualificationResult)
    if (
        qualification._result is not execution
        or qualification._admitted.execution is not execution._execution
        or execution._execution.context.prepared is not admitted.prepared
    ):
        reject(EffectCode.FOREIGN_OWNER)
    session = admitted.session
    state = session._state
    if state.identity is not session.base.identity or state.namespace != session.base.namespace:
        reject(EffectCode.FOREIGN_OWNER)
    proposals = _proposals(admitted, execution, qualification)
    targets = tuple(target.target for target in qualification.qualified)
    with state._lock:
        if state._revision != session.base.revision:
            return ContinuityCommit(
                _key=_CONTINUITY_KEY,
                status="conflicted",
                execution=execution,
                qualification=qualification,
                identity=state.identity,
                namespace=state.namespace,
                base_revision=session.base.revision,
                resulting_revision=state._revision,
                retained=(),
                conflicts=(),
                releasable=(),
                withheld=targets,
            )
        current = {entry.reference.key: entry for entry in state._decisions}
        conflicts = []
        conflicting_refs = set()
        for proposal in proposals:
            declaration = proposal.declaration
            old = current.get(declaration.key)
            conflict = (
                old is not None
                if declaration.change == "create"
                else old is None
                or old.reference != declaration.prior
                or declaration.change == "reuse"
                and (
                    old.value != proposal.value or old.supports != frozenset(s.reference for s in declaration.supports)
                )
            )
            if conflict:
                conflicting_refs.add(proposal.runtime)
                conflicts.append(
                    DecisionConflict(
                        key=declaration.key,
                        expected=declaration.prior,
                        current=None if old is None else old.reference,
                        proposed_change=declaration.change,
                    )
                )
        surviving, _steps = _survivors(
            qualified=qualification.qualified,
            data=admitted.prepared.data,
            supports={p.runtime: p.support_targets for p in proposals},
            conflicts=frozenset(conflicting_refs),
            maximum=admitted.limits.max_commit_steps,
        )
        needed = frozenset(
            d for target in qualification.qualified if target.target in surviving for d in target.required_decisions
        )
        staged = dict(current)
        retained = []
        for proposal in proposals:
            if proposal.runtime not in needed:
                continue
            declaration = proposal.declaration
            old = current.get(declaration.key)
            supports = frozenset(s.reference for s in declaration.supports)
            changed = old is None or old.value != proposal.value or old.supports != supports
            revision = 1 if old is None else old.reference.revision + int(changed)
            entry = RetainedDecision(
                reference=StateDecisionRef(key=declaration.key, revision=revision),
                value=proposal.value,
                supports=supports,
            )
            staged[declaration.key] = entry
            retained.append(entry.reference)
        decisions = tuple(staged[key] for key in sorted(staged, key=_key_order))
        changed = decisions != state._decisions
        resulting_revision = state._revision + int(changed)
        _encode_state(state.namespace, resulting_revision, decisions, admitted.limits)
        withheld = tuple(target for target in targets if target not in surviving)
        status = (
            "conflicted"
            if withheld and not surviving
            else "partial"
            if withheld
            else "committed"
            if changed
            else "no_change"
        )
        receipt = ContinuityCommit(
            _key=_CONTINUITY_KEY,
            status=status,
            execution=execution,
            qualification=qualification,
            identity=state.identity,
            namespace=state.namespace,
            base_revision=session.base.revision,
            resulting_revision=resulting_revision,
            retained=tuple(sorted(retained, key=lambda ref: _key_order(ref.key))),
            conflicts=tuple(sorted(conflicts, key=lambda conflict: _key_order(conflict.key))),
            releasable=tuple(target for target in targets if target in surviving),
            withheld=withheld,
        )
        state._decisions, state._revision = decisions, resulting_revision
        return receipt


def capture_snapshot(*, state: ContinuityState, expected_revision: int, limits: ContinuityLimits) -> ContinuitySnapshot:
    require_instance(state, ContinuityState)
    view = state._capture(expected_revision, limits)
    return ContinuitySnapshot(
        _key=_CONTINUITY_KEY,
        format=SNAPSHOT_FORMAT,
        version=SNAPSHOT_VERSION,
        namespace=view.namespace,
        revision=view.revision,
        decisions=view.decisions,
    )


def restore_snapshot(*, snapshot: ContinuitySnapshot, limits: ContinuityLimits) -> ContinuityState:
    encode_snapshot(snapshot=snapshot, limits=limits)
    return ContinuityState(
        _key=_CONTINUITY_KEY, namespace=snapshot.namespace, revision=snapshot.revision, decisions=snapshot.decisions
    )


def fork_continuity(
    *, state: ContinuityState, expected_revision: int, namespace: StateNamespace, limits: ContinuityLimits
) -> ContinuityState:
    require_instance(state, ContinuityState)
    require_instance(namespace, StateNamespace)
    view = state._capture(expected_revision, limits)
    _encode_state(namespace, 1, view.decisions, limits)
    if namespace == state.namespace:
        reject(EffectCode.CONTRADICTORY)
    return ContinuityState(_key=_CONTINUITY_KEY, namespace=namespace, revision=1, decisions=view.decisions)


def fresh_continuity(*, namespace: StateNamespace, limits: ContinuityLimits) -> ContinuityState:
    return create_continuity_state(namespace=namespace, limits=limits)


def create_handoff(*, state: ContinuityState, expected_revision: int, limits: ContinuityLimits) -> ContinuityHandoff:
    require_instance(state, ContinuityState)
    view = state._capture(expected_revision, limits)
    return ContinuityHandoff(
        _key=_CONTINUITY_KEY,
        source=view.identity,
        namespace=view.namespace,
        base_revision=view.revision,
        decisions=view.decisions,
    )


def accept_handoff(*, handoff: ContinuityHandoff, limits: ContinuityLimits) -> ContinuityState:
    require_instance(handoff, ContinuityHandoff)
    _encode_state(handoff.namespace, handoff.base_revision, handoff.decisions, limits)
    return ContinuityState(
        _key=_CONTINUITY_KEY, namespace=handoff.namespace, revision=handoff.base_revision, decisions=handoff.decisions
    )
