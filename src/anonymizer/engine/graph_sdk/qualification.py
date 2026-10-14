# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure reconciliation and release qualification of one executed invocation."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Literal, get_args

from anonymizer.engine.graph_sdk._effect_values import EffectCode, PrivateValue, reject, require_instance
from anonymizer.engine.graph_sdk._execution_topology import _lexical_path
from anonymizer.engine.graph_sdk._execution_values import (
    AdaptiveResourceAcquisition,
    ImplementationResourceRegistration,
)
from anonymizer.engine.graph_sdk._qualification_provenance import _ProvenanceValidator
from anonymizer.engine.graph_sdk.evidence import (
    AdmittedQualification,
    AssessmentSubmission,
    EvidenceRevisionView,
    MapItemSubjectRef,
    VerifiedEvidence,
    _RunFacts,
    evidence_validity,
)
from anonymizer.engine.graph_sdk.executor import (
    ExecutionResult,
    FinalOutputFact,
    OperationOutputKey,
    ProvenanceKey,
)
from anonymizer.engine.graph_sdk.records import CandidateRef, CanonicalRecord, DecisionRef, EvidenceRef, TargetStatus
from anonymizer.engine.graph_sdk.requests import (
    BindingRequestScope,
    ChildEvidence,
    ExactUsage,
    InvocationRequestScope,
    PhysicalRequestId,
    RequestAssociation,
    RequestChain,
    RequestReceipt,
    RequestReservation,
    RequestTerminalFact,
    SemanticAssociation,
)
from anonymizer.engine.graph_sdk.resources import (
    CleanupAssociation,
    CleanupFact,
    ResourceId,
    ResourceLease,
    ResourceOwner,
)
from anonymizer.graph._values import ActivationKey, DatumId
from anonymizer.graph.workflow import (
    MapItemPort,
    NodeId,
    ProtectionRequirement,
    SubgraphNode,
)

WithholdingCode = Literal[
    "execution_only",
    "incomplete_membership",
    "terminal_failure",
    "missing_candidate",
    "missing_assessment",
    "assessment_unsatisfied",
    "assessment_unknown",
    "stale_evidence",
    "incomplete_coverage",
    "request_accounting",
    "cleanup_verification",
    "cleanup_accounting",
    "dependency",
    "atomic_group",
    "inconsistent_attribution",
]
_CODES = frozenset(get_args(WithholdingCode))
_TERMINAL = frozenset({"success", "failure", "cancelled", "lost", "blocked", "inconsistent"})
_QUALIFICATION_KEY = object()


def _matches_admitted_key(admitted: object, retained: object) -> bool:
    """Resolve a registration against admitted structure without hashing raw values.

    Only the admitted declaration supplies schema and collection bounds. Config
    descent is iterative; set matching recurses only through the fixed operation
    schema. A foreign shape is rejected before visiting its additional contents.
    """
    pending = [(admitted, retained)]
    while pending:
        expected, actual = pending.pop()
        if type(expected) is not type(actual):
            return False
        if isinstance(expected, tuple):
            assert isinstance(actual, tuple)
            if len(expected) != len(actual):
                return False
            pending.extend(zip(expected, actual, strict=True))
        elif isinstance(expected, frozenset):
            assert isinstance(actual, frozenset)
            if len(expected) != len(actual):
                return False
            unmatched = list(actual)
            for item in expected:
                index = next((i for i, value in enumerate(unmatched) if _matches_admitted_key(item, value)), None)
                if index is None:
                    return False
                unmatched.pop(index)
        elif is_dataclass(expected) and not isinstance(expected, type):
            for field in fields(expected):
                if not hasattr(actual, field.name):
                    return False
                pending.append((getattr(expected, field.name), getattr(actual, field.name)))
        elif expected != actual:
            return False
    return True


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class TargetQualification(PrivateValue):
    target: DatumId
    candidate: CandidateRef | None
    verified: frozenset[EvidenceRef]
    required_decisions: frozenset[DecisionRef]
    withholding: frozenset[WithholdingCode]

    def __post_init__(self) -> None:
        require_instance(self.target, DatumId)
        if self.candidate is not None:
            require_instance(self.candidate, CandidateRef)
        require_instance(self.verified, frozenset)
        require_instance(self.required_decisions, frozenset)
        require_instance(self.withholding, frozenset)
        if (
            any(not isinstance(item, EvidenceRef) for item in self.verified)
            or any(not isinstance(item, DecisionRef) for item in self.required_decisions)
            or any(not isinstance(item, str) for item in self.withholding)
        ):
            reject(EffectCode.INVALID_TYPE)
        if not self.withholding <= _CODES:
            reject(EffectCode.INVALID_VALUE)
        if self.candidate is not None and self.candidate.target != self.target:
            reject(EffectCode.FOREIGN_OWNER)
        if self.candidate is not None and any(
            item.artifact.invocation != self.candidate.artifact.invocation
            for item in (*self.verified, *self.required_decisions)
        ):
            reject(EffectCode.FOREIGN_OWNER)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class QualifiedOutput(PrivateValue):
    target: DatumId
    candidate: CandidateRef
    evidence: frozenset[EvidenceRef]
    required_decisions: frozenset[DecisionRef]

    def __post_init__(self) -> None:
        require_instance(self.candidate, CandidateRef)
        TargetQualification(
            target=self.target,
            candidate=self.candidate,
            verified=self.evidence,
            required_decisions=self.required_decisions,
            withholding=frozenset(),
        )


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class QualificationResult(PrivateValue):
    record: CanonicalRecord
    verified: tuple[VerifiedEvidence, ...]
    targets: tuple[TargetQualification, ...]
    qualified: tuple[QualifiedOutput, ...]
    required_decisions: frozenset[DecisionRef]
    _admitted: AdmittedQualification
    _result: ExecutionResult
    _current: EvidenceRevisionView

    def __init__(self, *, _key: object, **values: object) -> None:
        if _key is not _QUALIFICATION_KEY:
            raise TypeError("qualification results require qualify")
        for name, value in values.items():
            object.__setattr__(self, name, value)


def qualify_retained(
    *, admitted: AdmittedQualification, result: ExecutionResult, current: EvidenceRevisionView
) -> QualificationResult:
    """Qualify a saved result using all retained assessments and explicit current revisions."""
    return qualify(admitted=admitted, result=result, current=current, submissions=None)


def qualify(
    *,
    admitted: AdmittedQualification,
    result: ExecutionResult,
    current: EvidenceRevisionView,
    submissions: tuple[AssessmentSubmission, ...] | None,
) -> QualificationResult:
    """Reconcile execution and release only targets supported by current evidence."""
    require_instance(admitted, AdmittedQualification)
    require_instance(result, ExecutionResult)
    require_instance(current, EvidenceRevisionView)
    if submissions is not None:
        require_instance(submissions, tuple)
    count = len(result.assessments) if submissions is None else len(submissions)
    if count > min(admitted.limits.max_submissions, admitted.limits.max_verified_evidence):
        reject(EffectCode.LIMIT_EXCEEDED)
    if submissions is None:
        submissions = tuple(AssessmentSubmission(fact=fact) for fact in result.assessments)
    if admitted.execution.context.prepared.configuration.purpose != "execution_only" and any(
        not isinstance(item, AssessmentSubmission) for item in submissions
    ):
        reject(EffectCode.INVALID_TYPE)
    if current._admitted is not admitted or current._result is not result:
        reject(EffectCode.FOREIGN_OWNER)
    facts = _RunFacts.from_result(admitted, result)
    qualification = _Qualification(admitted, result, current, facts)
    qualification.reconcile()
    qualification.validate_provenance()
    if admitted.execution.context.prepared.configuration.purpose == "execution_only":
        for codes in qualification.withholding.values():
            codes.add("execution_only")
        return qualification.finish(())
    for target in qualification.order:
        qualification.select_candidate(target)
    verified = facts.verify_submissions(submissions)
    qualification.validate_assessment_inventory()
    qualification.request_accounting()
    if not qualification.cleanup_accounting():
        # Evidence hashes retain the whole result. An unresolved declaration
        # must never reach those indexes, even when every target is withheld.
        qualification.propagate()
        return qualification.finish(())
    for target in qualification.order:
        qualification.assess(target, verified)
    qualification.propagate()
    return qualification.finish(verified)


class _Qualification:
    """One invocation's indexes and release decisions; never executes work."""

    def __init__(
        self,
        admitted: AdmittedQualification,
        result: ExecutionResult,
        current: EvidenceRevisionView,
        facts: _RunFacts,
    ) -> None:
        self.admitted = admitted
        self.result = result
        self.current = current
        self.facts = facts
        self.prepared = admitted.execution.context.prepared
        self.order = tuple(item.target for item in self.prepared.target_occurrences)
        self.nodes = admitted.execution.nodes
        self.recovered_children: set[PhysicalRequestId] = set()
        self.permissive_usage: set[PhysicalRequestId] = set()
        self.failed_terminals: set[ActivationKey] = set()
        self.withholding: dict[DatumId, set[WithholdingCode]] = {target: set() for target in self.order}
        self.closed = {target: True for target in self.order}
        self.uncertain: set[DatumId] = set()
        self.candidates: dict[DatumId, CandidateRef] = {}
        self.supporting: dict[DatumId, set[VerifiedEvidence]] = {target: set() for target in self.order}
        self.outputs: dict[DatumId, tuple[FinalOutputFact, ...]] = {}

    def incomplete(self, target: DatumId) -> None:
        self.closed[target] = False
        self.withholding[target].add("incomplete_membership")

    def affected(self, activation: ActivationKey) -> frozenset[DatumId]:
        return frozenset(self.order) if activation in self.facts.shared else frozenset({self.facts.owners[activation]})

    def offset(self, activation: ActivationKey) -> int:
        if activation in self.facts.shared:
            return self.prepared.invocation_offset
        return next(
            item.occurrence_offset
            for item in self.prepared.target_occurrences
            if item.target == self.facts.owners[activation]
        )

    def reconcile(self) -> None:
        record = self.result.record
        for target, state in zip(self.order, self.result.states, strict=True):
            if not state.complete:
                self.incomplete(target)
        if self.result.invocation_state is not None and not self.result.invocation_state.complete:
            for target in self.order:
                self.incomplete(target)
        self._memberships()
        self._terminals()
        recovered_leaves = self._child_receipts()
        for key in self.failed_terminals - recovered_leaves:
            for target in self.affected(key):
                self.withholding[target].add("terminal_failure")
        for pending in self.result.pending_decisions:
            target = self.facts.owners.get(pending.activation)
            if target is None:
                reject(EffectCode.FOREIGN_OWNER)
            self.incomplete(target)
        if {item.target for item in record.statuses} != set(self.order):
            reject(EffectCode.MISSING)

    def _memberships(self) -> None:
        expected: dict[ActivationKey | None, set[ActivationKey]] = {None: set()}
        closed: dict[ActivationKey | None, bool] = {
            None: all(state.complete for state in self.result.states)
            and (self.result.invocation_state is None or self.result.invocation_state.complete)
        }
        for key, entry in self.facts.entries.items():
            if key.parent is not None and (
                key.parent not in self.facts.entries or self.facts.owners.get(key.parent) != self.facts.owners.get(key)
            ):
                reject(EffectCode.FOREIGN_OWNER)
            expected.setdefault(key.parent, set()).add(key)
            if key.parent is not None:
                closed[key.parent] = self.facts.entries[key.parent].status in _TERMINAL
        for state in self.result.states:
            for expansion in state.expansions:
                if expansion.parent not in self.facts.entries:
                    reject(EffectCode.MISSING)
                expected.setdefault(expansion.parent, set())
                closed[expansion.parent] = expansion.status != "pending"
                parent = self.facts.entries[expansion.parent]
                scope = next(
                    scope for scope in self.prepared.workflow.bodies() if scope.workflow == parent.template.workflow
                )
                member_templates = {item.member for item in scope.maps if item.expander == parent.template} | {
                    item.member for item in scope.loops if item.starter == parent.template
                }
                dynamic_members = frozenset(
                    member
                    for member in expected[expansion.parent]
                    if self.facts.entries[member].template in member_templates
                )
                if expansion.members != dynamic_members:
                    self.incomplete(self.facts.owners[expansion.parent])
                if not expansion.members and expansion.status == "closed":
                    admitted_outcomes = {
                        outcome
                        for item in scope.maps
                        if item.expander == parent.template
                        for outcome in item.expansion_outcomes
                    }
                    admitted_outcomes.update(
                        outcome
                        for item in scope.loops
                        if item.starter == parent.template
                        for outcome in (*item.enter_outcomes, *item.bypass_outcomes)
                    )
                    if parent.status != "success" or parent.outcome not in admitted_outcomes:
                        self.incomplete(self.facts.owners[expansion.parent])
        observed = {item.parent: item for item in self.result.record.memberships}
        if len(observed) != len(self.result.record.memberships):
            reject(EffectCode.DUPLICATE)
        if set(observed) - set(expected):
            reject(EffectCode.CONTRADICTORY)
        all_members: set[ActivationKey] = set()
        for parent, members in expected.items():
            membership = observed.get(parent)
            affected = set(self.order) if parent is None else self.affected(parent)
            if membership is None or membership.members != frozenset(members) or membership.closed != closed[parent]:
                for target in affected:
                    self.incomplete(target)
            if membership is not None:
                if all_members & membership.members:
                    reject(EffectCode.DUPLICATE)
                if any(member not in self.facts.entries for member in membership.members):
                    reject(EffectCode.MISSING)
                all_members.update(membership.members)

    def _terminals(self) -> None:
        terminals = self.facts.terminals
        if set(terminals) - set(self.facts.entries):
            reject(EffectCode.MISSING)
        for key, entry in self.facts.entries.items():
            terminal = terminals.get(key)
            if terminal is None:
                for target in self.affected(key):
                    self.incomplete(target)
                continue
            terminal.__post_init__()
            structural = isinstance(self.nodes[entry.template], SubgraphNode)
            if terminal.structural != structural or terminal.category != entry.status:
                reject(EffectCode.CONTRADICTORY)
            if (
                structural
                and terminal.attempt is not None
                or not structural
                and entry.status == "success"
                and terminal.attempt is None
            ):
                reject(EffectCode.CONTRADICTORY)
            if entry.outcome is not None:
                outcome = next(
                    (item for item in self.nodes[entry.template].operation.outcomes if item.name == entry.outcome), None
                )
                if outcome is None or outcome.category != entry.status:
                    reject(EffectCode.CONTRADICTORY)
            elif entry.status == "success":
                reject(EffectCode.CONTRADICTORY)
            if entry.status != "success":
                self.failed_terminals.add(key)

    def _child_receipts(self) -> set[ActivationKey]:
        """Independently join retained child requests to the admitted fixed recipe."""
        receipt = self.result.requests
        policies = {policy.node: policy for policy in self.admitted.execution.policies}
        delegated = [policy for policy in policies.values() if policy.delegated is not None]
        if (
            not delegated
            and not receipt.chains
            and not receipt.child_evidence
            and not any(dispatch.child is not None for dispatch in receipt.dispatches)
        ):
            return set()
        bound = self.prepared.declared_request_upper_bound
        if bound is None:
            if (
                self.prepared.configuration.purpose != "execution_only"
                or not isinstance(receipt.scope, InvocationRequestScope)
                or receipt.scope.invocation != self.result.record.invocation
                or receipt.dispatched_count != 0
                or receipt.budget_limit is not None
                or any(
                    (
                        receipt.dispatches,
                        receipt.chains,
                        receipt.child_evidence,
                        receipt.terminals,
                        receipt.settlements,
                        receipt.defects,
                        receipt.bindings,
                        receipt.policies,
                        receipt.denials,
                        receipt.local_in_flight,
                        receipt.remote_outstanding,
                        receipt.cancel_requested,
                    )
                )
            ):
                reject(EffectCode.CONTRADICTORY)
            return set()
        containers = (
            receipt.dispatches,
            receipt.terminals,
            receipt.settlements,
            receipt.chains,
            receipt.local_in_flight,
            receipt.remote_outstanding,
            receipt.cancel_requested,
            receipt.denials,
        )
        if any(len(items) > bound for items in containers) or len(receipt.child_evidence) > 2 * bound:
            reject(EffectCode.LIMIT_EXCEEDED)
        if (
            len(receipt.defects) > bound
            or len(receipt.bindings) > self.prepared.total_activation_slots
            or len(receipt.policies) > self.prepared.limits.max_capabilities
            or any(len(binding.policies) > self.prepared.limits.max_capabilities for binding in receipt.bindings)
        ):
            reject(EffectCode.LIMIT_EXCEEDED)
        terminals = {item.attempt: item for item in self.result.record.terminals if item.attempt is not None}
        chains = {chain.chain: chain for chain in receipt.chains}
        if len(chains) != len(receipt.chains) or len(set(receipt.child_evidence)) != len(receipt.child_evidence):
            reject(EffectCode.DUPLICATE)
        expected: set[ActivationKey] = set()
        readiness_owners = {}
        readiness_dependents: set[ActivationKey] = set()
        physical_owners = {}
        probes: set[ActivationKey] = set()
        for owner in self.facts.entries.values():
            policy = policies.get(owner.template)
            if policy is None or policy.delegated is None:
                continue
            contract = policy.delegated
            offset = self.offset(owner.activation)
            descriptor = contract.readiness
            if descriptor is not None:
                readiness_offset = self.prepared.invocation_offset if self.facts.shared else offset
                semantic = next(
                    entry.activation
                    for entry in self.facts.entries.values()
                    if entry.activation.occurrence == readiness_offset + descriptor.owner_slot
                )
                physical = next(
                    entry.activation
                    for entry in self.facts.entries.values()
                    if entry.activation.occurrence == readiness_offset + descriptor.request_slot
                )
                probe = next(
                    entry.activation
                    for entry in self.facts.entries.values()
                    if entry.activation.occurrence == readiness_offset + descriptor.probe_slot
                )
                readiness_owners[semantic] = contract
                readiness_dependents.add(owner.activation)
                probes.add(probe)
                physical_owners[physical] = semantic
                expected.add(physical)
            for entry in self.facts.entries.values():
                if (
                    entry.activation.occurrence - offset in contract.child_slots
                    and self.facts.owners.get(entry.activation) == self.facts.owners[owner.activation]
                ):
                    physical_owners[entry.activation] = owner.activation
                    expected.add(entry.activation)
        used: set[ActivationKey] = set()
        recovered: set[ActivationKey] = set()
        known_evidence: set[ChildEvidence] = set()
        processed: set[PhysicalRequestId] = set()
        used_groups: set[tuple[ActivationKey, tuple[int, ...]]] = set()
        for chain in receipt.chains:
            chain.__post_init__()
            if (
                chain.chain.scope != receipt.scope
                or len(chain.associations) != 1
                or chain.role not in {"generation", "readiness"}
            ):
                reject(EffectCode.UNSUPPORTED)
            association = next(iter(chain.associations))
            if not isinstance(association, SemanticAssociation) or association.task not in terminals:
                reject(EffectCode.FOREIGN_OWNER)
            owner = terminals[association.task].activation
            owner_policy = policies[self.facts.entries[owner].template]
            contract = owner_policy.delegated
            slots = () if contract is None else contract.child_slots
            routes = () if contract is None else contract.routes
            followups = frozenset() if contract is None else contract.followups
            if chain.role == "readiness":
                relation = readiness_owners.get(owner)
                if relation is None:
                    reject(EffectCode.FOREIGN_OWNER)
                contract = relation
                descriptor = contract.readiness
                assert descriptor is not None
                slots, routes, followups = (descriptor.request_slot,), (descriptor.route,), frozenset()
            elif contract is not None and contract.readiness is not None:
                descriptor = contract.readiness
                offset = self.prepared.invocation_offset if self.facts.shared else self.offset(owner)
                for index, outcomes in (
                    (descriptor.owner_slot, frozenset({descriptor.ready_outcome})),
                    (descriptor.probe_slot, descriptor.probe_outcomes),
                ):
                    dependency = next(
                        entry for entry in self.facts.entries.values() if entry.activation.occurrence == offset + index
                    )
                    if (
                        dependency.status != "success"
                        or dependency.outcome not in outcomes
                        or not any(
                            row.activation == dependency.activation and row.attempt is not None
                            for row in terminals.values()
                        )
                    ):
                        if any(
                            item.child is not None and item.child.chain == chain.chain for item in receipt.dispatches
                        ):
                            reject(EffectCode.CONTRADICTORY)
            if (
                contract is None
                or contract.mode != "qualified"
                or owner_policy.request != chain.policy
                or routes != chain.routes
                or followups != chain.followups
            ):
                reject(EffectCode.CONTRADICTORY)
            history = [
                item for item in receipt.dispatches if item.child is not None and item.child.chain == chain.chain
            ]
            if not history and (
                (contract.readiness is not None and self.facts.entries[owner].status != "success")
                or contract.selection is not None
            ):
                continue
            if not history or len(history) > chain.policy.max_attempts:
                reject(EffectCode.LIMIT_EXCEEDED)
            offset = self.offset(owner)
            if chain.role == "generation" and contract.generation_groups:
                first = history[0].child
                if first is None or first.leaf is None:
                    reject(EffectCode.MISSING)
                groups = [group for group in contract.generations if first.leaf.activation.occurrence - offset in group]
                if len(groups) != 1 or (owner, groups[0]) in used_groups:
                    reject(EffectCode.CONTRADICTORY)
                slots = groups[0]
                used_groups.add((owner, slots))
            prior: RequestReservation | None = None
            for dispatch in history:
                dispatch.__post_init__()
                child = dispatch.child
                assert child is not None
                child.__post_init__()
                leaf = child.leaf
                if (
                    leaf is None
                    or leaf not in terminals
                    or leaf.activation not in expected
                    or leaf.activation in used
                    or self.facts.owners.get(leaf.activation) != self.facts.owners.get(owner)
                    or leaf.activation.parent != owner.parent
                    or leaf.activation.occurrence - offset not in slots
                ):
                    reject(EffectCode.FOREIGN_OWNER)
                if (
                    contract.optional_tail
                    and chain.role == "generation"
                    and (
                        leaf.activation.occurrence - offset != contract.child_slots[0 if prior is None else 1]
                        or child.source_kind != ("initial" if prior is None else "correction")
                    )
                ):
                    reject(EffectCode.CONTRADICTORY)
                slot = self.prepared.reservation_recipe[leaf.activation.occurrence - offset]
                if slot.parent_index is not None:
                    if owner.parent is None or owner.parent.occurrence != offset + slot.parent_index:
                        reject(EffectCode.FOREIGN_OWNER)
                elif owner.parent is not None:
                    reject(EffectCode.FOREIGN_OWNER)
                if (
                    dispatch.request in processed
                    or dispatch.associations != chain.associations
                    or dispatch.policy != chain.policy
                    or child.role != chain.role
                ):
                    reject(EffectCode.CONTRADICTORY)
                facts = [item for item in receipt.terminals if item.request == dispatch.request]
                settlements = [item for item in receipt.settlements if item.request == dispatch.request]
                if len(facts) != 1 or terminals[leaf].category != facts[0].category:
                    reject(EffectCode.CONTRADICTORY)
                facts[0].__post_init__()
                unsettled_loss = (
                    facts[0].category == "lost" and not settlements and dispatch.request in receipt.remote_outstanding
                )
                if len(settlements) != 1 and not unsettled_loss:
                    reject(EffectCode.CONTRADICTORY)
                for settlement in settlements:
                    settlement.__post_init__()
                for summary in facts[0].results:
                    summary.__post_init__()
                    if summary.category == "success" and (
                        summary.association != association
                        or summary.outcome != self.facts.entries[leaf.activation].outcome
                        or (
                            self.facts.entries[owner].status == "success"
                            and summary.outcome != self.facts.entries[owner].outcome
                        )
                        or summary.outcome not in owner_policy.result_outcomes
                        or not any(
                            outcome.name == summary.outcome and outcome.category == "success"
                            for outcome in self.nodes[self.facts.entries[owner].template].operation.outcomes
                        )
                    ):
                        reject(EffectCode.CONTRADICTORY)
                if self.facts.entries[leaf.activation].status not in {
                    "success",
                    "failure",
                    "lost",
                    "cancelled",
                    "inconsistent",
                }:
                    reject(EffectCode.CONTRADICTORY)
                if not self._child_edge(dispatch, prior, history, chain, receipt):
                    reject(EffectCode.CONTRADICTORY)
                for evidence in receipt.child_evidence:
                    if evidence.chain != chain.chain or evidence.request != dispatch.request:
                        continue
                    evidence.__post_init__()
                    if (
                        facts[0].category != "failure"
                        or facts[0].failure != "malformed_response"
                        or not settlements
                        or settlements[0].disposition != "completed"
                        or any(item.request == dispatch.request for item in receipt.defects)
                    ):
                        reject(EffectCode.CONTRADICTORY)
                    if evidence.kind == "generation_exhausted":
                        route = next(item for item in chain.routes if item.name == child.route)
                        parsed = [
                            item
                            for item in history[: history.index(dispatch) + 1]
                            if item.child is not None
                            and item.child.route == child.route
                            and ChildEvidence(chain=chain.chain, request=item.request, kind="parser_rejected")
                            in known_evidence
                        ]
                        if (
                            len(parsed) != route.parse_attempts
                            or ChildEvidence(chain=chain.chain, request=dispatch.request, kind="parser_rejected")
                            not in known_evidence
                        ):
                            reject(EffectCode.CONTRADICTORY)
                    known_evidence.add(evidence)
                if chain.role == "generation" and contract.usage_requirement == "unknown_allowed":
                    self.permissive_usage.add(dispatch.request)
                used.add(leaf.activation)
                processed.add(dispatch.request)
                prior = dispatch
            last = history[-1]
            terminal = next(item for item in receipt.terminals if item.request == last.request)
            if terminal.category == "success" and any(
                item.association == association and item.category == "success" for item in terminal.results
            ):
                for dispatch in history[:-1]:
                    child = dispatch.child
                    assert child is not None and child.leaf is not None
                    fact = next(item for item in receipt.terminals if item.request == dispatch.request)
                    if fact.category == "failure":
                        self.recovered_children.add(dispatch.request)
                        recovered.add(child.leaf.activation)
        for entry in self.facts.entries.values():
            policy = policies.get(entry.template)
            if policy is None or policy.delegated is None or not policy.delegated.generation_groups:
                continue
            entered = any(row.activation == entry.activation for row in terminals.values())
            own_chains = [
                chain
                for chain in receipt.chains
                if chain.role == "generation"
                and any(
                    isinstance(association, SemanticAssociation) and association.task.activation == entry.activation
                    for association in chain.associations
                )
            ]
            if len(own_chains) != (len(policy.delegated.generations) if entered else 0):
                reject(EffectCode.MISSING)
        omitted = self._unused_children(expected - used, physical_owners)
        omitted |= self._unselected_generations(expected - used, physical_owners)
        unused = expected - used - omitted
        if any(
            physical_owners[key] not in readiness_dependents | readiness_owners.keys()
            or self.facts.entries[key].status != "blocked"
            or self.facts.entries[physical_owners[key]].status == "success"
            or any(row.activation == key for row in terminals.values())
            for key in unused
        ):
            reject(EffectCode.CONTRADICTORY)
        if known_evidence != set(receipt.child_evidence) or used - expected:
            reject(EffectCode.CONTRADICTORY)
        for dispatch in receipt.dispatches:
            if dispatch.child is not None and dispatch.request not in processed:
                reject(EffectCode.MISSING)
            for association in dispatch.associations:
                # Declared physical slots belong exclusively to their child dispatch;
                # no ordinary or shared dispatch may reuse them as semantic owners.
                if isinstance(association, SemanticAssociation) and association.task.activation in expected | probes:
                    reject(EffectCode.CONTRADICTORY)
                if isinstance(association, SemanticAssociation) and association.task in terminals:
                    node = self.facts.entries[terminals[association.task].activation].template
                    if (
                        policies[node].delegated is not None or association.task.activation in readiness_owners
                    ) and dispatch.request not in processed:
                        reject(EffectCode.MISSING)
        return recovered | omitted

    def _unselected_generations(
        self, unused: set[ActivationKey], owners: dict[ActivationKey, ActivationKey]
    ) -> set[ActivationKey]:
        accepted: set[ActivationKey] = set()
        policies = {policy.node: policy for policy in self.admitted.execution.policies}
        terminals = self.facts.terminals
        values = self.facts.values
        for owner in set(owners.values()):
            contract = policies[self.facts.entries[owner].template].delegated
            if contract is None or contract.selection is None or self.facts.entries[owner].status != "success":
                continue
            selection = contract.selection
            ports = [port for port in self.result.ports if port.activation == owner and port.port == selection.port]
            if len(ports) != 1 or ports[0].artifact not in values:
                reject(EffectCode.MISSING)
            port = ports[0]
            expected = OperationOutputKey(target=self.facts.owners[owner], activation=owner, port=selection.port)
            provenance = [fact for fact in self.result.provenance if fact.key == expected]
            if len(provenance) != 1 or provenance[0].artifact != port.artifact:
                reject(EffectCode.MISSING)
            omitted = selection.omitted(values[ports[0].artifact])
            offset = self.offset(owner)
            for slot in omitted:
                matches = [key for key, parent in owners.items() if parent == owner and key.occurrence == offset + slot]
                if len(matches) != 1 or matches[0] not in unused:
                    reject(EffectCode.CONTRADICTORY)
                key = matches[0]
                terminal = terminals.get(key)
                if (
                    terminal is None
                    or terminal.attempt is not None
                    or terminal.category != "blocked"
                    or terminal.reasons != {"omitted_optional"}
                ):
                    reject(EffectCode.CONTRADICTORY)
                accepted.add(key)
        return accepted

    def _unused_children(
        self, unused: set[ActivationKey], owners: dict[ActivationKey, ActivationKey]
    ) -> set[ActivationKey]:
        """Authenticate unused capacity from admitted recipes and retained facts only."""
        receipt = self.result.requests
        terminals = self.facts.terminals
        accepted: set[ActivationKey] = set()
        policies = {row.node: row for row in self.admitted.execution.policies}
        for key, terminal in terminals.items():
            cause = terminal.unused_retry
            if cause is None:
                continue
            owner, generation = cause.owner.activation, cause.after.activation
            if (
                key not in unused
                or owners.get(key) != owner
                or self.facts.owners.get(generation) != self.facts.owners[key]
            ):
                reject(EffectCode.FOREIGN_OWNER)
            contract = policies[self.facts.entries[owner].template].delegated
            assert contract is not None
            offset = next(
                row.occurrence_offset
                for row in self.prepared.target_occurrences
                if row.target == self.facts.owners[key]
            )
            group = next((group for group in contract.generations if key.occurrence - offset in group), ())
            if (
                not group
                or generation.occurrence - offset not in group
                or group.index(generation.occurrence - offset) >= group.index(key.occurrence - offset)
                or key.parent != owner.parent
                or generation.parent != owner.parent
                or terminal.attempt is not None
                or self.facts.entries[key].outcome is not None
                or self.facts.entries[key].status != "blocked"
                or not any(
                    isinstance(row.key, OperationOutputKey) and row.key.activation == owner
                    for row in self.result.provenance
                )
                or any(
                    terminals.get(attempt.activation) is None
                    or terminals[attempt.activation].attempt != attempt
                    or terminals[attempt.activation].category != "success"
                    for attempt in (cause.owner, cause.after)
                )
            ):
                reject(EffectCode.CONTRADICTORY)
            association = SemanticAssociation(task=cause.owner)
            chains = [
                row
                for row in receipt.chains
                if row.associations == {association}
                and any(
                    item.child is not None and item.child.chain == row.chain and item.child.leaf == cause.after
                    for item in receipt.dispatches
                )
            ]
            if len(chains) != 1:
                reject(EffectCode.CONTRADICTORY)
            history = [
                row for row in receipt.dispatches if row.child is not None and row.child.chain == chains[0].chain
            ]
            if (
                not history
                or history[-1].child is None
                or history[-1].child.leaf != cause.after
                or tuple(
                    item.child.leaf.activation.occurrence - offset
                    for item in history
                    if item.child is not None and item.child.leaf is not None
                )
                != group[: group.index(generation.occurrence - offset) + 1]
            ):
                reject(EffectCode.CONTRADICTORY)
            request = history[-1].request
            settlements = [row for row in receipt.settlements if row.request == request]
            physical = [row for row in receipt.terminals if row.request == request]
            if (
                (
                    contract.optional_tail
                    and (
                        receipt.denials
                        or receipt.defects
                        or receipt.cancel_requested
                        or receipt.local_in_flight
                        or receipt.remote_outstanding
                    )
                )
                or len(settlements) != 1
                or settlements[0].disposition != "completed"
                or settlements[0].remote_stopped is not True
                or (
                    contract.optional_tail
                    and contract.usage_requirement == "exact"
                    and not isinstance(settlements[0].usage, ExactUsage)
                )
                or len(physical) != 1
                or physical[0].category != "success"
                or receipt.dispatched_count != len(receipt.dispatches)
                or len(receipt.terminals) != len(receipt.dispatches)
                or (contract.optional_tail and len(receipt.settlements) != len(receipt.dispatches))
                or any(
                    isinstance(row.association, SemanticAssociation) and row.association.task.activation == key
                    for row in receipt.bindings
                )
            ):
                reject(EffectCode.CONTRADICTORY)
            accepted.add(key)
        return accepted

    def _child_edge(
        self,
        dispatch: RequestReservation,
        prior: RequestReservation | None,
        history: list[RequestReservation],
        chain: RequestChain,
        receipt: RequestReceipt,
    ) -> bool:
        child = dispatch.child
        assert child is not None
        purpose = "correction" if child.source_kind == "conversation_restart" else child.source_kind
        if dispatch.purpose != purpose or child.route not in {route.name for route in chain.routes}:
            return False
        if prior is None:
            return (
                child.source_kind == "initial"
                and child.predecessor is None
                and child.evidence is None
                and (child.route == chain.routes[0].name)
            )
        if child.source_kind not in chain.followups or child.predecessor != prior.request or prior.child is None:
            return False
        terminal = next((item for item in receipt.terminals if item.request == prior.request), None)
        if terminal is None:
            return False
        if child.source_kind == "failover":
            names = [route.name for route in chain.routes]
            index = names.index(prior.child.route)
            return (
                index + 1 < len(names)
                and child.route == names[index + 1]
                and child.evidence
                == ChildEvidence(chain=chain.chain, request=prior.request, kind="generation_exhausted")
                and child.evidence in receipt.child_evidence
                and chain.policy.replay == "idempotent"
            )
        if child.route != prior.child.route or not all(
            self._followup(dispatch, terminal, item) for item in chain.associations
        ):
            return False
        if child.source_kind == "retry":
            return child.evidence is None
        route = next(item for item in chain.routes if item.name == child.route)
        parsed = sum(
            item.child is not None
            and item.child.route == child.route
            and ChildEvidence(chain=chain.chain, request=item.request, kind="parser_rejected") in receipt.child_evidence
            for item in history[: history.index(dispatch)]
        )
        return (
            parsed < route.parse_attempts
            and child.evidence == ChildEvidence(chain=chain.chain, request=prior.request, kind="parser_rejected")
            and child.evidence in receipt.child_evidence
        )

    def validate_assessment_inventory(self) -> None:
        """Require one retained fact per successful assessment-producing occurrence."""
        terminals = {item.activation: item for item in self.result.record.terminals}
        possible = {
            (activation, entry.template, entry.outcome, production.promise)
            for activation, entry in self.facts.entries.items()
            if entry.status == "success" and not isinstance(self.nodes[entry.template], SubgraphNode)
            for production in self.admitted.execution.assessment_productions
            if production.node == entry.template and production.outcome == entry.outcome
        }
        expected = {owner for owner in possible if owner[0] in terminals}
        retained = [(fact.activation, fact.node, fact.outcome, fact.promise) for fact in self.result.assessments]
        observed = set(retained)
        if len(retained) != len(observed):
            reject(EffectCode.DUPLICATE)
        if any(fact.activation.invocation != self.result.record.invocation for fact in self.result.assessments):
            reject(EffectCode.FOREIGN_OWNER)
        if any(fact.activation not in self.facts.entries for fact in self.result.assessments):
            reject(EffectCode.MISSING)
        if expected - observed:
            reject(EffectCode.MISSING)
        # Missing terminals already withhold release. Unsubmitted facts remain
        # unauthenticated history until terminal accounting is complete.
        if observed - possible:
            reject(EffectCode.UNSUPPORTED)

    def validate_provenance(self) -> None:
        _ProvenanceValidator(self).validate_provenance()

    def request_accounting(self) -> None:
        receipt = self.result.requests
        if (
            not isinstance(receipt.scope, InvocationRequestScope)
            or receipt.scope.invocation != self.result.record.invocation
        ):
            reject(EffectCode.FOREIGN_OWNER)
        self._requests(receipt)
        bound = self.admitted.execution.context.bound_context
        if bound is not None:
            binding = bound.receipt.requests
            if not isinstance(binding.scope, BindingRequestScope) or binding.scope.binding != bound.receipt.binding:
                reject(EffectCode.FOREIGN_OWNER)
            self._requests(binding)

    def _association_targets(self, association: RequestAssociation) -> frozenset[DatumId]:
        if isinstance(association, SemanticAssociation):
            terminal = next((item for item in self.result.record.terminals if item.attempt == association.task), None)
            return frozenset() if terminal is None else self.affected(terminal.activation)
        bound = self.admitted.execution.context.bound_context
        if bound is None:
            return frozenset()
        source = next((item for item in bound.receipt.sources if item.identity == association.declaration), None)
        return frozenset() if source is None else frozenset({source.declaration.target})

    def _requests(self, receipt: RequestReceipt) -> None:
        requests = {item.request: item for item in receipt.dispatches}
        unlocalized = receipt.dispatched_count != len(receipt.dispatches) or len(requests) != len(receipt.dispatches)
        unlocalized |= any(
            item.request not in requests for item in (*receipt.terminals, *receipt.settlements, *receipt.defects)
        )
        if receipt.budget_limit is not None and receipt.dispatched_count > receipt.budget_limit:
            unlocalized = True
        previous: dict[RequestAssociation, RequestReservation] = {}
        for dispatch in receipt.dispatches:
            targets = [self._association_targets(item) for item in dispatch.associations]
            if not all(targets) or dispatch.request.scope != receipt.scope:
                unlocalized = True
                continue
            affected = set().union(*targets)
            terminals = [item for item in receipt.terminals if item.request == dispatch.request]
            settlements = [item for item in receipt.settlements if item.request == dispatch.request]
            uncertain = len(terminals) != 1 or len(settlements) != 1
            bad = uncertain
            bad |= dispatch.policy not in receipt.policies
            bad |= any(
                not any(
                    binding.association == association and dispatch.policy in binding.policies
                    for binding in receipt.bindings
                )
                for association in dispatch.associations
            )
            uncertain |= any(item.request == dispatch.request for item in receipt.defects)
            uncertain |= dispatch.request in receipt.local_in_flight or dispatch.request in receipt.remote_outstanding
            if settlements:
                uncertain |= (
                    not isinstance(settlements[0].usage, ExactUsage) and dispatch.request not in self.permissive_usage
                ) or settlements[0].remote_stopped is not True
            if terminals:
                terminal = terminals[0]
                bad |= terminal.category in {"lost", "inconsistent", "cancelled"}
                uncertain |= terminal.category in {"lost", "inconsistent"}
                if terminal.category == "success":
                    associations = [item.association for item in terminal.results]
                    bad |= (
                        len(associations) != len(set(associations)) or frozenset(associations) != dispatch.associations
                    )
                    for summary in terminal.results:
                        if summary.category != "success" and not self._recovered(
                            dispatch, receipt, summary.association
                        ):
                            for target in self._association_targets(summary.association):
                                self.withholding[target].add("request_accounting")
                elif terminal.category == "failure":
                    for association in dispatch.associations:
                        if not self._recovered(dispatch, receipt, association):
                            for target in self._association_targets(association):
                                self.withholding[target].add("request_accounting")
            if dispatch.child is None and dispatch.purpose in {"retry", "correction", "failover"}:
                for association in dispatch.associations:
                    predecessor = previous.get(association)
                    prior = next(
                        (
                            item
                            for item in receipt.terminals
                            if predecessor is not None and item.request == predecessor.request
                        ),
                        None,
                    )
                    if (
                        predecessor is None
                        or prior is None
                        or predecessor.policy != dispatch.policy
                        or not self._followup(dispatch, prior, association)
                    ):
                        bad = True
            previous.update((association, dispatch) for association in dispatch.associations)
            if uncertain:
                self.uncertain.update(affected)
            if bad or uncertain:
                for target in affected:
                    self.withholding[target].add("request_accounting")
        if unlocalized:
            for codes in self.withholding.values():
                codes.add("inconsistent_attribution")

    @staticmethod
    def _followup(dispatch: RequestReservation, prior: RequestTerminalFact, association: RequestAssociation) -> bool:
        summary = next((item for item in prior.results if item.association == association), None)
        failure = prior.failure if summary is None else summary.failure
        policy = dispatch.policy
        if failure is None:
            return False
        if dispatch.purpose == "retry":
            return (
                failure == "rejected_before_acceptance"
                and policy.replay in {"before_acceptance", "idempotent"}
                or failure in {"retryable", "transport_unknown"}
                and policy.replay == "idempotent"
            )
        if dispatch.purpose == "correction":
            return failure == "malformed_response" and policy.replay == "idempotent"
        if dispatch.purpose == "failover":
            return failure not in {"malformed_response", "retryable", "transport_unknown"} and (
                policy.replay == "idempotent"
                or failure == "rejected_before_acceptance"
                and policy.replay == "before_acceptance"
            )
        return False

    def _recovered(
        self, dispatch: RequestReservation, receipt: RequestReceipt, association: RequestAssociation
    ) -> bool:
        if dispatch.child is not None:
            return dispatch.request in self.recovered_children
        later = False
        for successor in receipt.dispatches:
            if successor.request == dispatch.request:
                later = True
                continue
            if later and association in successor.associations:
                terminal = next((item for item in receipt.terminals if item.request == successor.request), None)
                if terminal is not None and terminal.category == "success":
                    return any(
                        item.association == association and item.category == "success" for item in terminal.results
                    )
        return False

    def cleanup_accounting(self) -> bool:
        registered = self._registered_handle_resources()
        if registered is not None:
            registered = self._registered_adaptive_resources(registered)
        if registered is None:
            for codes in self.withholding.values():
                codes.add("inconsistent_attribution")
        self._cleanup_ledger(self.result.cleanup, self.result.cleanup_associations, registered=registered)
        bound = self.admitted.execution.context.bound_context
        if bound is not None:
            self._cleanup_ledger(bound.receipt.cleanup, bound.receipt.cleanup_associations, binding=True)
        return registered is not None

    def _registered_handle_resources(self) -> dict[ResourceId, ResourceOwner] | None:
        """Join captured handle declarations to the independently admitted domain."""
        execution = self.admitted.execution
        rows = getattr(self.result, "implementation_resources", None)
        if not isinstance(rows, tuple):
            reject(EffectCode.INVALID_TYPE)
        if len(rows) > self.prepared.limits.max_capabilities:
            reject(EffectCode.LIMIT_EXCEEDED)
        for row in rows:
            require_instance(row, ImplementationResourceRegistration)
            row.__post_init__()
            lease = getattr(row, "_lease", None)
            if lease is not None and (
                not isinstance(lease, ResourceLease)
                or getattr(lease, "resource", None) != row.resource
                or getattr(lease, "owner", None) != row.owner
                or getattr(lease, "safe_detachment", None) != row.safe_detachment
                or getattr(lease, "handle", None) is None
            ):
                return None
        expected = {
            (item.implementation, item.capability.operation, item.configuration): (item.capability, policy)
            for policy, item in execution._service_implementations()
            if item.capability.resource_lifetime != "stateless"
        }
        observed = {}
        for row in rows:
            retained = (row.implementation, row.operation, row.configuration)
            key = next((key for key in expected if _matches_admitted_key(key, retained)), None)
            if key is None or key in observed:
                return None
            observed[key] = row
        if observed.keys() != expected.keys():
            return None
        resources: dict[ResourceId, ResourceOwner] = {}
        detachments = {}
        for key, row in observed.items():
            capability, policy = expected[key]
            if row.invocation != self.result.record.invocation:
                return None
            owner = "caller" if capability.resource_lifetime == "caller_owned" else "sdk"
            if row.owner != owner or row.safe_detachment != policy.safe_detachment:
                return None
            if row.resource in resources and (
                resources[row.resource] != owner or detachments[row.resource] != row.safe_detachment
            ):
                return None
            resources[row.resource] = owner
            detachments[row.resource] = row.safe_detachment
        return resources

    def _registered_adaptive_resources(
        self, registered: dict[ResourceId, ResourceOwner]
    ) -> dict[ResourceId, ResourceOwner] | None:
        """Join each admitted adaptive capability to its actual transfer outcome."""
        rows = getattr(self.result, "adaptive_acquisitions", None)
        if not isinstance(rows, tuple):
            reject(EffectCode.INVALID_TYPE)
        if len(rows) > self.prepared.limits.max_capabilities:
            reject(EffectCode.LIMIT_EXCEEDED)
        for row in rows:
            require_instance(row, AdaptiveResourceAcquisition)
            row.__post_init__()
        expected = tuple(
            cap for cap in self.admitted.execution.context.context_capabilities if "adaptive_retrieval" in cap.uses
        )
        seen: set[int] = set()
        detachments = {row.resource: row.safe_detachment for row in self.result.implementation_resources}
        for row in rows:
            index = next((i for i, cap in enumerate(expected) if _matches_admitted_key(cap, row.capability)), None)
            if index is None or index in seen:
                return None
            seen.add(index)
            capability = expected[index]
            if (
                row.invocation != self.result.record.invocation
                or row.owner != capability.resource_owner
                or row.safe_detachment != capability.safe_detachment
            ):
                return None
            if row.resource is None:
                if row.owner != "sdk":
                    return None
                nodes = {
                    item.node
                    for item in self.admitted.execution.context.adaptive_retrievals
                    if item.source == capability.source and item.materialization.item_type == capability.artifact_type
                }
                if any(
                    isinstance(association, SemanticAssociation)
                    and (entry := self.facts.entries.get(association.task.activation)) is not None
                    and entry.template in nodes
                    for dispatch in self.result.requests.dispatches
                    for association in dispatch.associations
                ):
                    return None
                continue
            if row.resource in registered and (
                registered[row.resource] != row.owner or detachments[row.resource] != row.safe_detachment
            ):
                return None
            registered[row.resource] = row.owner
            detachments[row.resource] = row.safe_detachment
        return registered if len(seen) == len(expected) else None

    def _retained_request_resource(self, resource: ResourceId) -> bool:
        """Authenticate forbidden-detachment retention against a live Lost request."""
        leases = getattr(self.result, "_retained_resources", ())
        if not isinstance(leases, tuple) or len(leases) > self.prepared.limits.max_capabilities:
            return False
        matching = [
            lease
            for lease in leases
            if isinstance(lease, ResourceLease) and getattr(lease, "resource", None) == resource
        ]
        if (
            len(matching) != 1
            or getattr(matching[0], "owner", None) != "sdk"
            or getattr(matching[0], "safe_detachment", None) != "forbidden"
            or getattr(matching[0], "handle", None) is None
        ):
            return False
        for dispatch in self.result.requests.dispatches:
            if dispatch.request not in self.result.requests.remote_outstanding or dispatch.child is None:
                continue
            terminals = [fact for fact in self.result.requests.terminals if fact.request == dispatch.request]
            if len(terminals) != 1 or terminals[0].category != "lost":
                continue
            if any(
                fact.request == dispatch.request and fact.remote_stopped for fact in self.result.requests.settlements
            ):
                continue
            leaf = dispatch.child.leaf
            if leaf is None or leaf.activation not in self.facts.entries:
                continue
            policy = next(
                item
                for item in self.admitted.execution.policies
                if item.node == self.facts.entries[leaf.activation].template
            )
            if policy.safe_detachment != "forbidden":
                continue
            for implementation in policy.implementations:
                if any(
                    row.resource == resource
                    and isinstance(issued := getattr(row, "_lease", None), ResourceLease)
                    and issued.handle is matching[0].handle
                    and row.owner == "sdk"
                    and row.safe_detachment == "forbidden"
                    and _matches_admitted_key(
                        (
                            implementation.implementation,
                            implementation.capability.operation,
                            implementation.configuration,
                        ),
                        (row.implementation, row.operation, row.configuration),
                    )
                    for row in self.result.implementation_resources
                ):
                    return True
        return False

    def _cleanup_ledger(
        self,
        cleanup: tuple[CleanupFact, ...],
        cleanup_associations: tuple[CleanupAssociation, ...],
        *,
        binding: bool = False,
        registered: dict[ResourceId, ResourceOwner] | None = None,
    ) -> None:
        require_instance(cleanup, tuple)
        require_instance(cleanup_associations, tuple)
        # Handle and adaptive source inventories have separate admitted bounds.
        # Binding acquisition completeness remains owned by its receipt boundary.
        limit = self.prepared.limits.max_capabilities + len(self.admitted.execution.context.context_capabilities)
        if binding:
            bound = self.admitted.execution.context.bound_context
            assert bound is not None
            limit = max(limit, len(bound.receipt.sources))
        if max(len(cleanup), len(cleanup_associations)) > limit:
            if binding:
                for codes in self.withholding.values():
                    codes.add("inconsistent_attribution")
                return
            reject(EffectCode.LIMIT_EXCEEDED)
        for fact in cleanup:
            require_instance(fact, CleanupFact)
            fact.__post_init__()
        for association in cleanup_associations:
            require_instance(association, CleanupAssociation)
            require_instance(association.resource, ResourceId)
            require_instance(association.targets, frozenset)
            if len(association.targets) > len(self.prepared.data.targets):
                reject(EffectCode.LIMIT_EXCEEDED)
        facts = {item.resource: item for item in cleanup}
        associations = {item.resource: item for item in cleanup_associations}
        if (
            len(facts) != len(cleanup)
            or len(associations) != len(cleanup_associations)
            or facts.keys() != associations.keys()
            or registered is not None
            and (
                facts.keys() != registered.keys()
                or any(facts[resource].owner != owner for resource, owner in registered.items() if resource in facts)
            )
        ):
            for codes in self.withholding.values():
                codes.add("inconsistent_attribution")
            return
        for resource, fact in facts.items():
            association = associations[resource]
            if (
                not association.targets
                and association.purpose != "transport_only"
                or not association.targets <= self.prepared.data.targets
                or association.purpose not in {"verification", "accounting", "transport_only"}
                or binding
                and association.purpose != "accounting"
                or fact.disposition == "left_open"
                and fact.owner != "caller"
                and not (
                    not binding
                    and registered is not None
                    and association.purpose == "accounting"
                    and self._retained_request_resource(resource)
                )
            ):
                for codes in self.withholding.values():
                    codes.add("inconsistent_attribution")
                continue
            if fact.owner == "caller" and fact.disposition != "left_open":
                for codes in self.withholding.values():
                    codes.add("inconsistent_attribution")
            elif fact.disposition in {"close_failed", "close_unknown"} and association.purpose != "transport_only":
                code: WithholdingCode = (
                    "cleanup_verification" if association.purpose == "verification" else "cleanup_accounting"
                )
                for target in association.targets:
                    self.withholding[target].add(code)

    def _path(self, activation: ActivationKey) -> tuple[NodeId, ...]:
        path = _lexical_path(activation, self.facts.entries, self.nodes)
        if path is None:
            reject(EffectCode.MISSING)
        return path

    def select_candidate(self, target: DatumId) -> None:
        outputs = tuple(
            item
            for item in self.result.final_outputs
            if item.target == target and item.port in self.admitted._subject_outputs(item.outcome)
        )
        self.outputs[target] = outputs
        candidates = {item.candidate for item in outputs}
        outcomes = {item.outcome for item in outputs}
        if len(candidates) > 1 or len(outcomes) > 1:
            reject(EffectCode.CONTRADICTORY)
        if not candidates:
            self.withholding[target].add("missing_candidate")
            return
        candidate = next(iter(candidates))
        self.candidates[target] = candidate
        if candidate not in self.current.candidates or candidate.artifact not in self.result.record.artifacts:
            self.withholding[target].add("missing_candidate")
            self.uncertain.add(target)

    def assess(self, target: DatumId, verified: tuple[VerifiedEvidence, ...]) -> None:
        candidate = self.candidates.get(target)
        if candidate is None:
            return
        outcome = self.outputs[target][0].outcome
        requirements = tuple(item for item in self.prepared.workflow.protection_requirements if item.outcome == outcome)
        if outcome not in self.prepared.configuration.required_protection_outcomes or not requirements:
            self.withholding[target].add("missing_assessment")
            return
        for requirement in requirements:
            endpoints = {
                port
                for port in (requirement.subject_port, *requirement.consumed_ports)
                if isinstance(port, MapItemPort)
            }
            if endpoints:
                self.assess_map_items(target, requirement, next(iter(endpoints)), verified)
                continue
            matching = [
                item
                for item in verified
                if item.subject == candidate
                and any(
                    projection.interface_outcome == outcome
                    and projection.node == item.node
                    and projection.outcome == item.outcome
                    and projection.promise.name == item.promise.name
                    and projection.path == self._path(item.activation)
                    and projection.matches(requirement)
                    for projection in self.admitted._projections
                )
            ]
            successful = [
                item
                for item in matching
                if evidence_validity(evidence=item, current=self.current) == "current"
                and item.finding.status == "satisfied"
                and item.coverage >= requirement.coverage
            ]
            if successful:
                self.supporting[target].update(successful)
            else:
                self.withholding[target].update(self._assessment_codes(matching, requirement))

    def assess_map_items(
        self,
        target: DatumId,
        requirement: ProtectionRequirement,
        endpoint: MapItemPort,
        verified: tuple[VerifiedEvidence, ...],
    ) -> None:
        """Check each actual member against the exact candidate's expansion ancestry."""
        candidate = self.candidates[target]
        outputs = [
            item
            for item in self.outputs[target]
            if item.candidate == candidate
            and item.outcome == requirement.outcome
            and (requirement.candidate_port is None or item.port == requirement.candidate_port)
        ]
        if len(outputs) != 1:
            self.withholding[target].add("missing_candidate")
            return
        ancestry: set[ProvenanceKey] = set()
        pending = [outputs[0].producer]
        while pending:
            key = pending.pop()
            if key in ancestry:
                continue
            ancestry.add(key)
            fact = self.facts.provenance.get(key)
            if fact is not None:
                pending.extend(fact.parents)
        expanders = [
            key
            for key, entry in self.facts.entries.items()
            if self.facts.owners.get(key) == target
            and entry.template == endpoint.expander
            and entry.status == "success"
            and entry.outcome == endpoint.expansion_outcome
            and self._path(key) == endpoint.path
        ]
        for expander in expanders:
            membership = OperationOutputKey(activation=expander, target=target, port=endpoint.membership_port)
            if membership not in ancestry:
                self.withholding[target].add("missing_assessment")
                continue
            successful_terminals = {
                terminal.activation for terminal in self.result.record.terminals if terminal.category == "success"
            }
            members = [
                key
                for key, entry in self.facts.entries.items()
                if key.parent == expander
                and entry.template == endpoint.member
                and entry.status == "success"
                and key in successful_terminals
            ]
            for member in members:
                matching = []
                for item in verified:
                    if not any(
                        projection.node == item.node
                        and projection.outcome == item.outcome
                        and projection.promise.name == item.promise.name
                        and projection.path == self._path(item.activation)
                        and projection.matches(requirement)
                        for projection in self.admitted._projections
                    ):
                        continue
                    if isinstance(requirement.subject_port, MapItemPort):
                        if not isinstance(item.subject, MapItemSubjectRef) or item.subject.producer.member != member:
                            continue
                    elif item.subject != candidate:
                        continue
                    if endpoint in requirement.consumed_ports:
                        fact = next(
                            fact
                            for fact in self.result.assessments
                            if (fact.activation, fact.node, fact.promise)
                            == (item.activation, item.node, item.promise.name)
                        )
                        consumed = [self.facts.map_item(fact, port, endpoint) for port, _ in item.consumed_by_port]
                        if not any(value is not None and value.producer.member == member for value in consumed):
                            continue
                    matching.append(item)
                successful = [
                    item
                    for item in matching
                    if evidence_validity(evidence=item, current=self.current) == "current"
                    and item.finding.status == "satisfied"
                    and item.coverage >= requirement.coverage
                ]
                if successful:
                    self.supporting[target].update(successful)
                else:
                    self.withholding[target].update(self._assessment_codes(matching, requirement))

    def _assessment_codes(
        self, matching: list[VerifiedEvidence], requirement: ProtectionRequirement
    ) -> set[WithholdingCode]:
        if not matching:
            return {"missing_assessment"}
        codes: set[WithholdingCode] = set()
        for item in matching:
            validity = evidence_validity(evidence=item, current=self.current)
            if validity == "stale":
                codes.add("stale_evidence")
            elif validity == "unknown" or item.finding.status == "unknown":
                codes.add("assessment_unknown")
            elif not item.coverage >= requirement.coverage:
                codes.add("incomplete_coverage")
            elif item.finding.status == "unsatisfied":
                codes.add("assessment_unsatisfied")
        return codes

    def propagate(self) -> None:
        withheld = {target for target, codes in self.withholding.items() if codes}
        for _ in range(self.admitted.limits.max_fixed_point_steps):
            before = set(withheld)
            for dependency in self.prepared.data.dependencies:
                if dependency.prerequisite in withheld:
                    withheld.add(dependency.dependent)
            for group in self.prepared.data.atomic:
                if group & withheld:
                    withheld.update(group)
            if withheld == before or withheld == set(self.order):
                break
        else:
            reject(EffectCode.CONTRADICTORY)
        for dependency in self.prepared.data.dependencies:
            if dependency.prerequisite in withheld:
                self.withholding[dependency.dependent].add("dependency")
        for group in self.prepared.data.atomic:
            if len(group) > 1 and group & withheld:
                for target in group:
                    self.withholding[target].add("atomic_group")

    def decisions(self, target: DatumId) -> frozenset[DecisionRef]:
        required = {
            reference
            for item in self.supporting[target]
            for reference in item.reference.consumed
            if isinstance(reference, DecisionRef)
        }
        pending = [item.producer for item in self.outputs[target]]
        visited: set[ProvenanceKey] = set()
        while pending:
            key = pending.pop()
            if key in visited:
                continue
            visited.add(key)
            fact = self.facts.provenance[key]
            if fact.decision:
                required.add(DecisionRef(artifact=fact.artifact))
            pending.extend(fact.parents)
        if len(required) > self.admitted.limits.max_required_decisions:
            reject(EffectCode.LIMIT_EXCEEDED)
        if any(item.artifact not in self.result.record.artifacts for item in required):
            reject(EffectCode.MISSING)
        return frozenset(required)

    def finish(self, verified: tuple[VerifiedEvidence, ...]) -> QualificationResult:
        targets: list[TargetQualification] = []
        qualified: list[QualifiedOutput] = []
        statuses: list[TargetStatus] = []
        execution_only = self.prepared.configuration.purpose == "execution_only"
        for target in self.order:
            candidate = self.candidates.get(target)
            codes = frozenset(self.withholding[target])
            evidence = frozenset(item.reference for item in self.supporting[target])
            decisions = self.decisions(target) if not codes else frozenset()
            targets.append(
                TargetQualification(
                    target=target,
                    candidate=candidate,
                    verified=evidence,
                    required_decisions=decisions,
                    withholding=codes,
                )
            )
            if candidate is not None and not codes:
                qualified.append(
                    QualifiedOutput(target=target, candidate=candidate, evidence=evidence, required_decisions=decisions)
                )
            if execution_only:
                status = "not_assessed"
                available = any(
                    item.target == target and item.candidate.artifact in self.result.record.artifacts
                    for item in self.result.final_outputs
                )
            else:
                available = (
                    candidate is not None
                    and candidate in self.current.candidates
                    and candidate.artifact in self.result.record.artifacts
                )
                if not codes:
                    status = "met"
                elif target in self.uncertain or codes & {"assessment_unknown", "inconsistent_attribution"}:
                    status = "unknown"
                else:
                    status = "unmet"
            statuses.append(
                TargetStatus(
                    target=target,
                    completion="closed" if self.closed[target] else "pending",
                    qualification=status,
                    artifact_available=available,
                    protection_available=not codes and not execution_only,
                )
            )
        required_decisions = frozenset(item for target in qualified for item in target.required_decisions)
        if len(required_decisions) > self.admitted.limits.max_required_decisions:
            reject(EffectCode.LIMIT_EXCEEDED)
        return QualificationResult(
            _key=_QUALIFICATION_KEY,
            record=replace(
                self.result.record, evidence=tuple(item.reference for item in verified), statuses=tuple(statuses)
            ),
            verified=verified,
            targets=tuple(targets),
            qualified=tuple(qualified),
            required_decisions=required_decisions,
            _admitted=self.admitted,
            _result=self.result,
            _current=self.current,
        )
