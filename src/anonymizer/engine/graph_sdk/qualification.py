# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure reconciliation and release qualification of one executed invocation."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, replace
from typing import Literal, get_args

from anonymizer.engine.graph_sdk._activation_values import ActivationEntry
from anonymizer.engine.graph_sdk._effect_values import EffectCode, PrivateValue, reject, require_instance, require_owner
from anonymizer.engine.graph_sdk._execution_topology import _lexical_path
from anonymizer.engine.graph_sdk._execution_values import (
    DelegatedRequestContract,
    ImplementationResourceRegistration,
    OperationExecutionPolicy,
)
from anonymizer.engine.graph_sdk._graph_values import ActivationKey, DatumId
from anonymizer.engine.graph_sdk._qualification_provenance import _ProvenanceValidator
from anonymizer.engine.graph_sdk._workflow_values import MapItemPort, NodeId, ProtectionRequirement, SubgraphNode
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
from anonymizer.engine.graph_sdk.records import CandidateRef, CanonicalRecord, EvidenceRef, TargetStatus
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


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class TargetQualification(PrivateValue):
    target: DatumId
    candidate: CandidateRef | None
    verified: frozenset[EvidenceRef]
    withholding: frozenset[WithholdingCode]

    def __post_init__(self) -> None:
        require_instance(self.target, DatumId)
        if self.candidate is not None:
            require_instance(self.candidate, CandidateRef)
        require_instance(self.verified, frozenset)
        require_instance(self.withholding, frozenset)
        if any(not isinstance(item, EvidenceRef) for item in self.verified) or any(
            not isinstance(item, str) for item in self.withholding
        ):
            reject(EffectCode.INVALID_TYPE)
        if not self.withholding <= _CODES:
            reject(EffectCode.INVALID_VALUE)
        if self.candidate is not None and self.candidate.target != self.target:
            reject(EffectCode.FOREIGN_OWNER)
        if self.candidate is not None and any(
            item.artifact.invocation != self.candidate.artifact.invocation for item in self.verified
        ):
            reject(EffectCode.FOREIGN_OWNER)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class QualifiedOutput(PrivateValue):
    target: DatumId
    candidate: CandidateRef
    evidence: frozenset[EvidenceRef]

    def __post_init__(self) -> None:
        require_instance(self.candidate, CandidateRef)
        TargetQualification(
            target=self.target,
            candidate=self.candidate,
            verified=self.evidence,
            withholding=frozenset(),
        )


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class QualificationResult(PrivateValue):
    _key: InitVar[object]
    record: CanonicalRecord
    verified: tuple[VerifiedEvidence, ...]
    targets: tuple[TargetQualification, ...]
    qualified: tuple[QualifiedOutput, ...]
    _admitted: AdmittedQualification
    _result: ExecutionResult
    _current: EvidenceRevisionView

    def __post_init__(self, _key: object) -> None:
        require_owner(_key, _QUALIFICATION_KEY)


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
    facts = current._facts  # built once by evidence_revision_view for exactly this pair
    assert facts.admitted is admitted and facts.result is result
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


@dataclass(slots=True)
class _ChildLedger:
    """Expected physical child slots and what the retained chains have joined so far."""

    expected: set[ActivationKey] = field(default_factory=set)
    readiness_owners: dict[ActivationKey, DelegatedRequestContract] = field(default_factory=dict)
    physical_owners: dict[ActivationKey, ActivationKey] = field(default_factory=dict)
    probes: set[ActivationKey] = field(default_factory=set)
    used: set[ActivationKey] = field(default_factory=set)
    recovered: set[ActivationKey] = field(default_factory=set)
    known_evidence: set[ChildEvidence] = field(default_factory=set)
    processed: set[PhysicalRequestId] = field(default_factory=set)
    used_groups: set[tuple[ActivationKey, tuple[int, ...]]] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class _ChainRecipe:
    """One retained chain joined to its owner's admitted contract."""

    chain: RequestChain
    owner: ActivationKey
    contract: DelegatedRequestContract
    slots: tuple[int, ...]
    offset: int
    history: list[RequestReservation]


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
        self.policies = {policy.node: policy for policy in admitted.execution.policies}
        self.receipt = result.requests
        self.attempts = {item.attempt: item for item in result.record.terminals if item.attempt is not None}
        self.offsets = {item.target: item.occurrence_offset for item in self.prepared.target_occurrences}
        # Retained entries by occurrence, in retained order (map and loop members share an occurrence).
        self.occurrences: dict[int, list[ActivationEntry]] = {}
        for entry in facts.entries.values():
            self.occurrences.setdefault(entry.activation.occurrence, []).append(entry)

    def incomplete(self, target: DatumId) -> None:
        self.closed[target] = False
        self.withholding[target].add("incomplete_membership")

    def affected(self, activation: ActivationKey) -> frozenset[DatumId]:
        return frozenset(self.order) if activation in self.facts.shared else frozenset({self.facts.owners[activation]})

    def offset(self, activation: ActivationKey) -> int:
        if activation in self.facts.shared:
            return self.prepared.invocation_offset
        return self.offsets[self.facts.owners[activation]]

    def require_occurrence(self, occurrence: int) -> ActivationEntry:
        """The first retained entry at an occurrence the admitted recipe requires."""
        entries = self.occurrences.get(occurrence)
        if not entries:
            reject(EffectCode.MISSING)
        return entries[0]

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
            structural = isinstance(self.nodes[entry.template], SubgraphNode)
            if terminal.structural != structural or terminal.category != entry.status:
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
        if not self._child_receipt_in_bounds(receipt):
            return set()
        chains = {chain.chain: chain for chain in receipt.chains}
        if len(chains) != len(receipt.chains) or len(set(receipt.child_evidence)) != len(receipt.child_evidence):
            reject(EffectCode.DUPLICATE)
        ledger = self._expected_children(_ChildLedger())
        for chain in receipt.chains:
            self._child_chain(chain, ledger)
        self._generation_chain_counts(ledger)
        omitted = self._unused_children(ledger.expected - ledger.used, ledger.physical_owners)
        omitted |= self._unselected_generations(ledger.physical_owners)
        self._close_child_inventory(ledger, omitted)
        return ledger.recovered | omitted

    def _child_receipt_in_bounds(self, receipt: RequestReceipt) -> bool:
        """False when no child request can exist; otherwise the receipt's containers fit the admitted bound."""
        if (
            not any(policy.delegated is not None for policy in self.policies.values())
            and not receipt.chains
            and not receipt.child_evidence
            and not any(dispatch.child is not None for dispatch in receipt.dispatches)
        ):
            return False
        bound = self.prepared.declared_request_upper_bound
        if bound is None:
            # Without a declared bound, only an empty invocation receipt for this run is consistent.
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
            return False
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
        limits = self.prepared.limits
        if (
            len(receipt.defects) > bound
            or len(receipt.bindings) > self.prepared.total_activation_slots
            or len(receipt.policies) > limits.max_capabilities
            or any(len(binding.policies) > limits.max_capabilities for binding in receipt.bindings)
        ):
            reject(EffectCode.LIMIT_EXCEEDED)
        return True

    def _expected_children(self, ledger: _ChildLedger) -> _ChildLedger:
        """Every physical child slot the admitted recipe reserves for a delegated owner present in this run."""
        for owner in self.facts.entries.values():
            policy = self.policies.get(owner.template)
            if policy is None or policy.delegated is None:
                continue
            contract = policy.delegated
            offset = self.offset(owner.activation)
            descriptor = contract.readiness
            if descriptor is not None:
                readiness_offset = self.prepared.invocation_offset if self.facts.shared else offset
                semantic = self.require_occurrence(readiness_offset + descriptor.owner_slot).activation
                physical = self.require_occurrence(readiness_offset + descriptor.request_slot).activation
                probe = self.require_occurrence(readiness_offset + descriptor.probe_slot).activation
                ledger.readiness_owners[semantic] = contract
                ledger.probes.add(probe)
                ledger.physical_owners[physical] = semantic
                ledger.expected.add(physical)
            for slot in dict.fromkeys(contract.child_slots):
                for entry in self.occurrences.get(offset + slot, ()):
                    if self.facts.owners.get(entry.activation) == self.facts.owners[owner.activation]:
                        ledger.physical_owners[entry.activation] = owner.activation
                        ledger.expected.add(entry.activation)
        return ledger

    def _child_chain(self, chain: RequestChain, ledger: _ChildLedger) -> None:
        """Join one retained chain to its owner's admitted route recipe, then each of its dispatches."""
        if (
            chain.chain.scope != self.receipt.scope
            or len(chain.associations) != 1
            or chain.role not in {"generation", "readiness"}
        ):
            reject(EffectCode.UNSUPPORTED)
        association = next(iter(chain.associations))
        if not isinstance(association, SemanticAssociation) or association.task not in self.attempts:
            reject(EffectCode.FOREIGN_OWNER)
        owner = self.attempts[association.task].activation
        policy = self.policies[self.facts.entries[owner].template]
        contract, slots = self._chain_recipe(chain, owner, policy, ledger)
        history = [
            item for item in self.receipt.dispatches if item.child is not None and item.child.chain == chain.chain
        ]
        if not history and (self.facts.entries[owner].status != "success" or contract.selection is not None):
            return
        if not history or len(history) > chain.policy.max_attempts:
            reject(EffectCode.LIMIT_EXCEEDED)
        offset = self.offset(owner)
        if chain.role == "generation" and contract.generation_groups:
            first = history[0].child
            if first is None or first.leaf is None:
                reject(EffectCode.MISSING)
            groups = [group for group in contract.generations if first.leaf.activation.occurrence - offset in group]
            if len(groups) != 1 or (owner, groups[0]) in ledger.used_groups:
                reject(EffectCode.CONTRADICTORY)
            slots = groups[0]
            ledger.used_groups.add((owner, slots))
        recipe = _ChainRecipe(chain, owner, contract, slots, offset, history)
        prior: RequestReservation | None = None
        for dispatch in history:
            self._child_dispatch(recipe, dispatch, prior, ledger)
            prior = dispatch
        terminal = next(item for item in self.receipt.terminals if item.request == history[-1].request)
        if terminal.category == "success" and any(
            item.association == association and item.category == "success" for item in terminal.results
        ):
            for dispatch in history[:-1]:
                child = dispatch.child
                assert child is not None and child.leaf is not None
                fact = next(item for item in self.receipt.terminals if item.request == dispatch.request)
                if fact.category == "failure":
                    self.recovered_children.add(dispatch.request)
                    ledger.recovered.add(child.leaf.activation)

    def _chain_recipe(
        self, chain: RequestChain, owner: ActivationKey, policy: OperationExecutionPolicy, ledger: _ChildLedger
    ) -> tuple[DelegatedRequestContract, tuple[int, ...]]:
        """The owner's admitted contract and slots for this chain's role; generation waits on readiness."""
        contract = policy.delegated
        slots = () if contract is None else contract.child_slots
        routes = () if contract is None else contract.routes
        followups = frozenset() if contract is None else contract.followups
        if chain.role == "readiness":
            relation = ledger.readiness_owners.get(owner)
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
                dependency = self.require_occurrence(offset + index)
                if (
                    dependency.status != "success"
                    or dependency.outcome not in outcomes
                    or not any(
                        row.activation == dependency.activation and row.attempt is not None
                        for row in self.attempts.values()
                    )
                ) and any(
                    item.child is not None and item.child.chain == chain.chain for item in self.receipt.dispatches
                ):
                    reject(EffectCode.CONTRADICTORY)
        if (
            contract is None
            or contract.mode != "qualified"
            or policy.request != chain.policy
            or routes != chain.routes
            or followups != chain.followups
        ):
            reject(EffectCode.CONTRADICTORY)
        return contract, slots

    def _child_dispatch(
        self, recipe: _ChainRecipe, dispatch: RequestReservation, prior: RequestReservation | None, ledger: _ChildLedger
    ) -> None:
        """One dispatch: its leaf is an unused owned slot, its terminal and settlement agree, its edge is legal."""
        chain, contract, offset = recipe.chain, recipe.contract, recipe.offset
        child = dispatch.child
        assert child is not None
        leaf = child.leaf
        if (
            leaf is None
            or leaf not in self.attempts
            or leaf.activation not in ledger.expected
            or leaf.activation in ledger.used
            or self.facts.owners.get(leaf.activation) != self.facts.owners.get(recipe.owner)
            or leaf.activation.parent != recipe.owner.parent
            or leaf.activation.occurrence - offset not in recipe.slots
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
            if recipe.owner.parent is None or recipe.owner.parent.occurrence != offset + slot.parent_index:
                reject(EffectCode.FOREIGN_OWNER)
        elif recipe.owner.parent is not None:
            reject(EffectCode.FOREIGN_OWNER)
        if (
            dispatch.request in ledger.processed
            or dispatch.associations != chain.associations
            or dispatch.policy != chain.policy
            or child.role != chain.role
        ):
            reject(EffectCode.CONTRADICTORY)
        facts = [item for item in self.receipt.terminals if item.request == dispatch.request]
        settlements = [item for item in self.receipt.settlements if item.request == dispatch.request]
        if len(facts) != 1 or self.attempts[leaf].category != facts[0].category:
            reject(EffectCode.CONTRADICTORY)
        # Lost work stays remotely outstanding; a confirmed stop or an
        # accepted response without optional settlement clears it without
        # settling usage. Request accounting withholds each.
        outstanding = dispatch.request in self.receipt.remote_outstanding
        unsettled = not settlements and outstanding == (facts[0].category == "lost")
        if len(settlements) != 1 and not unsettled:
            reject(EffectCode.CONTRADICTORY)
        owner = self.facts.entries[recipe.owner]
        policy = self.policies[owner.template]
        for summary in facts[0].results:
            if summary.category == "success" and (
                summary.association not in chain.associations
                or summary.outcome != self.facts.entries[leaf.activation].outcome
                or (owner.status == "success" and summary.outcome != owner.outcome)
                or summary.outcome not in policy.result_outcomes
                or not any(
                    outcome.name == summary.outcome and outcome.category == "success"
                    for outcome in self.nodes[owner.template].operation.outcomes
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
        if not self._child_edge(dispatch, prior, recipe.history, chain, self.receipt):
            reject(EffectCode.CONTRADICTORY)
        for evidence in self.receipt.child_evidence:
            if evidence.chain != chain.chain or evidence.request != dispatch.request:
                continue
            # Parser evidence names a completed malformed response; exhaustion follows every parse attempt.
            if (
                facts[0].category != "failure"
                or facts[0].failure != "malformed_response"
                or not settlements
                or settlements[0].disposition != "completed"
                or any(item.request == dispatch.request for item in self.receipt.defects)
            ):
                reject(EffectCode.CONTRADICTORY)
            if evidence.kind == "generation_exhausted":
                route = next(item for item in chain.routes if item.name == child.route)
                history = recipe.history
                parsed = [
                    item
                    for item in history[: history.index(dispatch) + 1]
                    if item.child is not None
                    and item.child.route == child.route
                    and ChildEvidence(chain=chain.chain, request=item.request, kind="parser_rejected")
                    in ledger.known_evidence
                ]
                if (
                    len(parsed) != route.parse_attempts
                    or ChildEvidence(chain=chain.chain, request=dispatch.request, kind="parser_rejected")
                    not in ledger.known_evidence
                ):
                    reject(EffectCode.CONTRADICTORY)
            ledger.known_evidence.add(evidence)
        if chain.role == "generation" and contract.usage_requirement == "unknown_allowed":
            self.permissive_usage.add(dispatch.request)
        ledger.used.add(leaf.activation)
        ledger.processed.add(dispatch.request)

    def _generation_chain_counts(self, ledger: _ChildLedger) -> None:
        """An entered generation owner has exactly one generation chain per admitted group."""
        for entry in self.facts.entries.values():
            policy = self.policies.get(entry.template)
            if policy is None or policy.delegated is None or not policy.delegated.generation_groups:
                continue
            entered = any(row.activation == entry.activation for row in self.attempts.values())
            own_chains = [
                chain
                for chain in self.receipt.chains
                if chain.role == "generation"
                and any(
                    isinstance(association, SemanticAssociation) and association.task.activation == entry.activation
                    for association in chain.associations
                )
            ]
            if len(own_chains) != (len(policy.delegated.generations) if entered else 0):
                reject(EffectCode.MISSING)

    def _close_child_inventory(self, ledger: _ChildLedger, omitted: set[ActivationKey]) -> None:
        """Undispatched slots are blocked, evidence is all joined, and no other dispatch claims a child slot."""
        unused = ledger.expected - ledger.used - omitted
        # A cancelled or failed owner may leave slots undispatched; its
        # non-success terminal then withholds the target.
        if any(
            self.facts.entries[key].status != "blocked"
            or self.facts.entries[ledger.physical_owners[key]].status == "success"
            or any(row.activation == key for row in self.attempts.values())
            for key in unused
        ):
            reject(EffectCode.CONTRADICTORY)
        if ledger.known_evidence != set(self.receipt.child_evidence) or ledger.used - ledger.expected:
            reject(EffectCode.CONTRADICTORY)
        claimed = ledger.expected | ledger.probes
        for dispatch in self.receipt.dispatches:
            if dispatch.child is not None and dispatch.request not in ledger.processed:
                reject(EffectCode.MISSING)
            for association in dispatch.associations:
                if not isinstance(association, SemanticAssociation):
                    continue
                # Declared physical slots belong exclusively to their child dispatch;
                # no ordinary or shared dispatch may reuse them as semantic owners.
                if association.task.activation in claimed:
                    reject(EffectCode.CONTRADICTORY)
                if association.task in self.attempts:
                    node = self.facts.entries[self.attempts[association.task].activation].template
                    if (
                        self.policies[node].delegated is not None
                        or association.task.activation in ledger.readiness_owners
                    ) and dispatch.request not in ledger.processed:
                        reject(EffectCode.MISSING)

    def _unselected_generations(self, owners: dict[ActivationKey, ActivationKey]) -> set[ActivationKey]:
        accepted: set[ActivationKey] = set()
        terminals = self.facts.terminals
        values = self.facts.values
        for owner in set(owners.values()):
            contract = self.policies[self.facts.entries[owner].template].delegated
            if contract is None or contract.selection is None or self.facts.entries[owner].status != "success":
                continue
            selection = contract.selection
            ports = [port for port in self.result.ports if port.activation == owner and port.port == selection.port]
            if len(ports) != 1 or ports[0].artifact not in values:
                reject(EffectCode.MISSING)
            omitted = selection.omitted(values[ports[0].artifact])
            offset = self.offset(owner)
            for slot in omitted:
                key = next(
                    (key for key, parent in owners.items() if parent == owner and key.occurrence == offset + slot), None
                )
                if key is None:
                    reject(EffectCode.MISSING)
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
            contract = self.policies[self.facts.entries[owner].template].delegated
            assert contract is not None
            offset = self.offsets[self.facts.owners[key]]
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
        """Retained facts belong to this invocation.

        The executor keys facts by (activation, promise) and captures them only for a committed, successful
        occurrence whose outcome declares the production, so duplicates, strays and unsupported facts cannot be
        retained; a success that omits a declared production is a publication failure instead.
        """
        if any(fact.activation.invocation != self.result.record.invocation for fact in self.result.assessments):
            reject(EffectCode.FOREIGN_OWNER)

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
        unlocalized = len(requests) != len(receipt.dispatches)
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
        rows = self.result.implementation_resources
        if len(rows) > self.prepared.limits.max_capabilities:
            reject(EffectCode.LIMIT_EXCEEDED)
        for row in rows:
            require_instance(row, ImplementationResourceRegistration)
            lease = row._lease
            if lease is not None and (
                not isinstance(lease, ResourceLease)
                or lease.resource != row.resource
                or lease.owner != row.owner
                or lease.safe_detachment != row.safe_detachment
                or lease.handle is None
            ):
                return None
        expected = {
            (item.implementation, item.capability.operation, item.configuration): (item.capability, policy)
            for policy, item in execution._service_implementations()
            if item.capability.resource_lifetime != "stateless"
        }
        observed = {}
        for row in rows:
            # Rows carry the very handle objects admission matched to these keys (set equality, U1.5 probe 4).
            key = (row.implementation, row.operation, row.configuration)
            if key not in expected or key in observed:
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

    def _retained_request_resource(self, resource: ResourceId) -> bool:
        """Authenticate forbidden-detachment retention against a live Lost request."""
        leases = self.result._retained_resources
        if not isinstance(leases, tuple) or len(leases) > self.prepared.limits.max_capabilities:
            return False
        matching = [lease for lease in leases if isinstance(lease, ResourceLease) and lease.resource == resource]
        if (
            len(matching) != 1
            or matching[0].owner != "sdk"
            or matching[0].safe_detachment != "forbidden"
            or matching[0].handle is None
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
                    and isinstance(issued := row._lease, ResourceLease)
                    and issued.handle is matching[0].handle
                    and row.owner == "sdk"
                    and row.safe_detachment == "forbidden"
                    and (row.implementation, row.operation, row.configuration)
                    == (
                        implementation.implementation,
                        implementation.capability.operation,
                        implementation.configuration,
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
        # Binding acquisition completeness remains owned by its receipt boundary.
        limit = self.prepared.limits.max_capabilities
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

    def finish(self, verified: tuple[VerifiedEvidence, ...]) -> QualificationResult:
        targets: list[TargetQualification] = []
        qualified: list[QualifiedOutput] = []
        statuses: list[TargetStatus] = []
        execution_only = self.prepared.configuration.purpose == "execution_only"
        for target in self.order:
            candidate = self.candidates.get(target)
            codes = frozenset(self.withholding[target])
            evidence = frozenset(item.reference for item in self.supporting[target])
            targets.append(
                TargetQualification(
                    target=target,
                    candidate=candidate,
                    verified=evidence,
                    withholding=codes,
                )
            )
            if candidate is not None and not codes:
                qualified.append(QualifiedOutput(target=target, candidate=candidate, evidence=evidence))
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
        return QualificationResult(
            _key=_QUALIFICATION_KEY,
            record=replace(
                self.result.record, evidence=tuple(item.reference for item in verified), statuses=tuple(statuses)
            ),
            verified=verified,
            targets=tuple(targets),
            qualified=tuple(qualified),
            _admitted=self.admitted,
            _result=self.result,
            _current=self.current,
        )
