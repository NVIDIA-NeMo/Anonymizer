# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure reconciliation and release qualification of one executed invocation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, get_args

from anonymizer.engine.graph_sdk._effect_values import EffectCode, PrivateValue, reject, require_instance
from anonymizer.engine.graph_sdk._workflow import reachable_workflows
from anonymizer.engine.graph_sdk.evidence import (
    AdmittedQualification,
    AssessmentSubmission,
    EvidenceRevisionView,
    VerifiedEvidence,
    _EvidenceFacts,
    evidence_validity,
    verify_evidence,
)
from anonymizer.engine.graph_sdk.executor import (
    ArtifactProvenanceFact,
    BoundInputKey,
    ExecutionResult,
    FinalOutputFact,
    InitialCollectionKey,
    MapItemKey,
    OperationOutputKey,
    ProvenanceKey,
    RootInputKey,
    _source_activation,
)
from anonymizer.engine.graph_sdk.records import CandidateRef, CanonicalRecord, DecisionRef, EvidenceRef, TargetStatus
from anonymizer.engine.graph_sdk.requests import (
    BindingRequestScope,
    ExactUsage,
    InvocationRequestScope,
    RequestAssociation,
    RequestReceipt,
    RequestReservation,
    RequestTerminalFact,
    SemanticAssociation,
    TextCollectionValue,
)
from anonymizer.engine.graph_sdk.resources import CleanupAssociation, CleanupFact
from anonymizer.graph._values import ActivationKey, DatumId
from anonymizer.graph.activation import ActivationEntry
from anonymizer.graph.workflow import NodeId, NodeOutputRef, ProtectionRequirement, SubgraphNode, WorkflowInputRef

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


def qualify(
    *,
    admitted: AdmittedQualification,
    result: ExecutionResult,
    current: EvidenceRevisionView,
    submissions: tuple[AssessmentSubmission, ...],
) -> QualificationResult:
    """Reconcile execution and release only targets supported by current evidence."""
    require_instance(admitted, AdmittedQualification)
    require_instance(result, ExecutionResult)
    require_instance(current, EvidenceRevisionView)
    require_instance(submissions, tuple)
    if len(submissions) > min(admitted.limits.max_submissions, admitted.limits.max_verified_evidence):
        reject(EffectCode.LIMIT_EXCEEDED)
    if admitted.execution.context.prepared.configuration.purpose != "execution_only" and any(
        not isinstance(item, AssessmentSubmission) for item in submissions
    ):
        reject(EffectCode.INVALID_TYPE)
    if current._admitted is not admitted or current._result is not result:
        reject(EffectCode.FOREIGN_OWNER)
    facts = _EvidenceFacts.from_result(admitted, result)
    qualification = _Qualification(admitted, result, current, facts)
    qualification.reconcile()
    qualification.validate_provenance()
    if admitted.execution.context.prepared.configuration.purpose == "execution_only":
        for codes in qualification.withholding.values():
            codes.add("execution_only")
        return qualification.finish(())
    for target in qualification.order:
        qualification.select_candidate(target)
    verified = verify_evidence(admitted=admitted, result=result, submissions=submissions)
    qualification.request_accounting()
    qualification.cleanup_accounting()
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
        facts: _EvidenceFacts,
    ) -> None:
        self.admitted = admitted
        self.result = result
        self.current = current
        self.facts = facts
        self.prepared = admitted.execution.context.prepared
        self.order = tuple(item.target for item in self.prepared.target_occurrences)
        self.nodes = {node.id: node for body in reachable_workflows(self.prepared.workflow) for node in body.nodes}
        self.entries: dict[ActivationKey, ActivationEntry] = {}
        self.owners: dict[ActivationKey, DatumId] = {}
        self.withholding: dict[DatumId, set[WithholdingCode]] = {target: set() for target in self.order}
        self.closed = {target: True for target in self.order}
        self.uncertain: set[DatumId] = set()
        self.candidates: dict[DatumId, CandidateRef] = {}
        self.supporting: dict[DatumId, set[VerifiedEvidence]] = {target: set() for target in self.order}
        self.outputs: dict[DatumId, tuple[FinalOutputFact, ...]] = {}
        self.provenance: dict[ProvenanceKey, ArtifactProvenanceFact] = {}
        self.input_parents: dict[tuple[DatumId, ActivationKey, str], ProvenanceKey] = {}
        self.passthrough_parents: dict[tuple[DatumId, ActivationKey, str], ProvenanceKey] = {}

    def incomplete(self, target: DatumId) -> None:
        self.closed[target] = False
        self.withholding[target].add("incomplete_membership")

    def reconcile(self) -> None:
        record = self.result.record
        if len(self.result.states) != len(self.order):
            reject(EffectCode.MISSING)
        for target, state in zip(self.order, self.result.states, strict=True):
            if state.invocation != record.invocation or state.workflow is not self.prepared.workflow:
                reject(EffectCode.FOREIGN_OWNER)
            expected = {
                self.prepared.target_occurrences[self.order.index(target)].occurrence_offset + slot.index: slot
                for slot in self.prepared.reservation_recipe
            }
            for entry in state.entries:
                if entry.activation in self.entries:
                    reject(EffectCode.DUPLICATE)
                slot = expected.get(entry.activation.occurrence)
                if slot is None or entry.template != slot.template:
                    reject(EffectCode.FOREIGN_OWNER)
                self.entries[entry.activation] = entry
                self.owners[entry.activation] = target
            if not state.complete:
                self.incomplete(target)
        self._memberships()
        self._terminals()
        for pending in self.result.pending_decisions:
            target = self.owners.get(pending.activation)
            if target is None:
                reject(EffectCode.FOREIGN_OWNER)
            self.incomplete(target)
        if {item.target for item in record.statuses} != set(self.order):
            reject(EffectCode.MISSING)

    def _memberships(self) -> None:
        expected: dict[ActivationKey | None, set[ActivationKey]] = {None: set()}
        closed: dict[ActivationKey | None, bool] = {None: all(state.complete for state in self.result.states)}
        for key, entry in self.entries.items():
            if key.parent is not None and (
                key.parent not in self.entries or self.owners[key.parent] != self.owners[key]
            ):
                reject(EffectCode.FOREIGN_OWNER)
            expected.setdefault(key.parent, set()).add(key)
            if key.parent is not None:
                closed[key.parent] = self.entries[key.parent].status in _TERMINAL
        for state in self.result.states:
            for expansion in state.expansions:
                if expansion.parent not in self.entries:
                    reject(EffectCode.MISSING)
                expected.setdefault(expansion.parent, set())
                closed[expansion.parent] = expansion.status != "pending"
                parent = self.entries[expansion.parent]
                scope = next(
                    scope
                    for scope in self.prepared.workflow.scopes
                    if scope.workflow.workflow == parent.template.workflow
                )
                member_templates = {item.member for item in scope.maps if item.expander == parent.template} | {
                    item.member for item in scope.loops if item.starter == parent.template
                }
                dynamic_members = frozenset(
                    member for member in expected[expansion.parent] if self.entries[member].template in member_templates
                )
                if expansion.members != dynamic_members:
                    self.incomplete(self.owners[expansion.parent])
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
                        self.incomplete(self.owners[expansion.parent])
        observed = {item.parent: item for item in self.result.record.memberships}
        if len(observed) != len(self.result.record.memberships):
            reject(EffectCode.DUPLICATE)
        if set(observed) - set(expected):
            reject(EffectCode.CONTRADICTORY)
        all_members: set[ActivationKey] = set()
        for parent, members in expected.items():
            membership = observed.get(parent)
            affected = set(self.order) if parent is None else {self.owners[parent]}
            if membership is None or membership.members != frozenset(members) or membership.closed != closed[parent]:
                for target in affected:
                    self.incomplete(target)
            if membership is not None:
                if all_members & membership.members:
                    reject(EffectCode.DUPLICATE)
                if any(member not in self.entries for member in membership.members):
                    reject(EffectCode.MISSING)
                all_members.update(membership.members)

    def _terminals(self) -> None:
        terminals = {item.activation: item for item in self.result.record.terminals}
        if len(terminals) != len(self.result.record.terminals):
            reject(EffectCode.DUPLICATE)
        if set(terminals) - set(self.entries):
            reject(EffectCode.MISSING)
        for key, entry in self.entries.items():
            target = self.owners[key]
            terminal = terminals.get(key)
            if terminal is None:
                self.incomplete(target)
                continue
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
                self.withholding[target].add("terminal_failure")

    def validate_provenance(self) -> None:
        if len(self.result.provenance) > self.admitted.limits.max_port_facts:
            reject(EffectCode.LIMIT_EXCEEDED)
        for fact in self.result.provenance:
            if fact.key in self.provenance:
                reject(EffectCode.DUPLICATE)
            if fact.artifact.invocation != self.result.record.invocation or fact.key.target not in self.order:
                reject(EffectCode.FOREIGN_OWNER)
            if fact.artifact not in self.result.record.artifacts:
                reject(EffectCode.MISSING)
            self.provenance[fact.key] = fact
        self._validate_input_parents()
        self._validate_passthrough_parents()
        for fact in self.result.provenance:
            if not fact.parents <= self.provenance.keys():
                reject(EffectCode.MISSING)
            self._producer(fact)
        self._validate_artifact_allocations()
        self._provenance_acyclic()
        outputs = [(item.target, item.port) for item in self.result.final_outputs]
        if len(outputs) != len(set(outputs)):
            reject(EffectCode.DUPLICATE)
        for output in self.result.final_outputs:
            if output.target not in self.order or output.candidate.target != output.target:
                reject(EffectCode.FOREIGN_OWNER)
            producer = self.provenance.get(output.producer)
            if producer is None:
                reject(EffectCode.MISSING)
            if producer.key.target != output.target:
                reject(EffectCode.FOREIGN_OWNER)
            if producer.artifact != output.candidate.artifact:
                reject(EffectCode.CONTRADICTORY)
            bindings = [
                item for item in self.prepared.workflow.workflow.output_bindings if item.destination.port == output.port
            ]
            if len(bindings) != 1:
                reject(EffectCode.MISSING)
            source = bindings[0].source
            if isinstance(source, NodeOutputRef):
                if not isinstance(output.producer, OperationOutputKey):
                    reject(EffectCode.CONTRADICTORY)
                source_entry = self.entries[output.producer.activation]
                if source_entry.template != source.node or output.producer.port != source.port:
                    reject(EffectCode.CONTRADICTORY)
            elif isinstance(source, WorkflowInputRef) and output.producer != RootInputKey(
                target=output.target, port=source.port
            ):
                reject(EffectCode.CONTRADICTORY)
            actual_outcomes = {
                binding.destination.outcome
                for binding in self.prepared.workflow.workflow.outcome_bindings
                for key, entry in self.entries.items()
                if self.owners[key] == output.target
                and entry.template == binding.source.node
                and entry.status == "success"
                and entry.outcome == binding.source.outcome
            }
            if actual_outcomes != {output.outcome}:
                reject(EffectCode.CONTRADICTORY)
            if output.outcome not in {
                item.name
                for item in self.prepared.workflow.workflow.interface.outcomes
                if output.port in item.produced_ports
            }:
                reject(EffectCode.CONTRADICTORY)

    def _validate_artifact_allocations(self) -> None:
        allocated: dict[tuple[object, ...], int] = {}
        owners: dict[int, tuple[object, ...]] = {}
        for fact in self.result.provenance:
            key = fact.key
            if isinstance(key, BoundInputKey):
                lineage = (BoundInputKey, key.binding_artifact.declaration, key.binding_artifact.key)
                version = key.binding_artifact.version
            elif isinstance(key, MapItemKey):
                lineage = (MapItemKey, key.expander, key.target, key.item_key)
                version = key.item_version
            elif isinstance(key, OperationOutputKey):
                node = self.nodes[self.entries[key.activation].template]
                if isinstance(node, SubgraphNode):
                    continue
                dependency = next(item for item in node.operation.output_dependencies if item.output == key.port)
                if dependency.identity_input is not None:
                    continue
                lineage = (OperationOutputKey, key)
                version = 1
            else:
                lineage = (type(key), key)
                version = 1
            if (
                fact.artifact.version != version
                or allocated.setdefault(lineage, fact.artifact.key) != fact.artifact.key
                or owners.setdefault(fact.artifact.key, lineage) != lineage
            ):
                reject(EffectCode.CONTRADICTORY)

    def _validate_input_parents(self) -> None:
        for target, activation, port, parent in self.result._input_parents:
            if activation not in self.entries or target != self.owners[activation] or parent.target != target:
                reject(EffectCode.FOREIGN_OWNER)
            key = (target, activation, port)
            if key in self.input_parents:
                reject(EffectCode.DUPLICATE)
            self.input_parents[key] = parent
        expected = {
            (fact.target, fact.activation, fact.port)
            for fact in self.result.ports
            if fact.activation in self.entries
            and not isinstance(self.nodes[self.entries[fact.activation].template], SubgraphNode)
            and fact.port in {item.name for item in self.nodes[self.entries[fact.activation].template].operation.inputs}
        }
        if expected - self.input_parents.keys():
            reject(EffectCode.MISSING)
        if self.input_parents.keys() - expected:
            reject(EffectCode.CONTRADICTORY)
        for (_, activation, port), parent in self.input_parents.items():
            source = self.provenance.get(parent)
            if source is None:
                reject(EffectCode.MISSING)
            if source.artifact != self.facts.ports[activation, port].artifact:
                reject(EffectCode.CONTRADICTORY)
            if isinstance(parent, BoundInputKey):
                bound = self.admitted.execution.context.bound_context
                if bound is None:
                    reject(EffectCode.MISSING)
                declaration = next(
                    (
                        item.declaration
                        for item in bound.receipt.sources
                        if item.identity == parent.binding_artifact.declaration
                    ),
                    None,
                )
                if declaration is None:
                    reject(EffectCode.FOREIGN_OWNER)
                if declaration.version_selection == "latest" and parent.binding_artifact.version != max(
                    (
                        item.reference.version
                        for item in bound.artifacts
                        if item.reference.declaration == parent.binding_artifact.declaration
                        and item.reference.key == parent.binding_artifact.key
                    ),
                    default=0,
                ):
                    reject(EffectCode.CONTRADICTORY)

    def _validate_passthrough_parents(self) -> None:
        for target, activation, port, parent in self.result._passthrough_parents:
            if activation not in self.entries or target != self.owners[activation] or parent.target != target:
                reject(EffectCode.FOREIGN_OWNER)
            key = (target, activation, port)
            if key in self.passthrough_parents:
                reject(EffectCode.DUPLICATE)
            self.passthrough_parents[key] = parent
        expected = set()
        for fact in self.result.provenance:
            key = fact.key
            if not isinstance(key, OperationOutputKey) or key.activation not in self.entries:
                continue
            node = self.nodes[self.entries[key.activation].template]
            if isinstance(node, SubgraphNode) and any(
                binding.destination.port == key.port and isinstance(binding.source, WorkflowInputRef)
                for binding in node.body.output_bindings
            ):
                expected.add((key.target, key.activation, key.port))
        if expected - self.passthrough_parents.keys():
            reject(EffectCode.MISSING)
        if self.passthrough_parents.keys() - expected:
            reject(EffectCode.CONTRADICTORY)
        if any(parent not in self.provenance for parent in self.passthrough_parents.values()):
            reject(EffectCode.MISSING)

    def _provenance_acyclic(self) -> None:
        remaining = {key: len(fact.parents) for key, fact in self.provenance.items()}
        children: dict[ProvenanceKey, list[ProvenanceKey]] = {key: [] for key in remaining}
        for key, fact in self.provenance.items():
            for parent in fact.parents:
                children[parent].append(key)
        ready = [key for key, count in remaining.items() if count == 0]
        visited = 0
        while ready:
            key = ready.pop()
            visited += 1
            for child in children[key]:
                remaining[child] -= 1
                if remaining[child] == 0:
                    ready.append(child)
        if visited != len(remaining):
            reject(EffectCode.CONTRADICTORY)

    def _producer(self, fact: ArtifactProvenanceFact) -> None:
        key = fact.key
        if isinstance(key, OperationOutputKey):
            entry = self.entries.get(key.activation)
            if entry is None:
                reject(EffectCode.MISSING)
            if self.owners[key.activation] != key.target:
                reject(EffectCode.FOREIGN_OWNER)
            port = self.facts.ports.get((key.activation, key.port))
            if port is None or port.artifact != fact.artifact or port.node != entry.template:
                reject(EffectCode.CONTRADICTORY)
            outcome = next(
                (item for item in self.nodes[entry.template].operation.outcomes if item.name == entry.outcome), None
            )
            if outcome is None or key.port not in outcome.produced_ports:
                reject(EffectCode.CONTRADICTORY)
            node = self.nodes[entry.template]
            if isinstance(node, SubgraphNode):
                bindings = [item for item in node.body.output_bindings if item.destination.port == key.port]
                if len(bindings) != 1 or len(fact.parents) != 1:
                    reject(EffectCode.CONTRADICTORY)
                source = bindings[0].source
                parent = self.provenance[next(iter(fact.parents))]
                if isinstance(source, WorkflowInputRef):
                    expected_key = self.passthrough_parents[key.target, key.activation, key.port]
                    if parent.key != expected_key:
                        reject(EffectCode.CONTRADICTORY)
                else:
                    if not isinstance(parent.key, OperationOutputKey):
                        reject(EffectCode.CONTRADICTORY)
                    state = self.result.states[self.order.index(key.target)]
                    expected_activation = _source_activation(state, key.activation, source.node)
                    source_entry = self.entries.get(parent.key.activation)
                    if (
                        source_entry is None
                        or parent.key.activation != expected_activation
                        or parent.key.target != key.target
                        or source_entry.template != source.node
                        or parent.key.port != source.port
                    ):
                        reject(EffectCode.CONTRADICTORY)
                if parent.artifact != fact.artifact or fact.decision != parent.decision:
                    reject(EffectCode.CONTRADICTORY)
            else:
                dependency = next(item for item in node.operation.output_dependencies if item.output == key.port)
                input_keys = [(key.target, key.activation, name) for name in dependency.inputs]
                if any(item not in self.input_parents for item in input_keys):
                    reject(EffectCode.MISSING)
                expected = frozenset(self.input_parents[item] for item in input_keys)
                if dependency.identity_input is not None:
                    identity_parent = self.input_parents[key.target, key.activation, dependency.identity_input]
                    if fact.artifact != self.provenance[identity_parent].artifact:
                        reject(EffectCode.CONTRADICTORY)
                if fact.parents != expected or fact.decision != any(
                    item.node == entry.template for item in self.admitted.execution.decisions
                ):
                    reject(EffectCode.CONTRADICTORY)
        elif isinstance(key, RootInputKey):
            if (
                fact.parents
                or fact.decision
                or not any(item.target == key.target and item.port == key.port for item in self.prepared.bound_inputs)
            ):
                reject(EffectCode.CONTRADICTORY)
        elif isinstance(key, (BoundInputKey, InitialCollectionKey)):
            self._bound_producer(fact, key)
        elif isinstance(key, MapItemKey):
            self._map_producer(fact, key)

    def _bound_producer(self, fact: ArtifactProvenanceFact, key: BoundInputKey | InitialCollectionKey) -> None:
        bound = self.admitted.execution.context.bound_context
        if bound is None or fact.decision:
            reject(EffectCode.CONTRADICTORY)
        declaration = key.binding_artifact.declaration if isinstance(key, BoundInputKey) else key.declaration
        source = next((item for item in bound.receipt.sources if item.identity == declaration), None)
        if source is None:
            reject(EffectCode.MISSING)
        if (source.declaration.target, source.declaration.node, source.declaration.port) != (
            key.target,
            key.node,
            key.port,
        ):
            reject(EffectCode.FOREIGN_OWNER)
        items = [item for item in bound.artifacts if item.reference.declaration == declaration]
        if isinstance(key, BoundInputKey):
            if fact.parents or not any(
                item.reference == key.binding_artifact
                and (item.target, item.node, item.port) == (key.target, key.node, key.port)
                for item in items
            ):
                reject(EffectCode.CONTRADICTORY)
        else:
            expected = frozenset(
                BoundInputKey(target=key.target, node=key.node, port=key.port, binding_artifact=item.reference)
                for item in items
            )
            if source.declaration.materialization.kind != "collection" or fact.parents != expected:
                reject(EffectCode.CONTRADICTORY)
            value = dict(self.result.artifacts)[fact.artifact]
            if not isinstance(value, TextCollectionValue) or {(item.key, item.version) for item in value.items} != {
                (item.reference.key, item.reference.version) for item in items
            }:
                reject(EffectCode.CONTRADICTORY)

    def _map_producer(self, fact: ArtifactProvenanceFact, key: MapItemKey) -> None:
        member = self.entries.get(key.member)
        expander = self.entries.get(key.expander)
        if member is None or expander is None:
            reject(EffectCode.MISSING)
        if (
            key.member.parent != key.expander
            or self.owners[key.member] != key.target
            or self.owners[key.expander] != key.target
        ):
            reject(EffectCode.FOREIGN_OWNER)
        declaration = next(
            (
                item
                for scope in self.prepared.workflow.scopes
                for item in scope.maps
                if item.expander == expander.template
            ),
            None,
        )
        publication = next(
            (
                item
                for item in self.admitted.execution.map_expansions
                if item.expander == expander.template and item.outcome == expander.outcome
            ),
            None,
        )
        if declaration is None or publication is None:
            reject(EffectCode.MISSING)
        if declaration.member != member.template or declaration.item_input != key.port or fact.decision:
            reject(EffectCode.CONTRADICTORY)
        parent = OperationOutputKey(activation=key.expander, target=key.target, port=publication.membership_port)
        if fact.parents != frozenset({parent}):
            reject(EffectCode.CONTRADICTORY)
        collection = dict(self.result.artifacts)[self.provenance[parent].artifact]
        if not isinstance(collection, TextCollectionValue):
            reject(EffectCode.CONTRADICTORY)
        if (key.item_key, key.item_version) not in {(item.key, item.version) for item in collection.items}:
            reject(EffectCode.MISSING)
        state = self.result.states[self.order.index(key.target)]
        expansion = next((item for item in state.expansions if item.parent == key.expander), None)
        if expansion is None or key.member not in expansion.members:
            reject(EffectCode.MISSING)
        children = sorted(expansion.members, key=lambda item: item.occurrence)
        index = children.index(key.member)
        if index >= len(collection.items) or (key.item_key, key.item_version) != (
            collection.items[index].key,
            collection.items[index].version,
        ):
            reject(EffectCode.CONTRADICTORY)

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

    def _association_target(self, association: RequestAssociation) -> DatumId | None:
        if isinstance(association, SemanticAssociation):
            terminal = next((item for item in self.result.record.terminals if item.attempt == association.task), None)
            return None if terminal is None else self.owners.get(terminal.activation)
        bound = self.admitted.execution.context.bound_context
        if bound is None:
            return None
        source = next((item for item in bound.receipt.sources if item.identity == association.declaration), None)
        return None if source is None else source.declaration.target

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
            targets = {self._association_target(item) for item in dispatch.associations}
            if None in targets or dispatch.request.scope != receipt.scope:
                unlocalized = True
                continue
            affected = {item for item in targets if item is not None}
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
                    not isinstance(settlements[0].usage, ExactUsage) or settlements[0].remote_stopped is not True
                )
            if terminals:
                terminal = terminals[0]
                bad |= terminal.category in {"lost", "inconsistent", "cancelled"}
                uncertain |= terminal.category in {"lost", "inconsistent"}
                if terminal.category == "success":
                    associations = [item.association for item in terminal.results]
                    bad |= (
                        len(associations) != len(set(associations)) or frozenset(associations) != dispatch.associations
                    )
                elif terminal.category == "failure":
                    for association in dispatch.associations:
                        if not self._recovered(dispatch, receipt, association):
                            target = self._association_target(association)
                            if target is not None:
                                self.withholding[target].add("request_accounting")
            if dispatch.purpose in {"retry", "correction", "failover"}:
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
                        or not self._followup(dispatch, prior)
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
    def _followup(dispatch: RequestReservation, prior: RequestTerminalFact) -> bool:
        failure = prior.failure
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
        later = False
        for successor in receipt.dispatches:
            if successor.request == dispatch.request:
                later = True
                continue
            if later and association in successor.associations:
                terminal = next((item for item in receipt.terminals if item.request == successor.request), None)
                if terminal is not None and terminal.category == "success":
                    return True
        return False

    def cleanup_accounting(self) -> None:
        self._cleanup_ledger(self.result.cleanup, self.result.cleanup_associations)
        bound = self.admitted.execution.context.bound_context
        if bound is not None:
            self._cleanup_ledger(bound.receipt.cleanup, bound.receipt.cleanup_associations, binding=True)

    def _cleanup_ledger(
        self,
        cleanup: tuple[CleanupFact, ...],
        cleanup_associations: tuple[CleanupAssociation, ...],
        *,
        binding: bool = False,
    ) -> None:
        facts = {item.resource: item for item in cleanup}
        associations = {item.resource: item for item in cleanup_associations}
        if (
            len(facts) != len(cleanup)
            or len(associations) != len(cleanup_associations)
            or facts.keys() != associations.keys()
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
        path = []
        parent = activation.parent
        while parent is not None:
            entry = self.entries[parent]
            if isinstance(self.nodes[entry.template], SubgraphNode):
                path.append(entry.template)
            parent = parent.parent
        return tuple(reversed(path))

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
        requirements = tuple(
            item for item in self.prepared.workflow.workflow.protection_requirements if item.outcome == outcome
        )
        if outcome not in self.prepared.configuration.required_protection_outcomes or not requirements:
            self.withholding[target].add("missing_assessment")
            return
        for requirement in requirements:
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
            fact = self.provenance[key]
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
