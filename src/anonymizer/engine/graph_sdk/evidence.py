# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure admission and authentication of execution-produced evidence."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from itertools import product
from typing import Literal

from anonymizer.engine.graph_sdk._effect_values import (
    EffectCode,
    PrivateValue,
    reject,
    require_count,
    require_instance,
)
from anonymizer.engine.graph_sdk._workflow import reachable_workflows
from anonymizer.engine.graph_sdk.capabilities import FrozenConfig, config_size
from anonymizer.engine.graph_sdk.executor import (
    AdmittedExecutionPlan,
    AssessmentEnvironment,
    AssessmentFinding,
    EvidenceProductionDecl,
    ExecutionAssessmentFact,
    ExecutionPortFact,
    ExecutionResult,
    MapItemKey,
    OperationOutputKey,
)
from anonymizer.engine.graph_sdk.preparation import StateRevisionView
from anonymizer.engine.graph_sdk.records import AbsenceRef, CandidateRef, ConsumedRef, DecisionRef, EvidenceRef
from anonymizer.engine.graph_sdk.requests import TextCollectionValue
from anonymizer.graph._values import ActivationKey, ArtifactRef, ContractViolation, DatumId
from anonymizer.graph._workflow_composition import _project_evidence_port, _selected_nodes
from anonymizer.graph.workflow import (
    AdmittedWorkflow,
    ContextInputRef,
    CoverageAtom,
    EvidencePromise,
    MapItemPort,
    NodeId,
    OperationNode,
    OperationSpec,
    OutcomeSpec,
    ProtectionRequirement,
    SubgraphNode,
    Vertex,
    WorkflowInputRef,
)

Validity = Literal["current", "stale", "unknown"]
_EVIDENCE_KEY = object()


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class QualificationLimits(PrivateValue):
    max_productions: int
    max_submissions: int
    max_port_facts: int
    max_consumed_per_assessment: int
    max_coverage_atoms: int
    max_verified_evidence: int
    max_required_decisions: int
    max_fixed_point_steps: int
    max_revision_entries: int
    max_absence_revisions: int
    max_provenance_edges: int

    def __post_init__(self) -> None:
        for descriptor in fields(self):
            require_count(getattr(self, descriptor.name), positive=descriptor.name == "max_fixed_point_steps")


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class AssessmentSubmission(PrivateValue):
    fact: ExecutionAssessmentFact

    def __post_init__(self) -> None:
        require_instance(self.fact, ExecutionAssessmentFact)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class _Production(PrivateValue):
    declaration: EvidenceProductionDecl
    operation: OperationSpec
    outcome: OutcomeSpec
    promise: EvidencePromise


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class AdmittedQualification(PrivateValue):
    execution: AdmittedExecutionPlan
    productions: tuple[EvidenceProductionDecl, ...]
    limits: QualificationLimits
    _resolved: tuple[_Production, ...]
    _projections: tuple[_ProjectedPromise, ...]

    def __init__(
        self,
        *,
        _key: object,
        execution: AdmittedExecutionPlan,
        productions: tuple[EvidenceProductionDecl, ...],
        limits: QualificationLimits,
        _resolved: tuple[_Production, ...],
        _projections: tuple[_ProjectedPromise, ...],
    ) -> None:
        if _key is not _EVIDENCE_KEY:
            raise TypeError("qualification admission requires admit_qualification")
        for name, value in locals().copy().items():
            if name not in {"self", "_key"}:
                object.__setattr__(self, name, value)

    def _subject_outputs(self, outcome: str) -> frozenset[str]:
        """Return final ports that carry this outcome's protection subjects."""
        workflow = self.execution.context.prepared.workflow.workflow
        if self.execution.context.prepared.configuration.purpose == "execution_only":
            return frozenset(port.name for port in workflow.interface.outputs)
        subjects = {item.subject_port for item in workflow.protection_requirements if item.outcome == outcome}
        return frozenset(
            dependency.output
            for dependency in workflow.interface.output_dependencies
            if dependency.output in subjects or dependency.identity_input in subjects
        ) | frozenset(
            item.candidate_port
            for item in workflow.protection_requirements
            if item.outcome == outcome and item.candidate_port is not None
        )


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class MapItemSubjectRef(PrivateValue):
    """An assessed scalar item and the exact occurrence that materialized it."""

    artifact: ArtifactRef
    producer: MapItemKey

    def __post_init__(self) -> None:
        require_instance(self.artifact, ArtifactRef)
        require_instance(self.producer, MapItemKey)
        if self.artifact.invocation != self.producer.member.invocation:
            reject(EffectCode.FOREIGN_OWNER)
        if self.artifact.version != self.producer.item_version:
            reject(EffectCode.CONTRADICTORY)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class VerifiedEvidence(PrivateValue):
    reference: EvidenceRef
    activation: ActivationKey
    node: NodeId
    outcome: str
    promise: EvidencePromise
    subject: CandidateRef | MapItemSubjectRef
    consumed_by_port: tuple[tuple[str, ConsumedRef], ...]
    environment: AssessmentEnvironment
    finding: AssessmentFinding
    coverage: frozenset[CoverageAtom]
    _admitted: AdmittedQualification
    _result: ExecutionResult

    def __init__(self, *, _key: object, **values: object) -> None:
        if _key is not _EVIDENCE_KEY:
            raise TypeError("verified evidence requires verify_evidence")
        for name, value in values.items():
            object.__setattr__(self, name, value)


def admit_qualification(
    *,
    execution: AdmittedExecutionPlan,
    productions: tuple[EvidenceProductionDecl, ...],
    limits: QualificationLimits,
) -> AdmittedQualification:
    """Resolve immutable evidence declarations without executing callbacks."""
    require_instance(execution, AdmittedExecutionPlan)
    require_instance(limits, QualificationLimits)
    require_instance(productions, tuple)
    if len(productions) > limits.max_productions or limits.max_fixed_point_steps < len(
        execution.context.prepared.data.targets
    ):
        reject(EffectCode.LIMIT_EXCEEDED)
    if any(not isinstance(item, EvidenceProductionDecl) for item in productions):
        reject(EffectCode.INVALID_TYPE)
    prepared = execution.context.prepared
    operations = {
        node.id: node.operation
        for body in reachable_workflows(prepared.workflow)
        for node in body.nodes
        if isinstance(node, OperationNode)
    }
    owners = {body.workflow for body in reachable_workflows(prepared.workflow)}
    if any(item.node.workflow not in owners for item in productions):
        reject(EffectCode.FOREIGN_OWNER)
    keys = [(item.node, item.outcome, item.promise) for item in productions]
    if len(keys) != len(set(keys)):
        reject(EffectCode.DUPLICATE)
    if any(item.node not in operations for item in productions):
        reject(EffectCode.MISSING)
    resolved = tuple(_resolve_production(item, operations[item.node], limits) for item in productions)
    projections = _PromiseProjection(execution).all()
    requirements = tuple(
        item
        for item in prepared.workflow.workflow.protection_requirements
        if item.outcome in prepared.configuration.required_protection_outcomes
    )
    required = {
        (item.node, item.outcome, item.promise.name)
        for item in projections
        if any(item.matches(requirement) for requirement in requirements)
    }
    observed = set(keys)
    if required - observed:
        reject(EffectCode.MISSING)
    if observed - required or any(item not in execution.assessment_productions for item in productions):
        reject(EffectCode.UNSUPPORTED)
    node_order = {
        node.id: index
        for index, node in enumerate(node for body in reachable_workflows(prepared.workflow) for node in body.nodes)
    }
    resolved = tuple(
        sorted(resolved, key=lambda item: (node_order[item.declaration.node], item.outcome.name, item.promise.name))
    )
    return AdmittedQualification(
        _key=_EVIDENCE_KEY,
        execution=execution,
        productions=tuple(item.declaration for item in resolved),
        limits=limits,
        _resolved=resolved,
        _projections=tuple(item for item in projections if (item.node, item.outcome, item.promise.name) in observed),
    )


def _resolve_production(
    declaration: EvidenceProductionDecl, operation: OperationSpec, limits: QualificationLimits
) -> _Production:
    outcome = next((item for item in operation.outcomes if item.name == declaration.outcome), None)
    if outcome is None:
        reject(EffectCode.UNSUPPORTED)
    promise = next((item for item in outcome.evidence if item.name == declaration.promise), None)
    if promise is None or declaration.evidence_port not in outcome.produced_ports:
        reject(EffectCode.UNSUPPORTED)
    if (
        len(promise.consumed_ports) > limits.max_consumed_per_assessment
        or len(promise.coverage) > limits.max_coverage_atoms
    ):
        reject(EffectCode.LIMIT_EXCEEDED)
    return _Production(declaration=declaration, operation=operation, outcome=outcome, promise=promise)


@dataclass(slots=True)
class _EvidenceFacts:
    """Bounded occurrence indexes for authenticating one result's assessments."""

    admitted: AdmittedQualification
    result: ExecutionResult
    ports: dict[tuple[ActivationKey, str], ExecutionPortFact]
    artifacts: frozenset[ArtifactRef]

    @classmethod
    def from_result(cls, admitted: AdmittedQualification, result: ExecutionResult) -> _EvidenceFacts:
        limits = admitted.limits
        execution = admitted.execution
        prepared = execution.context.prepared
        if (
            max(len(result.ports), len(result._input_parents), len(result._passthrough_parents))
            > min(limits.max_port_facts, execution.assessment_limits.max_port_facts)
            or len(result.assessments) > execution.assessment_limits.max_assessment_facts
            or sum(len(item.parents) for item in result.provenance)
            > min(limits.max_provenance_edges, execution.assessment_limits.max_provenance_edges)
        ):
            reject(EffectCode.LIMIT_EXCEEDED)
        record = result.record
        if (
            result._execution is not execution
            or record.plan != prepared.plan
            or record.graph != prepared.data.graph
            or record.targets != prepared.data.targets
        ):
            reject(EffectCode.FOREIGN_OWNER)
        if any(reference.invocation != record.invocation for reference, _ in result.artifacts):
            reject(EffectCode.FOREIGN_OWNER)
        references = [reference for reference, _ in result.artifacts]
        if len(references) != len(set(references)):
            reject(EffectCode.DUPLICATE)
        if frozenset(references) != record.artifacts:
            reject(EffectCode.CONTRADICTORY)
        ports: dict[tuple[ActivationKey, str], ExecutionPortFact] = {}
        for fact in result.ports:
            if fact.activation.invocation != record.invocation or fact.target not in record.targets:
                reject(EffectCode.FOREIGN_OWNER)
            key = (fact.activation, fact.port)
            if key in ports:
                reject(EffectCode.DUPLICATE)
            if fact.artifact not in record.artifacts:
                reject(EffectCode.MISSING)
            ports[key] = fact
        return cls(admitted=admitted, result=result, ports=ports, artifacts=record.artifacts)

    def port(self, fact: ExecutionAssessmentFact, name: str) -> ExecutionPortFact:
        port = self.ports.get((fact.activation, name))
        if port is None:
            reject(EffectCode.MISSING)
        if port.node != fact.node:
            reject(EffectCode.FOREIGN_OWNER)
        return port

    def production(self, fact: ExecutionAssessmentFact) -> _Production:
        production = next(
            (
                item
                for item in self.admitted._resolved
                if (item.declaration.node, item.outcome.name, item.promise.name)
                == (fact.node, fact.outcome, fact.promise)
            ),
            None,
        )
        if production is None:
            reject(EffectCode.UNSUPPORTED)
        return production

    def verify(self, fact: ExecutionAssessmentFact) -> VerifiedEvidence:
        if not any(fact is retained for retained in self.result.assessments):
            reject(EffectCode.FOREIGN_OWNER)
        production = self.production(fact)
        target = self._verify_activation(fact)
        evidence = self.port(fact, production.declaration.evidence_port)
        if not isinstance(production.promise.subject_port, str):
            reject(EffectCode.UNSUPPORTED)
        subject = self.port(fact, production.promise.subject_port)
        scoped_subjects = {
            item.promise.subject_port
            for item in self.admitted._projections
            if item.node == fact.node
            and item.outcome == fact.outcome
            and item.promise.name == fact.promise
            and isinstance(item.promise.subject_port, MapItemPort)
        }
        subject_reference: CandidateRef | MapItemSubjectRef
        if scoped_subjects:
            matches = [self.map_item(fact, subject.port, endpoint) for endpoint in scoped_subjects]
            actual = [item for item in matches if item is not None]
            if len(set(actual)) != 1 or subject.role != "artifact":
                reject(EffectCode.CONTRADICTORY)
            subject_reference = actual[0]
        else:
            if subject.role != "candidate":
                reject(EffectCode.CONTRADICTORY)
            subject_reference = CandidateRef(artifact=subject.artifact, target=target)
        if evidence.target != target or subject.target != target:
            reject(EffectCode.FOREIGN_OWNER)
        expected_role = self.admitted.execution._output_role(fact.node, production.outcome, evidence.port)
        if evidence.artifact != fact.evidence_artifact or evidence.role != expected_role:
            reject(EffectCode.CONTRADICTORY)
        consumed: list[tuple[str, ConsumedRef]] = []
        input_names = {item.name for item in production.operation.inputs}
        names: list[str] = []
        for name in production.promise.consumed_ports:
            if not isinstance(name, str):
                reject(EffectCode.UNSUPPORTED)
            names.append(name)
        for name in sorted(names):
            port = self.port(fact, name)
            if name not in input_names or port.target != target:
                reject(EffectCode.CONTRADICTORY)
            reference: ConsumedRef
            if port.role == "candidate":
                reference = CandidateRef(artifact=port.artifact, target=target)
            elif port.role == "decision":
                reference = DecisionRef(artifact=port.artifact)
            elif port.role in ("artifact", "evidence"):
                reference = port.artifact
            else:
                reject(EffectCode.UNSUPPORTED)
            consumed.append((name, reference))
        scoped_consumed = {
            port
            for item in self.admitted._projections
            if item.node == fact.node and item.outcome == fact.outcome and item.promise.name == fact.promise
            for port in item.promise.consumed_ports
            if isinstance(port, MapItemPort)
        }
        if scoped_consumed and not any(
            self.map_item(fact, name, endpoint) is not None for endpoint in scoped_consumed for name in names
        ):
            reject(EffectCode.CONTRADICTORY)
        self._verify_environment(fact, production)
        if fact.finding not in production.declaration.supported_findings:
            reject(EffectCode.UNSUPPORTED)
        return VerifiedEvidence(
            _key=_EVIDENCE_KEY,
            reference=EvidenceRef(
                artifact=evidence.artifact,
                consumed=frozenset(reference for _, reference in consumed) | fact.environment.absences,
            ),
            activation=fact.activation,
            node=fact.node,
            outcome=fact.outcome,
            promise=production.promise,
            subject=subject_reference,
            consumed_by_port=tuple(consumed),
            environment=fact.environment,
            finding=fact.finding,
            coverage=production.promise.coverage,
            _admitted=self.admitted,
            _result=self.result,
        )

    def map_item(self, fact: ExecutionAssessmentFact, port: str, endpoint: MapItemPort) -> MapItemSubjectRef | None:
        """Authenticate an endpoint against the captured input's materialization."""
        input_fact = self.port(fact, port)
        parents = [
            key
            for target, activation, name, key in self.result._input_parents
            if (target, activation, name) == (input_fact.target, fact.activation, port)
        ]
        if len(parents) != 1 or not isinstance(parents[0], MapItemKey):
            return None
        key = parents[0]
        entries = {entry.activation: entry for state in self.result.states for entry in state.entries}
        expander = entries.get(key.expander)
        member = entries.get(key.member)
        if (
            expander is None
            or member is None
            or expander.template != endpoint.expander
            or member.template != endpoint.member
            or expander.outcome != endpoint.expansion_outcome
            or key.port != endpoint.item_input
            or key.target != input_fact.target
        ):
            return None
        ancestors: set[ActivationKey] = set()
        current: ActivationKey | None = fact.activation
        while current is not None:
            ancestors.add(current)
            current = current.parent
        if key.member not in ancestors:
            return None
        path: list[NodeId] = []
        nodes = {
            node.id: node
            for body in reachable_workflows(self.admitted.execution.context.prepared.workflow)
            for node in body.nodes
        }
        current = key.expander.parent
        while current is not None:
            entry = entries.get(current)
            if entry is None:
                return None
            if isinstance(nodes.get(entry.template), SubgraphNode):
                path.append(entry.template)
            current = current.parent
        if tuple(reversed(path)) != endpoint.path:
            return None
        provenance = [item for item in self.result.provenance if item.key == key]
        if len(provenance) != 1 or provenance[0].artifact != input_fact.artifact or provenance[0].decision:
            reject(EffectCode.CONTRADICTORY)
        expected = OperationOutputKey(activation=key.expander, target=key.target, port=endpoint.membership_port)
        if provenance[0].parents != frozenset({expected}):
            reject(EffectCode.CONTRADICTORY)
        publications = [item for item in self.result.provenance if item.key == expected]
        if len(publications) != 1:
            reject(EffectCode.CONTRADICTORY)
        artifacts = dict(self.result.artifacts)
        collection = artifacts.get(publications[0].artifact)
        if not isinstance(collection, TextCollectionValue):
            reject(EffectCode.CONTRADICTORY)
        source_items = [
            item for item in collection.items if (item.key, item.version) == (key.item_key, key.item_version)
        ]
        if len(source_items) != 1 or source_items[0].value != artifacts.get(input_fact.artifact):
            reject(EffectCode.CONTRADICTORY)
        expansions = [
            expansion
            for state in self.result.states
            for expansion in state.expansions
            if expansion.parent == key.expander
        ]
        if len(expansions) != 1 or key.member not in expansions[0].members:
            reject(EffectCode.CONTRADICTORY)
        members = sorted(expansions[0].members, key=lambda item: item.occurrence)
        index = members.index(key.member)
        if index >= len(collection.items) or collection.items[index] != source_items[0]:
            reject(EffectCode.CONTRADICTORY)
        return MapItemSubjectRef(artifact=input_fact.artifact, producer=key)

    def _verify_activation(self, fact: ExecutionAssessmentFact) -> DatumId:
        prepared = self.admitted.execution.context.prepared
        matches = [
            (target.target, entry)
            for target, state in zip(prepared.target_occurrences, self.result.states, strict=True)
            for entry in state.entries
            if entry.activation == fact.activation
        ]
        if len(matches) != 1:
            reject(EffectCode.MISSING if not matches else EffectCode.DUPLICATE)
        target, entry = matches[0]
        terminals = [item for item in self.result.record.terminals if item.activation == fact.activation]
        if len(terminals) != 1:
            reject(EffectCode.MISSING if not terminals else EffectCode.DUPLICATE)
        terminal = terminals[0]
        if entry.template != fact.node or fact.activation.invocation != self.result.record.invocation:
            reject(EffectCode.FOREIGN_OWNER)
        if (
            entry.status != "success"
            or entry.outcome != fact.outcome
            or terminal.category != "success"
            or terminal.attempt is None
            or terminal.structural
        ):
            reject(EffectCode.CONTRADICTORY)
        return target

    def _verify_environment(self, fact: ExecutionAssessmentFact, production: _Production) -> None:
        execution = self.admitted.execution
        policy = next(item for item in execution.policies if item.node == fact.node)
        if any(item.invocation != self.result.record.invocation for item in fact.environment.absences):
            reject(EffectCode.FOREIGN_OWNER)
        queries = [item.query for item in fact.environment.absences]
        if len(queries) != len(set(queries)):
            reject(EffectCode.DUPLICATE)
        if frozenset(queries) != production.declaration.absence_queries:
            reject(EffectCode.MISSING)
        if fact.environment.configuration not in {item.configuration for item in policy.implementations}:
            reject(EffectCode.CONTRADICTORY)
        reads = frozenset(item for item in production.outcome.state_effects if item.kind == "read")
        expected = frozenset(item for item in execution.context.prepared.state.revisions if item.effect in reads)
        if fact.environment.state.revisions != expected:
            reject(EffectCode.CONTRADICTORY)


def verify_evidence(
    *, admitted: AdmittedQualification, result: ExecutionResult, submissions: tuple[AssessmentSubmission, ...]
) -> tuple[VerifiedEvidence, ...]:
    """Authenticate retained assessment objects and derive their exact dependencies."""
    require_instance(admitted, AdmittedQualification)
    require_instance(result, ExecutionResult)
    require_instance(submissions, tuple)
    if len(submissions) > min(admitted.limits.max_submissions, admitted.limits.max_verified_evidence):
        reject(EffectCode.LIMIT_EXCEEDED)
    if any(not isinstance(item, AssessmentSubmission) for item in submissions):
        reject(EffectCode.INVALID_TYPE)
    facts = _EvidenceFacts.from_result(admitted, result)
    if any(not any(item.fact is fact for fact in result.assessments) for item in submissions):
        reject(EffectCode.FOREIGN_OWNER)
    identities = [item.fact.evidence_artifact for item in submissions]
    if len(identities) != len(set(identities)):
        reject(EffectCode.DUPLICATE)
    verified = tuple(facts.verify(item.fact) for item in submissions)
    prepared = admitted.execution.context.prepared
    target_order = {item.target: index for index, item in enumerate(prepared.target_occurrences)}
    node_order = {
        node.id: index
        for index, node in enumerate(node for body in reachable_workflows(prepared.workflow) for node in body.nodes)
    }
    productions = {(item.declaration.node, item.outcome.name, item.promise.name): item for item in admitted._resolved}

    def canonical_order(item: VerifiedEvidence) -> tuple[int, int, int, int, int, int]:
        production = productions[item.node, item.outcome, item.promise.name]
        port = production.declaration.evidence_port
        target = facts.ports[item.activation, port].target
        port_order = next(index for index, output in enumerate(production.operation.outputs) if output.name == port)
        artifact = item.reference.artifact
        return (
            target_order[target],
            item.activation.occurrence,
            node_order[item.node],
            port_order,
            artifact.key,
            artifact.version,
        )

    return tuple(sorted(verified, key=canonical_order))


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class EvidenceRevisionView(PrivateValue):
    candidates: frozenset[CandidateRef]
    artifacts: frozenset[ArtifactRef]
    decisions: frozenset[DecisionRef]
    absences: frozenset[AbsenceRef]
    configurations: tuple[tuple[NodeId, FrozenConfig], ...]
    state: StateRevisionView
    _admitted: AdmittedQualification
    _result: ExecutionResult

    def __init__(self, *, _key: object, **values: object) -> None:
        if _key is not _EVIDENCE_KEY:
            raise TypeError("revision views require evidence_revision_view")
        for name, value in values.items():
            object.__setattr__(self, name, value)


def evidence_revision_view(
    *,
    admitted: AdmittedQualification,
    result: ExecutionResult,
    artifacts: tuple[ArtifactRef, ...],
    absences: tuple[AbsenceRef, ...],
    configurations: tuple[tuple[NodeId, FrozenConfig], ...],
    state: StateRevisionView,
) -> EvidenceRevisionView:
    """Bind explicit current revisions to one admitted execution result."""
    require_instance(admitted, AdmittedQualification)
    require_instance(result, ExecutionResult)
    require_instance(artifacts, tuple)
    require_instance(absences, tuple)
    require_instance(configurations, tuple)
    require_instance(state, StateRevisionView)
    limits = admitted.limits
    if (
        len(artifacts) + len(configurations) + len(state.revisions) > limits.max_revision_entries
        or len(absences) > limits.max_absence_revisions
    ):
        reject(EffectCode.LIMIT_EXCEEDED)
    if any(not isinstance(item, ArtifactRef) for item in artifacts) or any(
        not isinstance(item, AbsenceRef) for item in absences
    ):
        reject(EffectCode.INVALID_TYPE)
    if any(
        not isinstance(item, tuple)
        or len(item) != 2
        or not isinstance(item[0], NodeId)
        or not isinstance(item[1], FrozenConfig)
        for item in configurations
    ):
        reject(EffectCode.INVALID_TYPE)
    _EvidenceFacts.from_result(admitted, result)
    _validate_current_revisions(admitted, result, artifacts, absences, configurations, state)
    selected = frozenset(artifacts)
    candidates = frozenset(
        item.candidate
        for item in result.final_outputs
        if item.port in admitted._subject_outputs(item.outcome) and item.candidate.artifact in selected
    )
    if len({item.target for item in candidates}) != len(candidates):
        reject(EffectCode.CONTRADICTORY)
    prepared = admitted.execution.context.prepared
    order = {
        node.id: index
        for index, node in enumerate(node for body in reachable_workflows(prepared.workflow) for node in body.nodes)
    }
    return EvidenceRevisionView(
        _key=_EVIDENCE_KEY,
        candidates=candidates,
        artifacts=selected,
        decisions=frozenset(
            DecisionRef(artifact=item.artifact)
            for item in result.ports
            if item.role == "decision" and item.artifact in selected
        ),
        absences=frozenset(absences),
        configurations=tuple(sorted(configurations, key=lambda item: order[item[0]])),
        state=state,
        _admitted=admitted,
        _result=result,
    )


def _validate_current_revisions(
    admitted: AdmittedQualification,
    result: ExecutionResult,
    artifacts: tuple[ArtifactRef, ...],
    absences: tuple[AbsenceRef, ...],
    configurations: tuple[tuple[NodeId, FrozenConfig], ...],
    state: StateRevisionView,
) -> None:
    prepared = admitted.execution.context.prepared
    nodes = {item.node for item in prepared.implementations}
    queries = {query for item in admitted.productions for query in item.absence_queries}
    reads = {
        effect
        for item in prepared.implementations
        for outcome in item.capability.operation.outcomes
        for effect in outcome.state_effects
        if effect.kind == "read"
    }
    if any(item.invocation != result.record.invocation for item in (*artifacts, *absences)) or any(
        node.workflow not in {item.workflow for item in nodes} for node, _ in configurations
    ):
        reject(EffectCode.FOREIGN_OWNER)
    keys = (
        [(item.invocation, item.key) for item in artifacts],
        [(item.invocation, item.query) for item in absences],
        [node for node, _ in configurations],
        [item.effect for item in state.revisions],
    )
    if any(len(group) != len(set(group)) for group in keys):
        reject(EffectCode.DUPLICATE)
    if any(node not in nodes for node, _ in configurations) or any(
        item not in result.record.artifacts for item in artifacts
    ):
        reject(EffectCode.MISSING)
    if any(item.query not in queries for item in absences) or any(item.effect not in reads for item in state.revisions):
        reject(EffectCode.UNSUPPORTED)
    for _, configuration in configurations:
        depth, atoms = config_size(configuration)
        if depth > prepared.limits.max_config_depth or atoms > prepared.limits.max_config_atoms:
            reject(EffectCode.LIMIT_EXCEEDED)


def evidence_validity(*, evidence: VerifiedEvidence, current: EvidenceRevisionView) -> Validity:
    """Compare declared dependencies; known changes outrank missing current keys."""
    require_instance(evidence, VerifiedEvidence)
    require_instance(current, EvidenceRevisionView)
    if evidence._admitted is not current._admitted or evidence._result is not current._result:
        reject(EffectCode.FOREIGN_OWNER)
    artifacts = {(item.invocation, item.key): item for item in current.artifacts}
    candidates = {item.target: item for item in current.candidates}
    decisions = {(item.artifact.invocation, item.artifact.key): item for item in current.decisions}
    absences = {(item.invocation, item.query): item for item in current.absences}
    configurations = dict(current.configurations)
    revisions = {item.effect: item.revision for item in current.state.revisions}
    comparisons: list[tuple[object, object | None]] = [
        (
            evidence.reference.artifact,
            artifacts.get((evidence.reference.artifact.invocation, evidence.reference.artifact.key)),
        ),
        (evidence.environment.configuration, configurations.get(evidence.node)),
    ]
    if isinstance(evidence.subject, MapItemSubjectRef):
        artifact = evidence.subject.artifact
        comparisons.append((artifact, artifacts.get((artifact.invocation, artifact.key))))
    else:
        comparisons.append((evidence.subject, candidates.get(evidence.subject.target)))
    for reference in evidence.reference.consumed:
        if isinstance(reference, CandidateRef):
            comparisons.append((reference, candidates.get(reference.target)))
            comparisons.append(
                (reference.artifact, artifacts.get((reference.artifact.invocation, reference.artifact.key)))
            )
        elif isinstance(reference, DecisionRef):
            comparisons.append((reference, decisions.get((reference.artifact.invocation, reference.artifact.key))))
        elif isinstance(reference, AbsenceRef):
            comparisons.append((reference, absences.get((reference.invocation, reference.query))))
        else:
            comparisons.append((reference, artifacts.get((reference.invocation, reference.key))))
    comparisons.extend((item.revision, revisions.get(item.effect)) for item in evidence.environment.state.revisions)
    if any(actual is not None and actual != expected for expected, actual in comparisons):
        return "stale"
    if any(actual is None for _, actual in comparisons):
        return "unknown"
    return "current"


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class _ProjectedPromise(PrivateValue):
    interface_outcome: str
    path: tuple[NodeId, ...]
    node: NodeId
    outcome: str
    promise: EvidencePromise

    def matches(self, requirement: ProtectionRequirement) -> bool:
        return (
            self.interface_outcome == requirement.outcome
            and self.promise.meaning == requirement.meaning
            and self.promise.subject_port == requirement.subject_port
            and self.promise.consumed_ports >= requirement.consumed_ports
        )


class _PromiseProjection:
    """Lift operation promises through the admitted interface identity bindings."""

    def __init__(self, execution: AdmittedExecutionPlan) -> None:
        self.workflow = execution.context.prepared.workflow

    def all(self) -> tuple[_ProjectedPromise, ...]:
        return tuple(self._body(self.workflow.workflow, ()))

    def _body(self, body: AdmittedWorkflow, path: tuple[NodeId, ...]) -> list[_ProjectedPromise]:
        projected: set[_ProjectedPromise] = set()
        nodes = tuple(body.nodes)
        children = {
            node.id: self._body(node.body, (*path, node.id)) for node in nodes if isinstance(node, SubgraphNode)
        }
        for outcomes in product(*(node.operation.outcomes for node in nodes)):
            assignment = {node.id: outcome for node, outcome in zip(nodes, outcomes, strict=True)}
            selected = _selected_nodes(frozenset(assignment), tuple(body.choices), assignment)
            sinks = selected - {
                edge.before for edge in body.sequence if edge.before in selected and edge.after in selected
            }
            mapping = next(
                item
                for item in body.outcome_bindings
                if item.source.node in sinks and assignment[item.source.node].name == item.source.outcome
            )
            assignment = {node: outcome for node, outcome in assignment.items() if node in selected}
            edges = self._edges(body, assignment)
            interface_outcome = next(
                item for item in body.interface.outcomes if item.name == mapping.destination.outcome
            )
            declared_items = frozenset(
                port
                for promise in interface_outcome.evidence
                for port in (promise.subject_port, *promise.consumed_ports)
                if isinstance(port, MapItemPort)
            )
            for node in nodes:
                if node.id not in selected:
                    continue
                outcome = assignment[node.id]
                if isinstance(node, SubgraphNode):
                    promises = [item for item in children[node.id] if item.interface_outcome == outcome.name]
                else:
                    promises = [
                        _ProjectedPromise(
                            interface_outcome=outcome.name,
                            path=path,
                            node=node.id,
                            outcome=outcome.name,
                            promise=promise,
                        )
                        for promise in outcome.evidence
                    ]
                inputs = {item.name for item in node.operation.inputs}
                for item in promises:
                    promise = item.promise
                    try:
                        subject = _project_evidence_port(
                            promise.subject_port,
                            node=node.id,
                            input_port=promise.subject_port in inputs,
                            edges=edges,
                            declared=declared_items,
                            assignment=assignment,
                            allow_output_alias=True,
                        )
                        consumed = frozenset(
                            _project_evidence_port(
                                port,
                                node=node.id,
                                input_port=True,
                                edges=edges,
                                declared=declared_items,
                                assignment=assignment,
                            )
                            for port in promise.consumed_ports
                        )
                    except ContractViolation:
                        continue
                    projected.add(
                        replace(
                            item,
                            interface_outcome=mapping.destination.outcome,
                            promise=replace(
                                promise,
                                subject_port=subject,
                                consumed_ports=consumed,
                            ),
                        )
                    )
        return list(projected)

    def _edges(self, body: AdmittedWorkflow, assignment: dict[NodeId, OutcomeSpec]) -> set[tuple[Vertex, Vertex]]:
        scope = next(item for item in self.workflow.scopes if item.workflow is body)
        replaced = {(item.member, item.item_input) for item in scope.maps if item.item_input is not None}
        edges: set[tuple[Vertex, Vertex]] = set()
        for binding in body.input_bindings:
            if (
                binding.destination.node not in assignment
                or (binding.destination.node, binding.destination.port) in replaced
            ):
                continue
            if isinstance(binding.source, WorkflowInputRef):
                source: Vertex = ("wi", None, binding.source.port)
            elif isinstance(binding.source, ContextInputRef):
                source = ("ci", None, binding.source.port)
            else:
                source = ("out", binding.source.node, binding.source.port)
            edges.add((source, ("in", binding.destination.node, binding.destination.port)))
        for binding in body.output_bindings:
            if not isinstance(binding.source, WorkflowInputRef) and (
                binding.source.node not in assignment
                or binding.source.port not in assignment[binding.source.node].produced_ports
            ):
                continue
            source = (
                ("wi", None, binding.source.port)
                if isinstance(binding.source, WorkflowInputRef)
                else ("out", binding.source.node, binding.source.port)
            )
            edges.add((source, ("wo", None, binding.destination.port)))
        for node in body.nodes:
            if node.id not in assignment:
                continue
            for dependency in node.operation.output_dependencies:
                if dependency.identity_input is not None and dependency.output in assignment[node.id].produced_ports:
                    edges.add((("in", node.id, dependency.identity_input), ("out", node.id, dependency.output)))
        return edges
