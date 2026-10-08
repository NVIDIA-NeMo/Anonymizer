# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure admission and authentication of execution-produced evidence."""

from __future__ import annotations

from dataclasses import dataclass, fields
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
)
from anonymizer.engine.graph_sdk.preparation import StateRevisionView
from anonymizer.engine.graph_sdk.records import AbsenceRef, CandidateRef, ConsumedRef, DecisionRef, EvidenceRef
from anonymizer.graph._values import ActivationKey, ArtifactRef, DatumId
from anonymizer.graph.workflow import CoverageAtom, EvidencePromise, NodeId, OperationNode, OperationSpec, OutcomeSpec

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

    def __init__(
        self,
        *,
        _key: object,
        execution: AdmittedExecutionPlan,
        productions: tuple[EvidenceProductionDecl, ...],
        limits: QualificationLimits,
        _resolved: tuple[_Production, ...],
    ) -> None:
        if _key is not _EVIDENCE_KEY:
            raise TypeError("qualification admission requires admit_qualification")
        for name, value in locals().copy().items():
            if name not in {"self", "_key"}:
                object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class VerifiedEvidence(PrivateValue):
    reference: EvidenceRef
    activation: ActivationKey
    node: NodeId
    outcome: str
    promise: EvidencePromise
    subject: CandidateRef
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
    if frozenset(productions) != frozenset(execution.assessment_productions):
        reject(
            EffectCode.MISSING
            if frozenset(productions) < frozenset(execution.assessment_productions)
            else EffectCode.UNSUPPORTED
        )
    resolved = tuple(_resolve_production(item, operations[item.node], limits) for item in productions)
    node_order = {slot.template: slot.index for slot in reversed(prepared.reservation_recipe)}
    resolved = tuple(
        sorted(resolved, key=lambda item: (node_order[item.declaration.node], item.outcome.name, item.promise.name))
    )
    return AdmittedQualification(
        _key=_EVIDENCE_KEY,
        execution=execution,
        productions=tuple(item.declaration for item in resolved),
        limits=limits,
        _resolved=resolved,
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
            len(result.ports) > min(limits.max_port_facts, execution.assessment_limits.max_port_facts)
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
        subject = self.port(fact, production.promise.subject_port)
        if evidence.target != target or subject.target != target:
            reject(EffectCode.FOREIGN_OWNER)
        if evidence.artifact != fact.evidence_artifact or subject.role != "candidate":
            reject(EffectCode.CONTRADICTORY)
        consumed: list[tuple[str, ConsumedRef]] = []
        input_names = {item.name for item in production.operation.inputs}
        for name in sorted(production.promise.consumed_ports):
            port = self.port(fact, name)
            if name not in input_names or port.target != target:
                reject(EffectCode.CONTRADICTORY)
            reference: ConsumedRef
            if port.role == "candidate":
                reference = CandidateRef(artifact=port.artifact, target=target)
            elif port.role == "decision":
                reference = DecisionRef(artifact=port.artifact)
            else:
                reference = port.artifact
            consumed.append((name, reference))
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
            subject=CandidateRef(artifact=subject.artifact, target=target),
            consumed_by_port=tuple(consumed),
            environment=fact.environment,
            finding=fact.finding,
            coverage=production.promise.coverage,
            _admitted=self.admitted,
            _result=self.result,
        )

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
    return tuple(sorted(verified, key=lambda item: (item.activation.occurrence, item.reference.artifact.key)))


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
    candidates = frozenset(item.candidate for item in result.final_outputs if item.candidate.artifact in selected)
    if len({item.target for item in candidates}) != len(candidates):
        reject(EffectCode.CONTRADICTORY)
    prepared = admitted.execution.context.prepared
    order = {slot.template: slot.index for slot in reversed(prepared.reservation_recipe)}
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
        (evidence.subject, candidates.get(evidence.subject.target)),
        (evidence.environment.configuration, configurations.get(evidence.node)),
    ]
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
