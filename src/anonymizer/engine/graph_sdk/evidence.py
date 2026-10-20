# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure admission and authentication of execution-produced evidence."""

from __future__ import annotations

from collections import ChainMap
from collections.abc import Mapping, MutableMapping
from collections.abc import Set as AbstractSet
from dataclasses import InitVar, dataclass, field, fields, replace
from itertools import product
from types import MappingProxyType
from typing import Literal, TypeVar, cast

from anonymizer.engine.graph_sdk._activation_values import ActivationEntry, ActivationState
from anonymizer.engine.graph_sdk._effect_values import (
    EffectCode,
    PrivateValue,
    reject,
    require_count,
    require_instance,
    require_owner,
)
from anonymizer.engine.graph_sdk._execution_state import _ExecutionFacts
from anonymizer.engine.graph_sdk._execution_topology import _lexical_path
from anonymizer.engine.graph_sdk._graph_values import ActivationKey, ArtifactRef, ContractViolation, DatumId
from anonymizer.engine.graph_sdk._workflow_composition import (
    Vertex,
    _project_consumed_ports,
    _project_evidence_port,
    _selected_nodes,
)
from anonymizer.engine.graph_sdk._workflow_values import (
    ContextInputRef,
    CoverageAtom,
    EvidencePromise,
    Graph,
    MapItemPort,
    NodeId,
    OperationNode,
    OperationSpec,
    OutcomeSpec,
    ProtectionRequirement,
    SubgraphNode,
    WorkflowInputRef,
)
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
    ProvenanceKey,
)
from anonymizer.engine.graph_sdk.preparation import PreparedPlan
from anonymizer.engine.graph_sdk.records import (
    CandidateRef,
    ConsumedRef,
    EvidenceRef,
    TerminalFact,
)
from anonymizer.engine.graph_sdk.requests import TextCollectionValue

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
    max_fixed_point_steps: int
    max_revision_entries: int
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


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class AdmittedQualification(PrivateValue):
    _key: InitVar[object]
    execution: AdmittedExecutionPlan
    productions: tuple[EvidenceProductionDecl, ...]
    limits: QualificationLimits
    _resolved: tuple[_Production, ...]
    _projections: tuple[_ProjectedPromise, ...]

    def __post_init__(self, _key: object) -> None:
        require_owner(_key, _EVIDENCE_KEY)

    def _subject_outputs(self, outcome: str) -> frozenset[str]:
        """Return final ports that carry this outcome's protection subjects."""
        workflow = self.execution.context.prepared.workflow
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


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class VerifiedEvidence(PrivateValue):
    _key: InitVar[object]
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
    _admitted: AdmittedQualification = field(hash=False)
    _result: ExecutionResult = field(hash=False)

    def __post_init__(self, _key: object) -> None:
        require_owner(_key, _EVIDENCE_KEY)


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
        for body in prepared.workflow.bodies()
        for node in body.operations
        if isinstance(node, OperationNode)
    }
    owners = {body.workflow for body in prepared.workflow.bodies()}
    if any(item.node.workflow not in owners for item in productions):
        reject(EffectCode.FOREIGN_OWNER)
    keys = [(item.node, item.outcome, item.promise) for item in productions]
    if len(keys) != len(set(keys)):
        reject(EffectCode.DUPLICATE)
    if any(item.node not in operations for item in productions):
        reject(EffectCode.MISSING)
    resolved = tuple(_resolve_production(item, operations[item.node], limits) for item in productions)
    # Execution-only plans skip protection eligibility. Only a declared map-item
    # promise can lift a production's subject, so lift just those to keep the
    # assessment verifiable without projecting every other plan.
    lifted = productions and any(
        isinstance(port, MapItemPort)
        for body in prepared.workflow.bodies()
        for outcome in body.interface.outcomes
        for promise in outcome.evidence
        for port in (promise.subject_port, *promise.consumed_ports)
    )
    projections = (
        _PromiseProjection(execution).all() if prepared.configuration.purpose != "execution_only" or lifted else ()
    )
    requirements = tuple(
        item
        for item in prepared.workflow.protection_requirements
        if item.outcome in prepared.configuration.required_protection_outcomes
    )
    required = {
        (item.node, item.outcome, item.promise.name)
        for item in projections
        if any(item.matches(requirement) for requirement in requirements)
    }
    if prepared.configuration.purpose == "execution_only" and productions:
        required = {(item.node, item.outcome, item.promise) for item in execution.assessment_productions}
    observed = set(keys)
    if required - observed:
        reject(EffectCode.MISSING)
    if observed - required or any(item not in execution.assessment_productions for item in productions):
        reject(EffectCode.UNSUPPORTED)
    node_order = {
        node.id: index
        for index, node in enumerate(node for body in prepared.workflow.bodies() for node in body.operations)
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


_K = TypeVar("_K")
_V = TypeVar("_V")


def _sealed(index: ChainMap[_K, _V]) -> ChainMap[_K, _V]:
    """One read-only layer: any later write through the ChainMap raises TypeError."""
    return ChainMap(cast(MutableMapping[_K, _V], MappingProxyType(dict(index))))


@dataclass(slots=True, kw_only=True)
class _RunFacts(_ExecutionFacts):
    """Bounded occurrence indexes for authenticating one result's assessments."""

    admitted: AdmittedQualification
    result: ExecutionResult
    entries: Mapping[ActivationKey, ActivationEntry] = field(default_factory=dict)
    owners: Mapping[ActivationKey, DatumId] = field(default_factory=dict)
    shared: AbstractSet[ActivationKey] = field(default_factory=frozenset)
    terminals: Mapping[ActivationKey, TerminalFact] = field(default_factory=dict)
    input_parents: Mapping[tuple[DatumId, ActivationKey, str], ProvenanceKey] = field(default_factory=dict)

    @classmethod
    def from_result(cls, admitted: AdmittedQualification, result: ExecutionResult) -> _RunFacts:
        limits = admitted.limits
        execution = admitted.execution
        prepared = execution.context.prepared
        bound = execution.context.bound_context
        if bound is not None:
            source_limit = len(prepared.target_occurrences) * sum(
                isinstance(binding.source, ContextInputRef) for binding in prepared.workflow.input_bindings
            )
            if max(len(bound.artifacts), len(bound.receipt.artifacts)) > limits.max_port_facts or len(
                bound.receipt.sources
            ) > max(source_limit, limits.max_port_facts):
                reject(EffectCode.LIMIT_EXCEEDED)
        if (
            max(len(result.ports), len(result._input_parents), len(result._passthrough_parents))
            > min(limits.max_port_facts, execution.assessment_limits.max_port_facts)
            or len(result.provenance) > limits.max_port_facts
            or len(result.record.memberships) > prepared.limits.max_total_activation_slots + 1
            or len(result.record.terminals) > prepared.limits.max_total_activation_slots
            or len(result.assessments) > execution.assessment_limits.max_assessment_facts
            or sum(len(item.parents) for item in result.provenance)
            > min(limits.max_provenance_edges, execution.assessment_limits.max_provenance_edges)
        ):
            reject(EffectCode.LIMIT_EXCEEDED)
        record = result.record
        # The result is owner-keyed and its record was built from this same admitted plan, so plan, graph and
        # targets are those of ``prepared`` whenever the identity holds.
        if result._execution is not execution:
            reject(EffectCode.FOREIGN_OWNER)
        if any(reference.invocation != record.invocation for reference, _ in result.artifacts):
            reject(EffectCode.FOREIGN_OWNER)
        if any(
            terminal.activation.invocation != record.invocation
            or (terminal.attempt is not None and terminal.attempt.activation != terminal.activation)
            for terminal in record.terminals
        ):
            reject(EffectCode.FOREIGN_OWNER)
        if bound is not None:
            if bound.receipt.data is not prepared.data or bound.receipt.workflow is not prepared.workflow:
                reject(EffectCode.FOREIGN_OWNER)
            bound._validate_inventory(source_mismatch=EffectCode.FOREIGN_OWNER)
            bound._validate_artifact_joins(source_mismatch=EffectCode.FOREIGN_OWNER)
        # Every retained value passed ingress (callback outputs, bound sources) or was built by the executor
        # from validated data; the executor's own inventory is not re-inspected or re-joined here.
        retained = dict(result.artifacts)
        provenance = {fact.key: fact for fact in result.provenance}
        for fact in provenance.values():
            if fact.artifact.invocation != record.invocation or fact.key.target not in record.targets:
                reject(EffectCode.FOREIGN_OWNER)
        ports: dict[tuple[ActivationKey, str], ExecutionPortFact] = {}
        target_offsets = {item.target: item.occurrence_offset for item in prepared.target_occurrences}
        for fact in result.ports:
            if fact.activation.invocation != record.invocation or fact.target not in record.targets:
                reject(EffectCode.FOREIGN_OWNER)
            offset = target_offsets[fact.target]
            width = len(prepared.reservation_recipe)
            if (
                0 <= fact.activation.occurrence < width * len(target_offsets)
                and not offset <= fact.activation.occurrence < offset + width
            ):
                reject(EffectCode.FOREIGN_OWNER)
            ports[fact.activation, fact.port] = fact
        facts = cls(
            admitted=admitted,
            result=result,
            ports=ChainMap(ports),
            values=ChainMap(retained),
        )
        for fact in provenance.values():
            facts.add_provenance(fact)
        facts._index_states()
        facts.terminals = MappingProxyType({item.activation: item for item in record.terminals})
        input_parents: dict[tuple[DatumId, ActivationKey, str], ProvenanceKey] = {}
        for rows, index in (
            (result._input_parents, input_parents),
            (result._passthrough_parents, facts.passthrough_parents),
        ):
            for target, activation, port, parent in rows:
                index[target, activation, port] = parent
        facts.input_parents = MappingProxyType(input_parents)
        facts._seal()
        return facts

    def _seal(self) -> None:
        """Freeze the inherited indexes: a revision view may be reused by any number of qualifications."""
        self.values = _sealed(self.values)
        self.provenance = _sealed(self.provenance)
        self.ports = _sealed(self.ports)
        self.assessments = _sealed(self.assessments)
        self.passthrough_parents = _sealed(self.passthrough_parents)
        self._sources = _sealed(self._sources)

    @property
    def prepared(self) -> PreparedPlan:
        return self.admitted.execution.context.prepared

    def _index_states(self) -> None:
        record = self.result.record
        execution = self.admitted.execution
        invocation = self.result.invocation_state
        if (invocation is None) != (execution.invocation_workflow is None):
            reject(EffectCode.MISSING)
        if len(self.result.states) != len(self.admitted.execution.context.prepared.target_occurrences):
            reject(EffectCode.MISSING)
        states: list[tuple[DatumId | None, ActivationState, Graph, int]] = [
            (target, state, execution.target_workflow, occurrence.occurrence_offset)
            for target, state, occurrence in zip(
                (item.target for item in self.prepared.target_occurrences),
                self.result.states,
                self.prepared.target_occurrences,
                strict=True,
            )
        ]
        if invocation is not None:
            assert execution.invocation_workflow is not None
            states.append((None, invocation, execution.invocation_workflow, self.prepared.invocation_offset))
        entries: dict[ActivationKey, ActivationEntry] = {}
        owners: dict[ActivationKey, DatumId] = {}
        shared: set[ActivationKey] = set()
        for target, state, workflow, offset in states:
            if state.invocation != record.invocation or state.workflow is not workflow:
                reject(EffectCode.FOREIGN_OWNER)
            templates = {node.id for scope in state.workflow.bodies() for node in scope.operations}
            expected = {
                offset + slot.index: slot for slot in self.prepared.reservation_recipe if slot.template in templates
            }
            for entry in state.entries:
                if entry.activation in entries:
                    reject(EffectCode.DUPLICATE)
                slot = expected.get(entry.activation.occurrence)
                if slot is None or entry.template != slot.template:
                    reject(EffectCode.FOREIGN_OWNER)
                entries[entry.activation] = entry
                if target is None:
                    shared.add(entry.activation)
                else:
                    owners[entry.activation] = target
        self.entries, self.owners, self.shared = MappingProxyType(entries), MappingProxyType(owners), frozenset(shared)

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
        # verify_submissions admitted only facts retained by this result (identity).
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
            else:
                reference = port.artifact
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
        self._verify_environment(fact)
        if fact.finding not in production.declaration.supported_findings:
            reject(EffectCode.UNSUPPORTED)
        return VerifiedEvidence(
            _key=_EVIDENCE_KEY,
            reference=EvidenceRef(
                artifact=evidence.artifact,
                consumed=frozenset(reference for _, reference in consumed),
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
        key = self.input_parents.get((input_fact.target, fact.activation, port))
        if not isinstance(key, MapItemKey):
            return None
        entries = self.entries
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
        if _lexical_path(key.expander, entries, self.admitted.execution.nodes) != endpoint.path:
            return None
        provenance = self.provenance[key]
        expected = OperationOutputKey(activation=key.expander, target=key.target, port=endpoint.membership_port)
        if provenance.parents != frozenset({expected}):
            reject(EffectCode.CONTRADICTORY)
        publication = self.provenance.get(expected)
        if publication is None:
            reject(EffectCode.CONTRADICTORY)
        collection = self.values.get(publication.artifact)
        if not isinstance(collection, TextCollectionValue):
            reject(EffectCode.CONTRADICTORY)
        expansions = [
            expansion
            for state in self.result.states
            for expansion in state.expansions
            if expansion.parent == key.expander
        ]
        if len(expansions) != 1 or key.member not in expansions[0].members:
            reject(EffectCode.CONTRADICTORY)
        return MapItemSubjectRef(artifact=input_fact.artifact, producer=key)

    def _verify_activation(self, fact: ExecutionAssessmentFact) -> DatumId:
        entry = self.entries.get(fact.activation)
        target = self.owners.get(fact.activation)
        terminal = self.terminals.get(fact.activation)
        if entry is None or target is None or terminal is None:
            reject(EffectCode.MISSING)
        if (
            entry.status != "success"
            or entry.outcome != fact.outcome
            or terminal.category != "success"
            or terminal.attempt is None
            or terminal.structural
        ):
            reject(EffectCode.CONTRADICTORY)
        return target

    def verify_submissions(self, submissions: tuple[AssessmentSubmission, ...]) -> tuple[VerifiedEvidence, ...]:
        admitted, result = self.admitted, self.result
        require_instance(submissions, tuple)
        if len(submissions) > min(admitted.limits.max_submissions, admitted.limits.max_verified_evidence):
            reject(EffectCode.LIMIT_EXCEEDED)
        if any(not isinstance(item, AssessmentSubmission) for item in submissions):
            reject(EffectCode.INVALID_TYPE)
        if any(not any(item.fact is fact for fact in result.assessments) for item in submissions):
            reject(EffectCode.FOREIGN_OWNER)
        identities = [item.fact.evidence_artifact for item in submissions]
        if len(identities) != len(set(identities)):
            reject(EffectCode.DUPLICATE)
        verified = tuple(self.verify(item.fact) for item in submissions)
        prepared = admitted.execution.context.prepared
        target_order = {item.target: index for index, item in enumerate(prepared.target_occurrences)}
        node_order = {
            node.id: index
            for index, node in enumerate(node for body in prepared.workflow.bodies() for node in body.operations)
        }
        productions = {
            (item.declaration.node, item.outcome.name, item.promise.name): item for item in admitted._resolved
        }

        def canonical_order(item: VerifiedEvidence) -> tuple[int, int, int, int, int, int]:
            production = productions[item.node, item.outcome, item.promise.name]
            port = production.declaration.evidence_port
            target = self.ports[item.activation, port].target
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

    def _verify_environment(self, fact: ExecutionAssessmentFact) -> None:
        policy = next(item for item in self.admitted.execution.policies if item.node == fact.node)
        if fact.environment.configuration not in {item.configuration for item in policy.implementations}:
            reject(EffectCode.CONTRADICTORY)


def verify_evidence(
    *, admitted: AdmittedQualification, result: ExecutionResult, submissions: tuple[AssessmentSubmission, ...]
) -> tuple[VerifiedEvidence, ...]:
    """Authenticate retained assessment objects and derive their exact dependencies."""
    require_instance(admitted, AdmittedQualification)
    require_instance(result, ExecutionResult)
    return _RunFacts.from_result(admitted, result).verify_submissions(submissions)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class EvidenceRevisionView(PrivateValue):
    _key: InitVar[object]
    candidates: frozenset[CandidateRef]
    artifacts: frozenset[ArtifactRef]
    configurations: tuple[tuple[NodeId, FrozenConfig], ...]
    _admitted: AdmittedQualification
    _result: ExecutionResult
    # The one index of this (admitted, result) pair; qualify reuses it. Not part of equality, hash or repr.
    _facts: _RunFacts = field(compare=False, hash=False)

    def __post_init__(self, _key: object) -> None:
        require_owner(_key, _EVIDENCE_KEY)


def evidence_revision_view(
    *,
    admitted: AdmittedQualification,
    result: ExecutionResult,
    artifacts: tuple[ArtifactRef, ...],
    configurations: tuple[tuple[NodeId, FrozenConfig], ...],
) -> EvidenceRevisionView:
    """Bind explicit current revisions to one admitted execution result."""
    require_instance(admitted, AdmittedQualification)
    require_instance(result, ExecutionResult)
    require_instance(artifacts, tuple)
    require_instance(configurations, tuple)
    if len(artifacts) + len(configurations) > admitted.limits.max_revision_entries:
        reject(EffectCode.LIMIT_EXCEEDED)
    if any(not isinstance(item, ArtifactRef) for item in artifacts):
        reject(EffectCode.INVALID_TYPE)
    if any(
        not isinstance(item, tuple)
        or len(item) != 2
        or not isinstance(item[0], NodeId)
        or not isinstance(item[1], FrozenConfig)
        for item in configurations
    ):
        reject(EffectCode.INVALID_TYPE)
    facts = _RunFacts.from_result(admitted, result)
    _validate_current_revisions(admitted, result, artifacts, configurations)
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
        for index, node in enumerate(node for body in prepared.workflow.bodies() for node in body.operations)
    }
    return EvidenceRevisionView(
        _key=_EVIDENCE_KEY,
        candidates=candidates,
        artifacts=selected,
        configurations=tuple(sorted(configurations, key=lambda item: order[item[0]])),
        _admitted=admitted,
        _result=result,
        _facts=facts,
    )


def _validate_current_revisions(
    admitted: AdmittedQualification,
    result: ExecutionResult,
    artifacts: tuple[ArtifactRef, ...],
    configurations: tuple[tuple[NodeId, FrozenConfig], ...],
) -> None:
    prepared = admitted.execution.context.prepared
    nodes = {item.node for item in prepared.implementations}
    if any(item.invocation != result.record.invocation for item in artifacts) or any(
        node.workflow not in {item.workflow for item in nodes} for node, _ in configurations
    ):
        reject(EffectCode.FOREIGN_OWNER)
    keys = (
        [(item.invocation, item.key) for item in artifacts],
        [node for node, _ in configurations],
    )
    if any(len(group) != len(set(group)) for group in keys):
        reject(EffectCode.DUPLICATE)
    if any(node not in nodes for node, _ in configurations) or any(
        item not in result.record.artifacts for item in artifacts
    ):
        reject(EffectCode.MISSING)
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
    configurations = dict(current.configurations)
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
        artifact = evidence.subject.artifact
        comparisons.append((artifact, artifacts.get((artifact.invocation, artifact.key))))
    for reference in evidence.reference.consumed:
        if isinstance(reference, CandidateRef):
            comparisons.append((reference, candidates.get(reference.target)))
            comparisons.append(
                (reference.artifact, artifacts.get((reference.artifact.invocation, reference.artifact.key)))
            )
        else:
            comparisons.append((reference, artifacts.get((reference.invocation, reference.key))))
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
        return tuple(self._body(self.workflow, ()))

    def _body(self, body: Graph, path: tuple[NodeId, ...]) -> list[_ProjectedPromise]:
        projected: set[_ProjectedPromise] = set()
        nodes = tuple(body.operations)
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
            dependency_edges = self._edges(body, assignment, dependencies=True)
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
                        consumed = _project_consumed_ports(
                            promise.consumed_ports,
                            node=node.id,
                            identity_edges=edges,
                            dependency_edges=dependency_edges,
                            declared=declared_items,
                            assignment=assignment,
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

    def _edges(
        self, body: Graph, assignment: dict[NodeId, OutcomeSpec], *, dependencies: bool = False
    ) -> set[tuple[Vertex, Vertex]]:
        scope = body
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
        for node in body.operations:
            if node.id not in assignment:
                continue
            for dependency in node.operation.output_dependencies:
                if dependency.output not in assignment[node.id].produced_ports:
                    continue
                if dependencies:
                    edges.update(
                        (("in", node.id, port), ("out", node.id, dependency.output)) for port in dependency.inputs
                    )
                elif dependency.identity_input is not None:
                    edges.add((("in", node.id, dependency.identity_input), ("out", node.id, dependency.output)))
        return edges
