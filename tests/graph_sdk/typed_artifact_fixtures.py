# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One provider-free typed detection/filter/render graph for M1 witnesses."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from anonymizer.engine.detection.artifacts import ENTITY_SPANS_TYPE, EntitySpansValue
from anonymizer.engine.detection.postprocess import EntitySpan, filter_excluded_entity_spans
from anonymizer.engine.graph_sdk.capabilities import (
    ImplementationCapability,
    ImplementationRef,
    ImplementationSelection,
)
from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.data import DataGraph, DataLimits
from anonymizer.engine.graph_sdk.executor import (
    AdmittedExecutionPlan,
    AssessmentLimits,
    DecisionLimits,
    ExecutionImplementation,
    ExecutionLimits,
    ExecutionResult,
    ExecutionServices,
    ImplementationHandle,
    IngressLimits,
    LocalCompleted,
    OperationExecutionPolicy,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import BoundInput, PreparationConfiguration, StateRevisionView, prepare
from anonymizer.engine.graph_sdk.requests import AssociationInput, AssociationResult, PortArtifact, TextArtifactValue
from anonymizer.engine.replace.strategies import ReplacementEntry, apply_entity_spans
from anonymizer.graph.activation import ActivationLimits
from anonymizer.graph.workflow import (
    ArtifactType,
    DynamicLimits,
    DynamicScope,
    InputBinding,
    InputPort,
    NodeId,
    NodeInputRef,
    NodeOutcomeRef,
    NodeOutputRef,
    OperationNode,
    OutcomeBinding,
    OutputBinding,
    OutputDependency,
    OutputPort,
    SequenceEdge,
    WorkflowInputRef,
    WorkflowOutcomeRef,
    WorkflowOutputRef,
    admit_activation_workflow,
    admit_static_workflow,
)
from tests.graph_sdk.effects_admission_fixtures import _valid_runtime_rows, _ZeroClock
from tests.graph_sdk.test_preparation import _capability, _limits, _workflow

SOURCE = "Alice met Bob."
ENTITIES = (
    EntitySpan("a", "Alice", "first_name", 0, 5, 0.9, "detector"),
    EntitySpan("b", "Bob", "last_name", 10, 13, 0.8, "detector"),
)
TEXT_TYPE = ArtifactType(name="text", revision=1)


@dataclass
class SpanOperation:
    mode: str
    mutate: Callable[[EntitySpansValue], EntitySpansValue] | None = None

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        (item,) = request
        if self.mode == "alias":
            candidates = next(port.value for port in item.inputs if port.port == "spans")
            assert isinstance(candidates, EntitySpansValue)
            return self.completed(item, candidates, ENTITY_SPANS_TYPE)
        source = next(port for port in item.inputs if port.port == "source")
        assert source.artifact is not None and isinstance(source.value, TextArtifactValue)
        if self.mode == "detect":
            value = EntitySpansValue(source=source.artifact, entities=ENTITIES)
        else:
            candidates = next(port.value for port in item.inputs if port.port == "spans")
            assert isinstance(candidates, EntitySpansValue)
            if self.mode == "filter":
                value = EntitySpansValue(
                    source=candidates.source,
                    entities=tuple(filter_excluded_entity_spans(list(candidates.entities), ("last_name",))),
                )
            else:
                text, _ = apply_entity_spans(
                    source.value.text,
                    candidates.entities,
                    [ReplacementEntry("Alice", "first_name", "[name]")],
                    allow_value_fallback=False,
                )
                return self.completed(item, TextArtifactValue(text=text), TEXT_TYPE)
        if self.mutate is not None:
            value = self.mutate(value)
        return self.completed(item, value, ENTITY_SPANS_TYPE)

    @staticmethod
    def completed(
        item: AssociationInput, value: EntitySpansValue | TextArtifactValue, kind: ArtifactType
    ) -> LocalCompleted:
        return LocalCompleted(
            results=(
                AssociationResult(
                    association=item.association,
                    outcome="ok",
                    consumed_context_ports=frozenset(),
                    outputs=(PortArtifact(port="value", artifact_type=kind, artifact=None, value=value),),
                ),
            )
        )


def span_graph(
    *,
    mutate: Callable[[EntitySpansValue], EntitySpansValue] | None = None,
    ingress: IngressLimits = IngressLimits(),
    alias_filter: bool = False,
    span_type: ArtifactType = ENTITY_SPANS_TYPE,
    source_type: ArtifactType = TEXT_TYPE,
) -> tuple[AdmittedExecutionPlan, tuple[ImplementationCapability, ...], ExecutionServices, tuple[NodeId, ...]]:
    base, _, _ = _workflow(with_input=True)
    raw = base.workflow
    owner = raw.workflow
    nodes = tuple(NodeId.new(workflow=owner) for _ in range(3))
    operations = []
    for index, name in enumerate(("detect", "filter", "render")):
        inputs = (InputPort(name="source", artifact_type=source_type),)
        if index:
            inputs += (InputPort(name="spans", artifact_type=span_type),)
        if alias_filter and index == 1:
            inputs = inputs[1:]
        operations.append(
            replace(
                raw.interface,
                name=name,
                inputs=inputs,
                outputs=(OutputPort(name="value", artifact_type=TEXT_TYPE if index == 2 else span_type),),
                output_dependencies=(
                    OutputDependency(
                        output="value",
                        inputs=frozenset(p.name for p in inputs),
                        identity_input="spans" if alias_filter and index == 1 else None,
                    ),
                ),
                outcomes=(replace(raw.interface.outcomes[0], produced_ports=frozenset({"value"})),),
            )
        )
    interface = replace(
        operations[-1],
        name="typed-spans",
        inputs=operations[0].inputs,
        output_dependencies=(OutputDependency(output="value", inputs=frozenset({"source"}), identity_input=None),),
        outcomes=(
            replace(operations[-1].outcomes[0], ceiling=replace(operations[-1].outcomes[0].ceiling, max_activations=3)),
        ),
    )
    bindings = tuple(
        InputBinding(source=WorkflowInputRef(port="source"), destination=NodeInputRef(node=n, port="source"))
        for index, n in enumerate(nodes)
        if not (alias_filter and index == 1)
    )
    bindings += tuple(
        InputBinding(
            source=NodeOutputRef(node=nodes[i - 1], port="value"), destination=NodeInputRef(node=nodes[i], port="spans")
        )
        for i in (1, 2)
    )
    static = admit_static_workflow(
        workflow=owner,
        interface=interface,
        nodes=tuple(OperationNode(id=n, operation=o) for n, o in zip(nodes, operations, strict=True)),
        input_bindings=bindings,
        output_bindings=(
            OutputBinding(
                source=NodeOutputRef(node=nodes[-1], port="value"), destination=WorkflowOutputRef(port="value")
            ),
        ),
        outcome_bindings=(
            OutcomeBinding(
                source=NodeOutcomeRef(node=nodes[-1], outcome="ok"), destination=WorkflowOutcomeRef(outcome="ok")
            ),
        ),
        sequence=tuple(SequenceEdge(before=nodes[i - 1], after=nodes[i]) for i in (1, 2)),
        choices=(),
        protection=(),
        limits=replace(raw.limits, max_nodes=3, max_bindings=12, max_sequence_edges=2),
    )
    workflow = admit_activation_workflow(
        workflow=static,
        scopes=(DynamicScope(workflow=static, maps=(), loops=(), joins=()),),
        limits=DynamicLimits(
            max_maps=0,
            max_joins=0,
            max_loops=0,
            max_children_per_map=0,
            max_iterations_per_loop=0,
            max_dynamic_depth=1,
            max_activation_occurrences=3,
        ),
    )
    graph = DataGraph.new()
    graph, target = graph.add_text(SOURCE)
    data = graph.validate(
        targets=(target,),
        source_relations=(),
        contexts=(),
        dependencies=(),
        coherence=(),
        atomic=(),
        output_regions=(),
        limits=DataLimits(max_datums=1, max_targets=1, max_text_bytes=100, max_declarations=0, max_group_members=0),
    )
    capabilities = tuple(
        replace(_capability(workflow), operation=o, implementation=ImplementationRef(name=o.name, revision=1))
        for o in operations
    )
    prepared = prepare(
        data=data,
        workflow=workflow,
        capabilities=capabilities,
        activation_limits=ActivationLimits(max_events=40, max_entries=10, max_parent_depth=1),
        selections=tuple(
            ImplementationSelection(node=n, implementation=c.implementation, configuration=c.configuration)
            for n, c in zip(nodes, capabilities, strict=True)
        ),
        limits=_limits(capabilities=3),
        configuration=PreparationConfiguration(
            purpose="execution_only", required_protection_outcomes=frozenset(), hard_request_limit=None
        ),
        bound_inputs=(BoundInput(target=target, source=target, port="source", artifact_type=source_type),),
        state=StateRevisionView(revisions=frozenset()),
    )
    policies = tuple(
        OperationExecutionPolicy(
            node=n,
            kind="local",
            request=None,
            safe_detachment="forbidden",
            implementations=(
                ExecutionImplementation(
                    implementation=c.implementation, configuration=c.configuration, capability=c, request=None
                ),
            ),
            result_outcomes=frozenset({"ok"}),
            runtime_outcomes=_valid_runtime_rows("local", frozenset({"ok"})),
        )
        for n, c in zip(nodes, capabilities, strict=True)
    )
    admitted = admit_execution_plan(
        context=admit_context_plan(
            prepared=prepared, bound_context=None, adaptive_retrievals=(), context_capabilities=()
        ),
        capabilities=capabilities,
        policies=policies,
        decisions=(),
        assessment_productions=(),
        ingress_limits=ingress,
        assessment_limits=AssessmentLimits(
            max_productions=0,
            max_findings_per_production=0,
            max_finding_code_bytes=0,
            max_absence_queries=0,
            max_assessment_facts=0,
            max_port_facts=20,
            max_provenance_edges=20,
        ),
    )
    services = ExecutionServices(
        handles=tuple(
            ImplementationHandle(
                implementation=c.implementation,
                operation=c.operation,
                configuration=c.configuration,
                local=SpanOperation("alias" if alias_filter and i == 1 else o.name, mutate if i == 0 else None),
                transport=None,
                resource=None,
            )
            for i, (o, c) in enumerate(zip(operations, capabilities, strict=True))
        ),
        context_resources=(),
        limits=ExecutionLimits(
            max_local_in_flight=1,
            max_remote_outstanding=0,
            max_runtime_artifacts=10,
            max_runtime_artifact_bytes=1000,
            max_collection_items=10,
        ),
        decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
        clock=_ZeroClock(),
    )
    return admitted, capabilities, services, nodes


async def execute_spans(
    *,
    mutate: Callable[[EntitySpansValue], EntitySpansValue] | None = None,
    ingress: IngressLimits = IngressLimits(),
    alias_filter: bool = False,
) -> tuple[AdmittedExecutionPlan, ExecutionResult, tuple[NodeId, ...]]:
    admitted, capabilities, services, nodes = span_graph(mutate=mutate, ingress=ingress, alias_filter=alias_filter)
    result = await (await start_execution(admitted=admitted, capabilities=capabilities, services=services)).wait()
    return admitted, result, nodes
