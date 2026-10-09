# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-node real execution fixture for retained latest-binding obligations."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from anonymizer.engine.graph_sdk.binding import start_initial_binding
from anonymizer.engine.graph_sdk.capabilities import ImplementationSelection
from anonymizer.engine.graph_sdk.context import (
    BindingLimits,
    BindingResult,
    ContextMaterialization,
    ContextResource,
    ContextSelector,
    ContextSourceCapability,
    ContextSourceRef,
    InitialContextDecl,
    RetrievalBounds,
    admit_context_plan,
)
from anonymizer.engine.graph_sdk.data import DataGraph, DataLimits
from anonymizer.engine.graph_sdk.executor import (
    AssessmentLimits,
    DecisionLimits,
    ExecutionImplementation,
    ExecutionLimits,
    ExecutionResult,
    ExecutionServices,
    ImplementationHandle,
    LocalCompleted,
    OperationExecutionPolicy,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import BoundInput, PreparationConfiguration, StateRevisionView, prepare
from anonymizer.engine.graph_sdk.requests import AssociationInput, AssociationResult, PortArtifact, TextArtifactValue
from anonymizer.graph.activation import ActivationLimits
from anonymizer.graph.workflow import (
    AdmittedActivationWorkflow,
    ArtifactType,
    ContextInputRef,
    ContextUse,
    DynamicLimits,
    DynamicScope,
    InputBinding,
    InputPort,
    NodeId,
    NodeInputRef,
    NodeOutcomeRef,
    NodeOutputRef,
    OperationNode,
    OperationSpec,
    OutcomeBinding,
    OutputBinding,
    OutputDependency,
    OutputPort,
    SequenceEdge,
    WorkflowId,
    WorkflowInputRef,
    WorkflowLimits,
    WorkflowOutcomeRef,
    WorkflowOutputRef,
    admit_activation_workflow,
    admit_static_workflow,
)
from tests.graph_sdk.effects_provider_fixtures import _CaseProvider
from tests.graph_sdk.effects_request_fixtures import _policy
from tests.graph_sdk.test_dynamic_executor import _capability, _outcome, _rows
from tests.graph_sdk.test_local_executor import _Clock
from tests.graph_sdk.test_preparation import _limits


@dataclass
class _Publisher:
    artifact_type: ArtifactType
    value: str
    consumed: frozenset[str]
    expected_inputs: dict[str, str]
    calls: int = 0
    associations: list[AssociationInput] = field(default_factory=list)

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        self.calls += 1
        (item,) = request
        self.associations.append(item)
        actual_inputs = {}
        for port in item.inputs:
            assert port.artifact_type == self.artifact_type
            assert port.artifact is not None
            assert isinstance(port.value, TextArtifactValue)
            actual_inputs[port.port] = port.value.text
        assert len(actual_inputs) == len(item.inputs)
        assert actual_inputs == self.expected_inputs
        return LocalCompleted(
            results=(
                AssociationResult(
                    association=item.association,
                    outcome="ok",
                    outputs=(
                        PortArtifact(
                            port="result",
                            artifact_type=self.artifact_type,
                            artifact=None,
                            value=TextArtifactValue(text=self.value),
                        ),
                    ),
                    consumed_context_ports=self.consumed,
                ),
            )
        )


def _workflow(
    rollback: bool,
) -> tuple[AdmittedActivationWorkflow, dict[str, NodeId], ArtifactType, tuple[OperationSpec, OperationSpec]]:
    owner = WorkflowId.new()
    nodes = {name: NodeId.new(workflow=owner) for name in ("N0", "N1")}
    text_type = ArtifactType(name="text", revision=1)
    required_context = ContextUse(port="context", meaning="source", capture="whole_artifact")
    optional_context = ContextUse(port="optional_context", meaning="source", capture="whole_artifact")
    first = OperationSpec(
        name="N0",
        inputs=(InputPort(name="context", artifact_type=text_type),),
        outputs=(OutputPort(name="result", artifact_type=text_type),),
        output_dependencies=(OutputDependency(output="result", inputs=frozenset({"context"}), identity_input=None),),
        outcomes=(
            replace(
                _outcome("ok", produced=frozenset({"result"}), max_output_bytes=16),
                context=frozenset({required_context}),
            ),
        ),
    )
    second = OperationSpec(
        name="N1",
        inputs=(InputPort(name="right" if rollback else "optional_context", artifact_type=text_type),),
        outputs=(OutputPort(name="result", artifact_type=text_type),) if rollback else (),
        output_dependencies=(OutputDependency(output="result", inputs=frozenset({"right"}), identity_input=None),)
        if rollback
        else (),
        outcomes=(
            replace(
                _outcome(
                    "ok",
                    produced=frozenset({"result"}) if rollback else frozenset(),
                    max_output_bytes=16 if rollback else 0,
                ),
                context=frozenset() if rollback else frozenset({optional_context}),
            ),
        ),
    )
    operations = (first, second)
    interface_ports = ("left", "right", "context") if rollback else ("context", "optional_context")
    interface = OperationSpec(
        name="root",
        inputs=tuple(InputPort(name=name, artifact_type=text_type) for name in interface_ports),
        outputs=(OutputPort(name="result", artifact_type=text_type),),
        output_dependencies=(
            OutputDependency(
                output="result", inputs=frozenset({"right" if rollback else "context"}), identity_input=None
            ),
        ),
        outcomes=(
            replace(
                _outcome("ok", produced=frozenset({"result"}), max_activations=2, max_output_bytes=32),
                context=frozenset({required_context}) if rollback else frozenset({required_context, optional_context}),
            ),
        ),
    )
    input_bindings = [
        InputBinding(source=ContextInputRef(port="context"), destination=NodeInputRef(node=nodes["N0"], port="context"))
    ]
    if rollback:
        input_bindings.extend(
            InputBinding(source=WorkflowInputRef(port=port), destination=NodeInputRef(node=nodes[name], port=port))
            for port, name in (("right", "N1"),)
        )
    else:
        input_bindings.append(
            InputBinding(
                source=ContextInputRef(port="optional_context"),
                destination=NodeInputRef(node=nodes["N1"], port="optional_context"),
            )
        )
    final_node = nodes["N1" if rollback else "N0"]
    static = admit_static_workflow(
        workflow=owner,
        interface=interface,
        nodes=tuple(
            OperationNode(id=node, operation=operation)
            for node, operation in zip(nodes.values(), operations, strict=True)
        ),
        input_bindings=tuple(input_bindings),
        output_bindings=(
            OutputBinding(
                source=NodeOutputRef(node=final_node, port="result"), destination=WorkflowOutputRef(port="result")
            ),
        ),
        outcome_bindings=(
            OutcomeBinding(
                source=NodeOutcomeRef(node=final_node, outcome="ok"), destination=WorkflowOutcomeRef(outcome="ok")
            ),
        ),
        sequence=(SequenceEdge(before=nodes["N0" if rollback else "N1"], after=final_node),),
        choices=(),
        protection=(),
        limits=WorkflowLimits(
            max_nodes=2,
            max_bindings=6,
            max_sequence_edges=1,
            max_choices=0,
            max_branch_members=0,
            max_subgraph_depth=1,
            max_choice_states=1,
        ),
    )
    workflow = admit_activation_workflow(
        workflow=static,
        scopes=(DynamicScope(workflow=static, maps=(), joins=(), loops=()),),
        limits=DynamicLimits(
            max_maps=0,
            max_joins=0,
            max_loops=0,
            max_children_per_map=0,
            max_iterations_per_loop=0,
            max_dynamic_depth=1,
            max_activation_occurrences=2,
        ),
    )
    return workflow, nodes, text_type, operations


async def _execute_case(
    case: dict[str, Any],
) -> tuple[ExecutionResult, BindingResult, dict[str, NodeId], tuple[_Publisher, _Publisher], _CaseProvider]:
    raw = case["declaration"]
    rollback = bool(raw.get("root_publications"))
    workflow, nodes, text_type, operations = _workflow(rollback)
    graph = DataGraph.new()
    graph, target = graph.add_text("T")
    root_sources = {}
    for event in case["events"]:
        if event["kind"] == "root_artifact":
            graph, source = graph.add_text(event["value"])
            root_sources[event["port"]] = source
    data = graph.validate(
        targets=(target,),
        source_relations=(),
        contexts=(),
        dependencies=(),
        coherence=(),
        atomic=(),
        output_regions=(),
        limits=DataLimits(max_datums=3, max_targets=1, max_text_bytes=3, max_declarations=0, max_group_members=1),
    )
    source = ContextSourceRef(name="S0", revision=1)
    policy = _policy(raw["policies"]["P0"])
    association_names = {}
    request_names = {}
    provider = _CaseProvider(
        events=[event for event in case["events"] if event["kind"] in {"materialize_result", "source_failure"}],
        sources={"S0": source},
        association_names=association_names,
        request_names=request_names,
        fallback_associations=list(raw["binding_declarations"]),
        default_source=source,
    )
    capability = ContextSourceCapability(
        source=source,
        artifact_type=text_type,
        uses=frozenset({"initial_binding"}),
        execution="async",
        resource_owner="sdk",
        cancellation="cooperative_ack",
        settlement="explicit_ack",
        usage="exact",
        request=policy,
        safe_detachment="forbidden",
    )
    declarations = tuple(
        InitialContextDecl(
            target=target,
            node=nodes[spec["node"]],
            port=spec["port"],
            artifact_type=text_type,
            source=source,
            selector=ContextSelector(fields=()),
            requirement=raw.get("binding_requirements", {}).get(spec["declaration"], "required"),
            bounds=RetrievalBounds(
                max_items=spec["max_items"], max_bytes=spec["max_bytes"], max_requests=policy.max_attempts
            ),
            materialization=ContextMaterialization(kind="single", item_type=text_type),
            version_selection=spec["version_selection"],
        )
        for spec in raw["materializations"]
    )
    bound = await (
        await start_initial_binding(
            data=data,
            workflow=workflow,
            declarations=declarations,
            capabilities=(capability,),
            resources=(ContextResource(source=source, capability=capability, lease=None, factory=lambda: provider),),
            limits=BindingLimits(
                max_declarations=len(declarations),
                max_sources=1,
                max_capabilities=1,
                max_selector_fields=0,
                max_selector_bytes=0,
                max_items=raw["binding_limits"]["max_items"],
                max_bytes=raw["binding_limits"]["max_bytes"],
                max_requests=raw["hard_limit"],
                max_resources=1,
            ),
        )
    ).wait()
    assert bound.context is not None
    capabilities = tuple(_capability(operation, index) for index, operation in enumerate(operations))
    prepared = prepare(
        data=data,
        workflow=workflow,
        activation_limits=ActivationLimits(max_events=16, max_entries=2, max_parent_depth=1),
        bound_inputs=tuple(
            BoundInput(target=target, source=source, port=port, artifact_type=text_type)
            for port, source in root_sources.items()
        ),
        configuration=PreparationConfiguration(
            purpose="execution_only", required_protection_outcomes=frozenset(), hard_request_limit=None
        ),
        state=StateRevisionView(revisions=frozenset()),
        selections=tuple(
            ImplementationSelection(node=node, implementation=cap.implementation, configuration=cap.configuration)
            for node, cap in zip(nodes.values(), capabilities, strict=True)
        ),
        capabilities=capabilities,
        limits=_limits(capabilities=2, slots=2),
    )
    admitted = admit_execution_plan(
        context=admit_context_plan(
            prepared=prepared, bound_context=bound.context, adaptive_retrievals=(), context_capabilities=()
        ),
        capabilities=capabilities,
        policies=tuple(
            OperationExecutionPolicy(
                node=node,
                kind="local",
                request=None,
                safe_detachment="forbidden",
                implementations=(
                    ExecutionImplementation(
                        implementation=cap.implementation, configuration=cap.configuration, capability=cap, request=None
                    ),
                ),
                result_outcomes=frozenset({"ok"}),
                runtime_outcomes=_rows(frozenset({"ok"})),
            )
            for node, cap in zip(nodes.values(), capabilities, strict=True)
        ),
        decisions=(),
        assessment_productions=(),
        assessment_limits=AssessmentLimits(
            max_productions=0,
            max_findings_per_production=0,
            max_finding_code_bytes=0,
            max_absence_queries=0,
            max_assessment_facts=0,
            max_port_facts=16,
            max_provenance_edges=raw["materialization_limits"]["max_provenance_edges"],
        ),
    )
    values = {
        event["node"]: event["value"]
        for event in case["events"]
        if event["kind"] in {"operation_publish", "root_operation_publish"}
    }
    retrieved = next(event["items"] for event in case["events"] if event["kind"] == "materialize_result")
    selected = max(retrieved, key=lambda item: item["version"])
    callbacks = (
        _Publisher(text_type, values["N0"], frozenset({"context"}), {"context": selected["value"]}),
        _Publisher(
            text_type,
            values.get("N1", "unused"),
            frozenset() if rollback else frozenset({"optional_context"}),
            {
                "right": next(
                    event["value"]
                    for event in case["events"]
                    if event["kind"] == "root_artifact" and event["port"] == "right"
                )
            }
            if rollback
            else {},
        ),
    )
    result = await (
        await start_execution(
            admitted=admitted,
            capabilities=capabilities,
            services=ExecutionServices(
                handles=tuple(
                    ImplementationHandle(
                        implementation=cap.implementation,
                        operation=cap.operation,
                        configuration=cap.configuration,
                        local=callback,
                        transport=None,
                        resource=None,
                    )
                    for cap, callback in zip(capabilities, callbacks, strict=True)
                ),
                context_resources=(),
                limits=ExecutionLimits(
                    max_local_in_flight=1,
                    max_remote_outstanding=0,
                    max_runtime_artifacts=raw["materialization_limits"]["max_artifacts"],
                    max_runtime_artifact_bytes=raw["materialization_limits"]["max_artifact_bytes"],
                    max_collection_items=raw["materialization_limits"]["max_collection_items"],
                ),
                decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
                clock=_Clock(),
            ),
        )
    ).wait()
    return result, bound, nodes, callbacks, provider
