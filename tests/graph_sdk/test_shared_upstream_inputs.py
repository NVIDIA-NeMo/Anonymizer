# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two inputs bound to one upstream output publish one shared provenance parent."""

from __future__ import annotations

import asyncio

import pytest

from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.executor import (
    AssessmentLimits,
    ExecutionLimits,
    ExecutionServices,
    ImplementationHandle,
    LocalCompleted,
    OperationExecutionPolicy,
    OperationOutputKey,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import BoundInput, PreparationConfiguration, prepare
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.requests import AssociationInput, AssociationResult, PortArtifact, TextArtifactValue
from anonymizer.graph import (
    ActivationLimits,
    ArtifactType,
    InputBinding,
    InputPort,
    NodeId,
    NodeInputRef,
    NodeOutcomeRef,
    NodeOutputRef,
    OperationNode,
    OperationSpec,
    OutcomeBinding,
    OutcomeSpec,
    OutputBinding,
    OutputDependency,
    OutputPort,
    ResourceCeiling,
    SequenceEdge,
    WorkflowId,
    WorkflowInputRef,
    WorkflowLimits,
    WorkflowOutcomeRef,
    WorkflowOutputRef,
    compile_graph,
)
from tests.graph_sdk.implementation_fixtures import _execution_implementation, _selection
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.graph_sdk.test_dynamic_executor import _capability, _rows
from tests.graph_sdk.test_local_executor import _Clock
from tests.graph_sdk.test_preparation import _data, _limits

TEXT = ArtifactType(name="text", revision=1)


def _operation(name: str, inputs: tuple[str, ...], output: str, activations: int = 1) -> OperationSpec:
    return OperationSpec(
        name=name,
        inputs=tuple(InputPort(name=port, artifact_type=TEXT) for port in inputs),
        outputs=(OutputPort(name=output, artifact_type=TEXT),),
        output_dependencies=(OutputDependency(output=output, inputs=frozenset(inputs), identity_input=None),),
        outcomes=(
            OutcomeSpec(
                name="ok",
                category="success",
                produced_ports=frozenset({output}),
                context=frozenset(),
                evidence=frozenset(),
                model_requirements=frozenset(),
                ceiling=ResourceCeiling(
                    max_activations=activations,
                    max_model_requests=0,
                    max_input_bytes=100 * activations,
                    max_output_bytes=100 * activations,
                ),
            ),
        ),
    )


class _Emit:
    def __init__(self, port: str) -> None:
        self.port = port

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        output = PortArtifact(
            port=self.port, artifact_type=TEXT, artifact=None, value=TextArtifactValue(text=self.port)
        )
        return LocalCompleted(
            results=(
                AssociationResult(
                    association=request[0].association,
                    outcome="ok",
                    outputs=(output,),
                    consumed_context_ports=frozenset(),
                ),
            )
        )


@pytest.mark.parametrize("shared", [True, False])
def test_inputs_bound_to_one_upstream_output_publish(shared: bool) -> None:
    workflow = WorkflowId.new()
    first, second, joined = (NodeId.new(workflow=workflow) for _ in range(3))
    nodes = (
        OperationNode(id=first, operation=_operation("first", ("text",), "out")),
        OperationNode(id=second, operation=_operation("second", ("text",), "out")),
        OperationNode(id=joined, operation=_operation("joined", ("left", "right"), "joined")),
    )
    graph = compile_graph(
        workflow=workflow,
        interface=_operation("interface", ("input",), "joined", activations=3),
        nodes=nodes,
        input_bindings=(
            InputBinding(source=WorkflowInputRef(port="input"), destination=NodeInputRef(node=first, port="text")),
            InputBinding(source=WorkflowInputRef(port="input"), destination=NodeInputRef(node=second, port="text")),
            InputBinding(
                source=NodeOutputRef(node=first, port="out"), destination=NodeInputRef(node=joined, port="left")
            ),
            InputBinding(
                source=NodeOutputRef(node=first if shared else second, port="out"),
                destination=NodeInputRef(node=joined, port="right"),
            ),
        ),
        output_bindings=(
            OutputBinding(
                source=NodeOutputRef(node=joined, port="joined"), destination=WorkflowOutputRef(port="joined")
            ),
        ),
        outcome_bindings=(
            OutcomeBinding(
                source=NodeOutcomeRef(node=joined, outcome="ok"), destination=WorkflowOutcomeRef(outcome="ok")
            ),
        ),
        sequence=(SequenceEdge(before=first, after=joined), SequenceEdge(before=second, after=joined)),
        protection=(),
        limits=WorkflowLimits(
            max_nodes=3,
            max_bindings=6,
            max_sequence_edges=2,
            max_choices=0,
            max_branch_members=0,
            max_subgraph_depth=1,
            max_choice_states=1,
        ),
    )
    capabilities = tuple(_capability(node.operation, index) for index, node in enumerate(nodes))
    by_node = {node.id: capability for node, capability in zip(nodes, capabilities, strict=True)}
    data = _data(1)
    target = next(iter(data.targets))
    prepared = prepare(
        data=data,
        workflow=graph,
        activation_limits=ActivationLimits(max_events=20, max_entries=3, max_parent_depth=1),
        bound_inputs=(BoundInput(target=target, source=target, port="input", artifact_type=TEXT),),
        configuration=PreparationConfiguration(
            purpose="execution_only", required_protection_outcomes=frozenset(), hard_request_limit=None
        ),
        selections=tuple(_selection(node.id, by_node[node.id]) for node in nodes),
        capabilities=capabilities,
        limits=_limits(capabilities=3, slots=3),
    )
    execution = admit_execution_plan(
        context=admit_context_plan(prepared=prepared, bound_context=None),
        capabilities=capabilities,
        policies=tuple(
            OperationExecutionPolicy(
                node=node.id,
                kind="local",
                request=None,
                safe_detachment="forbidden",
                implementations=(_execution_implementation(by_node[node.id], request=None),),
                result_outcomes=frozenset({"ok"}),
                runtime_outcomes=_rows(frozenset({"ok"})),
            )
            for node in nodes
        ),
        assessment_productions=(),
        assessment_limits=AssessmentLimits(
            max_productions=0,
            max_findings_per_production=0,
            max_finding_code_bytes=0,
            max_assessment_facts=0,
            max_port_facts=10,
            max_provenance_edges=10,
        ),
    )
    handles = tuple(
        ImplementationHandle(
            implementation=by_node[node.id].implementation,
            operation=node.operation,
            configuration=by_node[node.id].configuration,
            local=_Emit(node.operation.outputs[0].name),
            transport=None,
            resource=None,
        )
        for node in nodes
    )
    services = ExecutionServices(
        handles=handles,
        limits=ExecutionLimits(
            max_local_in_flight=3,
            max_remote_outstanding=0,
            max_runtime_artifacts=8,
            max_runtime_artifact_bytes=300,
            max_collection_items=0,
        ),
        clock=_Clock(),
    )

    async def run():
        running = await start_execution(admitted=execution, capabilities=capabilities, services=services)
        return await running.wait()

    result = asyncio.run(run())
    assert [row.category for row in result.record.terminals] == ["success"] * 3
    assert [(row.outcome, row.port) for row in result.final_outputs] == [("ok", "joined")]
    templates = {entry.activation: entry.template for entry in result.states[0].entries}
    (fact,) = (
        row
        for row in result.provenance
        if isinstance(row.key, OperationOutputKey) and templates[row.key.activation] == joined
    )
    parents = {templates[parent.activation] for parent in fact.parents if isinstance(parent, OperationOutputKey)}
    assert len(fact.parents) == len(parents) == (1 if shared else 2)
    assert parents == ({first} if shared else {first, second})
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert output.targets[0].withholding == frozenset({"execution_only"})
