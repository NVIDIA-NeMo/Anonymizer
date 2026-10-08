# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Context source providers, workflow declarations, and initial admission fixtures."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from anonymizer.engine.graph_sdk._effect_values import EffectRejected
from anonymizer.engine.graph_sdk.binding import start_initial_binding
from anonymizer.engine.graph_sdk.context import (
    BindingLimits,
    ContextMaterialization,
    ContextResource,
    ContextSelector,
    ContextSourceCapability,
    ContextSourceRef,
    InitialContextDecl,
    RetrievalBounds,
    SourceFailure,
    SourceItem,
    SourceResponse,
)
from anonymizer.engine.graph_sdk.executor import (
    LocalCompleted,
)
from anonymizer.engine.graph_sdk.requests import (
    AssociationInput,
    AssociationResult,
    BindingAssociation,
    ExactUsage,
    ExternalSettlement,
    PhysicalRequestId,
    PhysicalRequestPolicy,
    PortArtifact,
    SemanticAssociation,
    StopConfirmed,
)
from anonymizer.engine.graph_sdk.resources import ResourceLease
from anonymizer.graph.workflow import (
    ArtifactType,
    ContextInputRef,
    ContextUse,
    CoverageAtom,
    DynamicLimits,
    DynamicScope,
    EvidencePromise,
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
    SubgraphNode,
    WorkflowId,
    WorkflowInputRef,
    WorkflowLimits,
    WorkflowOutcomeRef,
    WorkflowOutputRef,
    admit_activation_workflow,
    admit_static_workflow,
    substitute,
)
from tests.graph_sdk.test_preparation import _data

SOURCE = ContextSourceRef(name="context-source", revision=1)

CONTEXT_CASES = tuple(
    json.loads((Path(__file__).parent / "reference" / "context_source_v3" / "cases.json").read_bytes())["cases"]
)


@dataclass
class _ContextProvider:
    items: tuple[str, ...]
    omit: bool = False
    calls: int = 0
    same_key_versions: bool = False

    async def retrieve(
        self,
        *,
        request: PhysicalRequestId,
        association: SemanticAssociation | BindingAssociation,
        selector: ContextSelector,
        bounds: RetrievalBounds,
    ) -> SourceResponse | SourceFailure:
        del selector, bounds
        self.calls += 1
        settlement = ExternalSettlement(
            request=request,
            disposition="completed",
            usage=ExactUsage(input_units=0, output_units=0),
            remote_stopped=True,
        )
        if self.omit:
            return SourceFailure(
                source=SOURCE,
                failure="permanent",
                settlement=settlement,
                disposition="omitted_optional",
            )
        return SourceResponse(
            source=SOURCE,
            items=tuple(
                SourceItem(
                    association=association,
                    key=7 if self.same_key_versions else index,
                    version=index + 1 if self.same_key_versions else 1,
                    text=text,
                )
                for index, text in enumerate(self.items)
            ),
            settlement=settlement,
        )

    async def cancel(self, request: PhysicalRequestId) -> StopConfirmed:
        del request
        return StopConfirmed(usage=ExactUsage(input_units=0, output_units=0))


@dataclass
class _ProviderFactory:
    provider: _ContextProvider
    calls: int = 0

    def __call__(self) -> _ContextProvider:
        self.calls += 1
        return self.provider


@dataclass
class _IdentityCallback:
    calls: int = 0
    observed: list[object] = field(default_factory=list)

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        self.calls += 1
        value = request[0].inputs[0]
        self.observed.append(value.value)
        return LocalCompleted(
            results=(
                AssociationResult(
                    association=request[0].association,
                    outcome="ok",
                    outputs=(
                        PortArtifact(
                            port="output",
                            artifact_type=value.artifact_type,
                            artifact=None,
                            value=value.value,
                        ),
                    ),
                    consumed_context_ports=frozenset({"context"}),
                ),
            )
        )


@dataclass
class _FixtureCallback:
    outcome: str
    input_port: str
    output_port: str
    consumed: frozenset[str]
    calls: int = 0
    seen_ports: frozenset[str] = frozenset()

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        self.calls += 1
        self.seen_ports = frozenset(item.port for item in request[0].inputs)
        selected = next(item for item in request[0].inputs if item.port == self.input_port)
        return LocalCompleted(
            results=(
                AssociationResult(
                    association=request[0].association,
                    outcome=self.outcome,
                    outputs=(
                        PortArtifact(
                            port=self.output_port,
                            artifact_type=selected.artifact_type,
                            artifact=None,
                            value=selected.value,
                        ),
                    ),
                    consumed_context_ports=self.consumed,
                ),
            )
        )


def _operation(artifact_type: ArtifactType) -> OperationSpec:
    return OperationSpec(
        name="context-identity",
        inputs=(InputPort(name="context", artifact_type=artifact_type),),
        outputs=(OutputPort(name="output", artifact_type=artifact_type),),
        output_dependencies=(
            OutputDependency(output="output", inputs=frozenset({"context"}), identity_input="context"),
        ),
        outcomes=(
            OutcomeSpec(
                name="ok",
                category="success",
                produced_ports=frozenset({"output"}),
                context=frozenset({ContextUse(port="context", meaning="retrieved", capture="whole_artifact")}),
                evidence=frozenset(),
                state_effects=frozenset(),
                model_requirements=frozenset(),
                ceiling=ResourceCeiling(
                    max_activations=2,
                    max_model_requests=0,
                    max_input_bytes=16,
                    max_output_bytes=16,
                ),
            ),
        ),
    )


def _static_fixture(case: dict[str, Any]):
    raw = cast(dict[str, Any], case["input"])
    owner = WorkflowId.new()
    artifact_types: dict[str, ArtifactType] = {}

    def artifact_type(name: str) -> ArtifactType:
        return artifact_types.setdefault(name, ArtifactType(name=name, revision=1))

    node_ids = {name: NodeId.new(workflow=owner) for name in cast(dict[str, object], raw["nodes"])}
    outcomes_raw = cast(dict[str, dict[str, object]] | None, raw.get("outcomes"))
    dependencies_raw = cast(
        dict[str, dict[str, object]],
        raw.get("outcome_output_dependencies", {"output": raw["interface_output_dependency"]}),
    )
    outputs = tuple(
        OutputPort(
            name=name,
            artifact_type=artifact_type(
                cast(dict[str, str], raw["interface_inputs"])[cast(str, dependency["identity_input"])]
            ),
        )
        for name, dependency in dependencies_raw.items()
    )
    output_dependencies = tuple(
        OutputDependency(
            output=name,
            inputs=frozenset(cast(list[str], dependency["inputs"])),
            identity_input=cast(str, dependency["identity_input"]),
        )
        for name, dependency in dependencies_raw.items()
    )

    def operation(name: str, declaration: dict[str, object]) -> OperationSpec:
        input_types = cast(dict[str, str], declaration["inputs"])
        evidence_raw = cast(dict[str, object] | None, raw.get("evidence"))
        outcome_values = outcomes_raw or {
            "ok": {
                "context_ports": cast(list[str], declaration["context_ports"]),
                "produced_ports": list(dependencies_raw),
            }
        }
        return OperationSpec(
            name=f"fixture-{name}",
            inputs=tuple(
                InputPort(name=port, artifact_type=artifact_type(type_name)) for port, type_name in input_types.items()
            ),
            outputs=outputs,
            output_dependencies=output_dependencies,
            outcomes=tuple(
                OutcomeSpec(
                    name=outcome_name,
                    category="success",
                    produced_ports=frozenset(cast(list[str], outcome["produced_ports"])),
                    context=frozenset(
                        ContextUse(port=port, meaning="retrieved", capture="whole_artifact")
                        for port in cast(list[str], outcome["context_ports"])
                    ),
                    evidence=(
                        frozenset(
                            {
                                EvidencePromise(
                                    name=cast(str, evidence_raw["name"]),
                                    meaning=cast(str, evidence_raw["meaning"]),
                                    subject_port=cast(str, evidence_raw["subject_port"]),
                                    consumed_ports=frozenset(cast(list[str], evidence_raw["consumed_ports"])),
                                    coverage=frozenset(
                                        CoverageAtom(kind=cast(Any, item["kind"]), name=item["name"])
                                        for item in cast(list[dict[str, str]], evidence_raw["coverage"])
                                    ),
                                )
                            }
                        )
                        if evidence_raw is not None and outcome_name in {"ok", "main"}
                        else frozenset()
                    ),
                    state_effects=frozenset(),
                    model_requirements=frozenset(),
                    ceiling=ResourceCeiling(
                        max_activations=max(1, len(node_ids)),
                        max_model_requests=0,
                        max_input_bytes=32,
                        max_output_bytes=32,
                    ),
                )
                for outcome_name, outcome in outcome_values.items()
            ),
        )

    node_operations = {
        name: operation(name, declaration)
        for name, declaration in cast(dict[str, dict[str, object]], raw["nodes"]).items()
    }
    nodes = tuple(OperationNode(id=node_ids[name], operation=node_operations[name]) for name in node_ids)
    bindings = tuple(
        InputBinding(
            source=(
                ContextInputRef(port=cast(str, binding["port"]))
                if binding["kind"] == "context"
                else WorkflowInputRef(port=cast(str, binding["port"]))
            ),
            destination=NodeInputRef(
                node=node_ids[cast(str, binding["node"])],
                port=cast(str, binding["destination"]),
            ),
        )
        for binding in cast(list[dict[str, object]], raw["bindings"])
    )
    names = list(node_ids)
    sink_name = names[-1]
    sink = node_ids[sink_name]
    interface = OperationSpec(
        name="fixture-interface",
        inputs=tuple(
            InputPort(name=port, artifact_type=artifact_type(type_name))
            for port, type_name in cast(dict[str, str], raw["interface_inputs"]).items()
        ),
        outputs=outputs,
        output_dependencies=output_dependencies,
        outcomes=node_operations[sink_name].outcomes,
    )
    static = admit_static_workflow(
        workflow=owner,
        interface=interface,
        nodes=nodes,
        input_bindings=bindings,
        output_bindings=tuple(
            OutputBinding(
                source=NodeOutputRef(node=sink, port=output.name),
                destination=WorkflowOutputRef(port=output.name),
            )
            for output in outputs
        ),
        outcome_bindings=tuple(
            OutcomeBinding(
                source=NodeOutcomeRef(node=sink, outcome=outcome.name),
                destination=WorkflowOutcomeRef(outcome=outcome.name),
            )
            for outcome in interface.outcomes
        ),
        sequence=tuple(
            SequenceEdge(before=node_ids[left], after=node_ids[right])
            for left, right in zip(names, names[1:], strict=False)
        ),
        choices=(),
        protection=(),
        limits=WorkflowLimits(
            max_nodes=len(nodes),
            max_bindings=len(bindings) + len(outputs) + len(interface.outcomes),
            max_sequence_edges=max(0, len(nodes) - 1),
            max_choices=0,
            max_branch_members=0,
            max_subgraph_depth=1,
            max_choice_states=len(interface.outcomes),
        ),
    )
    return static, node_ids, artifact_types


CONTEXT_ADMISSION_CASES = tuple(
    case
    for case in CONTEXT_CASES
    if "context" in cast(dict[str, object], case["expected"])
    or "initial_binding_admission" in cast(dict[str, object], case["expected"])
)


async def _assert_context_source_initial_admission(case: dict[str, Any]) -> None:
    raw = cast(dict[str, Any], case["input"])
    expected = cast(dict[str, Any], case["expected"])
    if raw.get("nested"):
        declared_type = next(iter(cast(dict[str, str], raw["interface_inputs"]).values()))
        workflow, initial_node, _ = _nested_workflow(ArtifactType(name=declared_type, revision=1))
        nodes = {"N0": initial_node}
        artifact_types = {
            name: ArtifactType(name=name, revision=1)
            for name in {
                *cast(dict[str, str], raw["interface_inputs"]).values(),
                "text",
            }
        }
    else:
        static, nodes, artifact_types = _static_fixture(case)
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
                max_activation_occurrences=max(1, len(nodes)),
            ),
        )
    data = _data(len(cast(list[str], raw["targets"])))
    target_values = sorted(data.targets, key=repr)
    targets = {name: target for name, target in zip(cast(list[str], raw["targets"]), target_values, strict=True)}
    foreign_data = _data(1)
    foreign_target = next(iter(foreign_data.targets))
    policy = PhysicalRequestPolicy(
        visibility="dispatch_and_settlement",
        pre_dispatch_control="executor",
        retry_owner="executor",
        replay="idempotent",
        max_attempts=1,
    )
    source_items = cast(list[dict[str, Any]], raw["source_items"])
    provider = _ContextProvider(
        items=tuple(item["text"] for item in source_items),
        omit=case["id"] == "optional_context_omission",
    )
    initial_values = cast(list[dict[str, Any]], raw["initial"])
    scalar_type = artifact_types.setdefault("text", ArtifactType(name="text", revision=1))
    declarations: list[InitialContextDecl] = []
    item_types: set[ArtifactType] = set()
    for value in initial_values:
        type_name = cast(str, value["type"])
        output_type = artifact_types.setdefault(type_name, ArtifactType(name=type_name, revision=1))
        item_name = cast(
            str,
            value.get("item_type", "text" if type_name.endswith("collection") else type_name),
        )
        item_type = artifact_types.setdefault(item_name, ArtifactType(name=item_name, revision=1))
        item_types.add(item_type)
        target_name = cast(str, value["target"])
        target = foreign_target if target_name.startswith("FOREIGN:") else targets[target_name]
        node_name = cast(str, value["node"])
        if node_name.startswith("FOREIGN:"):
            node = NodeId.new(workflow=WorkflowId.new())
        elif node_name == "ABSENT_SAME_OWNER":
            node = NodeId.new(workflow=workflow.workflow.workflow)
        else:
            node = nodes[node_name]
        shape = cast(str, value.get("shape", "collection" if type_name.endswith("collection") else "single"))
        declarations.append(
            InitialContextDecl(
                target=target,
                node=node,
                port=cast(str, value["port"]),
                artifact_type=output_type,
                source=SOURCE,
                selector=ContextSelector(fields=()),
                requirement="optional" if value["optional"] else "required",
                bounds=RetrievalBounds(
                    max_items=max(1, len(source_items)) if shape == "collection" else 1,
                    max_bytes=max(1, sum(len(item["text"].encode()) for item in source_items)),
                    max_requests=1,
                ),
                materialization=ContextMaterialization(kind=cast(Any, shape), item_type=item_type),
            )
        )
    source_item_type = next(iter(item_types), scalar_type)
    source_capability = ContextSourceCapability(
        source=SOURCE,
        artifact_type=source_item_type,
        uses=frozenset({"initial_binding"}),
        execution="async",
        resource_owner="caller",
        cancellation="cooperative_ack",
        settlement="explicit_ack",
        usage="exact",
        request=policy,
        safe_detachment="forbidden",
    )
    capabilities = (source_capability,) if declarations else ()
    resources = (
        (
            ContextResource(
                source=SOURCE,
                capability=source_capability,
                lease=ResourceLease.create(owner="caller", safe_detachment="forbidden", handle=provider),
                factory=None,
            ),
        )
        if declarations
        else ()
    )
    expected_code = cast(str, expected.get("context", expected.get("initial_binding_admission")))
    try:
        result = await (
            await start_initial_binding(
                data=data,
                workflow=workflow,
                declarations=tuple(declarations),
                capabilities=capabilities,
                resources=resources,
                limits=BindingLimits(
                    max_declarations=max(2, len(declarations)),
                    max_sources=1,
                    max_capabilities=1,
                    max_selector_fields=0,
                    max_selector_bytes=0,
                    max_items=max(2, len(source_items) * max(1, len(declarations))),
                    max_bytes=max(
                        8, sum(len(item["text"].encode()) for item in source_items) * max(1, len(declarations))
                    ),
                    max_requests=max(1, len(declarations)),
                    max_resources=1,
                ),
            )
        ).wait()
    except EffectRejected as rejected:
        assert rejected.code.value == expected_code
        assert provider.calls == cast(int, expected.get("provider_calls", 0))
        return
    assert expected_code == "accepted"
    assert result.context is not None
    assert provider.calls == len(declarations)


def _single_workflow(artifact_type: ArtifactType):
    owner = WorkflowId.new()
    node = NodeId.new(workflow=owner)
    operation = _operation(artifact_type)
    static = admit_static_workflow(
        workflow=owner,
        interface=operation,
        nodes=(OperationNode(id=node, operation=operation),),
        input_bindings=(
            InputBinding(
                source=ContextInputRef(port="context"),
                destination=NodeInputRef(node=node, port="context"),
            ),
        ),
        output_bindings=(
            OutputBinding(
                source=NodeOutputRef(node=node, port="output"),
                destination=WorkflowOutputRef(port="output"),
            ),
        ),
        outcome_bindings=(
            OutcomeBinding(
                source=NodeOutcomeRef(node=node, outcome="ok"),
                destination=WorkflowOutcomeRef(outcome="ok"),
            ),
        ),
        sequence=(),
        choices=(),
        protection=(),
        limits=WorkflowLimits(
            max_nodes=1,
            max_bindings=3,
            max_sequence_edges=0,
            max_choices=0,
            max_branch_members=0,
            max_subgraph_depth=1,
            max_choice_states=1,
        ),
    )
    return (
        admit_activation_workflow(
            workflow=static,
            scopes=(DynamicScope(workflow=static, maps=(), joins=(), loops=()),),
            limits=DynamicLimits(
                max_maps=0,
                max_joins=0,
                max_loops=0,
                max_children_per_map=0,
                max_iterations_per_loop=0,
                max_dynamic_depth=1,
                max_activation_occurrences=1,
            ),
        ),
        node,
        node,
    )


def _nested_workflow(artifact_type: ArtifactType, *, via_substitution: bool = False):
    operation = _operation(artifact_type)
    body_owner = WorkflowId.new()
    inner = NodeId.new(workflow=body_owner)
    body = admit_static_workflow(
        workflow=body_owner,
        interface=operation,
        nodes=(OperationNode(id=inner, operation=operation),),
        input_bindings=(
            InputBinding(
                source=ContextInputRef(port="context"),
                destination=NodeInputRef(node=inner, port="context"),
            ),
        ),
        output_bindings=(
            OutputBinding(
                source=NodeOutputRef(node=inner, port="output"),
                destination=WorkflowOutputRef(port="output"),
            ),
        ),
        outcome_bindings=(
            OutcomeBinding(
                source=NodeOutcomeRef(node=inner, outcome="ok"),
                destination=WorkflowOutcomeRef(outcome="ok"),
            ),
        ),
        sequence=(),
        choices=(),
        protection=(),
        limits=WorkflowLimits(
            max_nodes=1,
            max_bindings=3,
            max_sequence_edges=0,
            max_choices=0,
            max_branch_members=0,
            max_subgraph_depth=1,
            max_choice_states=1,
        ),
    )
    owner = WorkflowId.new()
    wrapper = NodeId.new(workflow=owner)
    root_node = (
        OperationNode(id=wrapper, operation=operation)
        if via_substitution
        else SubgraphNode(id=wrapper, operation=operation, body=body)
    )
    root = admit_static_workflow(
        workflow=owner,
        interface=operation,
        nodes=(root_node,),
        input_bindings=(
            InputBinding(
                source=ContextInputRef(port="context"),
                destination=NodeInputRef(node=wrapper, port="context"),
            ),
        ),
        output_bindings=(
            OutputBinding(
                source=NodeOutputRef(node=wrapper, port="output"),
                destination=WorkflowOutputRef(port="output"),
            ),
        ),
        outcome_bindings=(
            OutcomeBinding(
                source=NodeOutcomeRef(node=wrapper, outcome="ok"),
                destination=WorkflowOutcomeRef(outcome="ok"),
            ),
        ),
        sequence=(),
        choices=(),
        protection=(),
        limits=WorkflowLimits(
            max_nodes=2,
            max_bindings=3,
            max_sequence_edges=0,
            max_choices=0,
            max_branch_members=0,
            max_subgraph_depth=2,
            max_choice_states=1,
        ),
    )
    if via_substitution:
        root = substitute(workflow=root, target=wrapper, replacement=body)
    return (
        admit_activation_workflow(
            workflow=root,
            scopes=(
                DynamicScope(workflow=root, maps=(), joins=(), loops=()),
                DynamicScope(workflow=body, maps=(), joins=(), loops=()),
            ),
            limits=DynamicLimits(
                max_maps=0,
                max_joins=0,
                max_loops=0,
                max_children_per_map=0,
                max_iterations_per_loop=0,
                max_dynamic_depth=2,
                max_activation_occurrences=2,
            ),
        ),
        wrapper,
        inner,
    )
