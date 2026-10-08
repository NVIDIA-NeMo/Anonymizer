# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executable coverage for explicit static context input sources."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectRejected
from anonymizer.engine.graph_sdk.binding import start_initial_binding
from anonymizer.engine.graph_sdk.capabilities import ImplementationSelection
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
    admit_context_plan,
)
from anonymizer.engine.graph_sdk.evidence import admit_qualification, evidence_revision_view
from anonymizer.engine.graph_sdk.executor import (
    AssessmentLimits,
    BoundInputKey,
    DecisionLimits,
    ExecutionImplementation,
    ExecutionLimits,
    ExecutionServices,
    ImplementationHandle,
    InitialCollectionKey,
    LocalCompleted,
    OperationExecutionPolicy,
    OperationOutputKey,
    RuntimeOutcome,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import BoundInput, PreparationConfiguration, StateRevisionView, prepare
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.requests import (
    AssociationInput,
    AssociationResult,
    BindingAssociation,
    BindingId,
    ExactUsage,
    ExternalSettlement,
    PhysicalRequestId,
    PhysicalRequestPolicy,
    PortArtifact,
    SemanticAssociation,
    StopConfirmed,
    TextArtifactValue,
    TextCollectionValue,
)
from anonymizer.engine.graph_sdk.resources import ResourceLease
from anonymizer.graph.activation import ActivationLimits
from anonymizer.graph.workflow import (
    ArtifactType,
    ContextInputRef,
    ContextUse,
    ContractViolation,
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
from tests.graph_sdk.test_evidence import _qualification_limits
from tests.graph_sdk.test_local_executor import _Clock, _runtime_rows
from tests.graph_sdk.test_preparation import _capability, _data, _limits

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


@pytest.mark.parametrize("case", CONTEXT_CASES, ids=lambda case: cast(str, case["id"]))
def test_context_source_static_reference_cases(case: dict[str, Any]) -> None:
    expected = cast(dict[str, object], case["expected"])
    nested = bool(cast(dict[str, object], case["input"]).get("nested"))
    if nested:
        input_name = next(iter(cast(dict[str, str], cast(dict[str, object], case["input"])["interface_inputs"])))
        input_type = ArtifactType(
            name=cast(dict[str, str], cast(dict[str, object], case["input"])["interface_inputs"])[input_name],
            revision=1,
        )
        workflow, _, _ = _nested_workflow(input_type)
        admitted = workflow.workflow
    elif expected["static"] == "accepted":
        admitted, _, _ = _static_fixture(case)
    else:
        with pytest.raises(ContractViolation) as rejected:
            _static_fixture(case)
        assert rejected.value.code.value == expected["static"]
        return
    assert admitted.workflow is not None
    if case["id"] == "context_evidence_projection":
        outcome = admitted.interface.outcomes[0]
        promise = next(iter(outcome.evidence))
        assert promise.subject_port == "context"
        assert promise.consumed_ports == frozenset({"context"})
        assert next(iter(promise.coverage)) == CoverageAtom(kind="field", name="text")
        assert next(iter(outcome.context)).port == "context"
        assert admitted.interface.output_dependencies[0].inputs == frozenset({"context"})


CONTEXT_ADMISSION_CASES = tuple(
    case
    for case in CONTEXT_CASES
    if "context" in cast(dict[str, object], case["expected"])
    or "initial_binding_admission" in cast(dict[str, object], case["expected"])
)


@pytest.mark.parametrize("case", CONTEXT_ADMISSION_CASES, ids=lambda case: cast(str, case["id"]))
def test_context_source_initial_admission_reference_cases(case: dict[str, Any]) -> None:
    asyncio.run(_assert_context_source_initial_admission(case))


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


def test_scalar_collection_and_nested_context_sources_execute_with_exact_identity() -> None:
    asyncio.run(_assert_context_execution("scalar"))
    asyncio.run(_assert_context_execution("collection"))
    asyncio.run(_assert_context_execution("nested"))
    asyncio.run(_assert_context_execution("substitution"))


@pytest.mark.parametrize("latest", [False, True])
def test_optional_context_omission_closes_unstarted_without_an_attempt(latest: bool) -> None:
    asyncio.run(_assert_context_execution("omitted", latest=latest))


def test_ordinary_root_and_actual_outcome_context_union_execute_at_real_boundaries() -> None:
    asyncio.run(_assert_special_fixture_execution("ordinary_context_use_preserved"))
    asyncio.run(_assert_special_fixture_execution("outcome_context_union"))


async def _assert_special_fixture_execution(case_id: str) -> None:
    case = next(item for item in CONTEXT_CASES if item["id"] == case_id)
    raw = cast(dict[str, Any], case["input"])
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
            max_activation_occurrences=1,
        ),
    )
    node = nodes["N0"]
    data = _data(1)
    target = next(iter(data.targets))
    policy = PhysicalRequestPolicy(
        visibility="dispatch_and_settlement",
        pre_dispatch_control="executor",
        retry_owner="executor",
        replay="idempotent",
        max_attempts=1,
    )
    item_type = artifact_types["text"]
    source_capability = ContextSourceCapability(
        source=SOURCE,
        artifact_type=item_type,
        uses=frozenset({"initial_binding"}),
        execution="async",
        resource_owner="caller",
        cancellation="cooperative_ack",
        settlement="explicit_ack",
        usage="exact",
        request=policy,
        safe_detachment="forbidden",
    )
    declarations = tuple(
        InitialContextDecl(
            target=target,
            node=node,
            port=cast(str, value["port"]),
            artifact_type=artifact_types[cast(str, value["type"])],
            source=SOURCE,
            selector=ContextSelector(fields=()),
            requirement="required",
            bounds=RetrievalBounds(max_items=1, max_bytes=8, max_requests=1),
            materialization=ContextMaterialization(kind="single", item_type=item_type),
        )
        for value in cast(list[dict[str, object]], raw["initial"])
    )
    provider = _ContextProvider(items=("ab",))
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
    binding = await (
        await start_initial_binding(
            data=data,
            workflow=workflow,
            declarations=declarations,
            capabilities=capabilities,
            resources=resources,
            limits=BindingLimits(
                max_declarations=2,
                max_sources=1,
                max_capabilities=1,
                max_selector_fields=0,
                max_selector_bytes=0,
                max_items=2,
                max_bytes=16,
                max_requests=2,
                max_resources=1,
            ),
        )
    ).wait()
    assert binding.context is not None
    capability = _capability(workflow)
    root_values = cast(dict[str, object], raw["root_values"])
    prepared = prepare(
        data=data,
        workflow=workflow,
        activation_limits=ActivationLimits(max_events=4, max_entries=1, max_parent_depth=1),
        bound_inputs=tuple(
            # The data target is the exact scalar P1 source; caller roots never use collection values.
            BoundInput(
                target=target,
                source=target,
                port=port,
                artifact_type=artifact_types[
                    cast(str, value["type"])
                    if isinstance(value, dict)
                    else cast(dict[str, str], raw["interface_inputs"])[port]
                ],
            )
            for port, value in root_values.items()
        ),
        configuration=PreparationConfiguration(
            purpose="execution_only",
            required_protection_outcomes=frozenset(),
            hard_request_limit=None,
        ),
        state=StateRevisionView(revisions=frozenset()),
        selections=(
            ImplementationSelection(
                node=node,
                implementation=capability.implementation,
                configuration=capability.configuration,
            ),
        ),
        capabilities=(capability,),
        limits=_limits(slots=1),
    )
    context = admit_context_plan(
        prepared=prepared,
        bound_context=binding.context,
        adaptive_retrievals=(),
        context_capabilities=(),
    )
    implementation = ExecutionImplementation(
        implementation=capability.implementation,
        configuration=capability.configuration,
        capability=capability,
        request=None,
    )
    returned_outcome = cast(str, raw.get("returned_outcome", "ok"))
    output_port = "output" if returned_outcome in {"ok", "main"} else "extra_output"
    input_port = "context" if returned_outcome in {"ok", "main"} else "extra"
    callback = _FixtureCallback(
        outcome=returned_outcome,
        input_port=input_port,
        output_port=output_port,
        consumed=frozenset({input_port}),
    )
    admitted = admit_execution_plan(
        context=context,
        capabilities=(capability,),
        policies=(
            OperationExecutionPolicy(
                node=node,
                kind="local",
                request=None,
                safe_detachment="forbidden",
                implementations=(implementation,),
                result_outcomes=frozenset(outcome.name for outcome in capability.operation.outcomes),
                runtime_outcomes=tuple(
                    row
                    for outcome in capability.operation.outcomes
                    for row in _runtime_rows()
                    if row.condition != "result" or (row.reported_outcome == "ok" and outcome.name == "ok")
                )
                if len(capability.operation.outcomes) == 1
                else (
                    *tuple(
                        RuntimeOutcome(
                            condition="result",
                            reported_outcome=outcome.name,
                            failure=None,
                            outcome=outcome.name,
                            category="success",
                        )
                        for outcome in capability.operation.outcomes
                    ),
                    *tuple(row for row in _runtime_rows() if row.condition != "result"),
                ),
            ),
        ),
        decisions=(),
        assessment_productions=(),
        assessment_limits=AssessmentLimits(
            max_productions=0,
            max_findings_per_production=0,
            max_finding_code_bytes=0,
            max_absence_queries=0,
            max_assessment_facts=0,
            max_port_facts=6,
            max_provenance_edges=6,
        ),
    )
    result = await (
        await start_execution(
            admitted=admitted,
            capabilities=(capability,),
            services=ExecutionServices(
                handles=(
                    ImplementationHandle(
                        implementation=capability.implementation,
                        operation=capability.operation,
                        configuration=capability.configuration,
                        local=callback,
                        transport=None,
                        resource=None,
                    ),
                ),
                context_resources=(),
                limits=ExecutionLimits(
                    max_local_in_flight=1,
                    max_remote_outstanding=0,
                    max_runtime_artifacts=8,
                    max_runtime_artifact_bytes=32,
                    max_collection_items=2,
                ),
                decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
                clock=_Clock(),
            ),
        )
    ).wait()
    assert callback.calls == 1
    assert result.record.terminals[0].category == "success"
    assert result.final_outputs[0].port == output_port
    if case_id == "ordinary_context_use_preserved":
        assert callback.seen_ports == frozenset({"context"})
        assert any(type(fact.key).__name__ == "RootInputKey" for fact in result.provenance)
    else:
        assert callback.seen_ports == frozenset({"context", "extra"})
        assert len(binding.context.artifacts) == 2
        assert result.final_outputs[0].candidate.artifact == next(
            fact.artifact
            for fact in result.provenance
            if type(fact.key).__name__ == "BoundInputKey" and getattr(fact.key, "port") == "context"
        )


def test_initial_collection_schema_rejects_same_typed_caller_root_before_provider_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_collection_root_preflight(monkeypatch)


def _assert_collection_root_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    binding_ids = 0
    original_new = BindingId.new

    def counted_new(cls: type[BindingId]) -> BindingId:
        del cls
        nonlocal binding_ids
        binding_ids += 1
        return original_new()

    monkeypatch.setattr(BindingId, "new", classmethod(counted_new))
    factory = asyncio.run(_assert_collection_schema_rejects_caller_root())
    assert binding_ids == 0
    assert factory.calls == 0


async def _assert_collection_schema_rejects_caller_root() -> _ProviderFactory:
    collection_type = ArtifactType(name="collection", revision=1)
    item_type = ArtifactType(name="item", revision=1)
    owner = WorkflowId.new()
    node = NodeId.new(workflow=owner)
    operation = OperationSpec(
        name="context-with-caller-root",
        inputs=(
            InputPort(name="context", artifact_type=collection_type),
            InputPort(name="caller", artifact_type=collection_type),
        ),
        outputs=(OutputPort(name="output", artifact_type=collection_type),),
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
                    max_activations=1,
                    max_model_requests=0,
                    max_input_bytes=16,
                    max_output_bytes=16,
                ),
            ),
        ),
    )
    static = admit_static_workflow(
        workflow=owner,
        interface=operation,
        nodes=(OperationNode(id=node, operation=operation),),
        input_bindings=(
            InputBinding(
                source=ContextInputRef(port="context"),
                destination=NodeInputRef(node=node, port="context"),
            ),
            InputBinding(
                source=WorkflowInputRef(port="caller"),
                destination=NodeInputRef(node=node, port="caller"),
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
            max_bindings=4,
            max_sequence_edges=0,
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
            max_activation_occurrences=1,
        ),
    )
    data = _data(1)
    policy = PhysicalRequestPolicy(
        visibility="dispatch_and_settlement",
        pre_dispatch_control="executor",
        retry_owner="executor",
        replay="idempotent",
        max_attempts=1,
    )
    capability = ContextSourceCapability(
        source=SOURCE,
        artifact_type=item_type,
        uses=frozenset({"initial_binding"}),
        execution="async",
        resource_owner="sdk",
        cancellation="cooperative_ack",
        settlement="explicit_ack",
        usage="exact",
        request=policy,
        safe_detachment="forbidden",
    )
    provider = _ContextProvider(items=("ab", "cd"))
    factory = _ProviderFactory(provider=provider)
    declaration = InitialContextDecl(
        target=next(iter(data.targets)),
        node=node,
        port="context",
        artifact_type=collection_type,
        source=SOURCE,
        selector=ContextSelector(fields=()),
        requirement="required",
        bounds=RetrievalBounds(max_items=2, max_bytes=8, max_requests=1),
        materialization=ContextMaterialization(kind="collection", item_type=item_type),
    )
    with pytest.raises(EffectRejected) as rejected:
        await start_initial_binding(
            data=data,
            workflow=workflow,
            declarations=(declaration,),
            capabilities=(capability,),
            resources=(
                ContextResource(
                    source=SOURCE,
                    capability=capability,
                    lease=None,
                    factory=factory,
                ),
            ),
            limits=BindingLimits(
                max_declarations=1,
                max_sources=1,
                max_capabilities=1,
                max_selector_fields=0,
                max_selector_bytes=0,
                max_items=2,
                max_bytes=8,
                max_requests=1,
                max_resources=1,
            ),
        )
    assert rejected.value.code.value == "contradictory"
    assert provider.calls == 0
    foreign_data = _data(1)
    foreign_target = next(iter(foreign_data.targets))
    foreign_declaration = replace(declaration, target=foreign_target)
    assert foreign_target not in data.targets
    assert foreign_declaration.target is foreign_target
    with pytest.raises(EffectRejected) as rejected:
        await start_initial_binding(
            data=data,
            workflow=workflow,
            declarations=(foreign_declaration,),
            capabilities=(capability,),
            resources=(ContextResource(source=SOURCE, capability=capability, lease=None, factory=factory),),
            limits=BindingLimits(
                max_declarations=1,
                max_sources=1,
                max_capabilities=1,
                max_selector_fields=0,
                max_selector_bytes=0,
                max_items=2,
                max_bytes=8,
                max_requests=1,
                max_resources=1,
            ),
        )
    assert rejected.value.code.value == "foreign_owner"
    assert provider.calls == 0
    return factory


async def _assert_context_execution(
    mode: str, *, same_key_versions: bool = False, artifact_limit: int = 6, latest: bool = False
) -> None:
    item_type = ArtifactType(name="text", revision=1)
    input_type = (
        ArtifactType(name="text_collection", revision=1)
        if mode in {"collection", "nested", "substitution"}
        else item_type
    )
    workflow, initial_node, implementation_node = (
        _nested_workflow(input_type, via_substitution=mode == "substitution")
        if mode in {"nested", "substitution"}
        else _single_workflow(input_type)
    )
    if mode == "substitution":
        assert isinstance(next(iter(workflow.workflow.nodes)), SubgraphNode)
        assert isinstance(next(iter(workflow.workflow.input_bindings)).source, ContextInputRef)
    data = _data(1)
    target = next(iter(data.targets))
    request_policy = PhysicalRequestPolicy(
        visibility="dispatch_and_settlement",
        pre_dispatch_control="executor",
        retry_owner="executor",
        replay="idempotent",
        max_attempts=1,
    )
    source_capability = ContextSourceCapability(
        source=SOURCE,
        artifact_type=item_type,
        uses=frozenset({"initial_binding"}),
        execution="async",
        resource_owner="caller",
        cancellation="cooperative_ack",
        settlement="explicit_ack",
        usage="exact",
        request=request_policy,
        safe_detachment="forbidden",
    )
    provider = _ContextProvider(
        items=("ab", "cd") if mode in {"collection", "nested", "substitution"} else ("ab",),
        omit=mode == "omitted",
        same_key_versions=same_key_versions,
    )
    declaration = InitialContextDecl(
        target=target,
        node=initial_node,
        port="context",
        artifact_type=input_type,
        source=SOURCE,
        selector=ContextSelector(fields=()),
        requirement="optional" if mode == "omitted" else "required",
        bounds=RetrievalBounds(
            max_items=2 if mode in {"collection", "nested", "substitution"} else 1,
            max_bytes=8,
            max_requests=1,
        ),
        materialization=ContextMaterialization(
            kind="collection" if mode in {"collection", "nested", "substitution"} else "single",
            item_type=item_type,
        ),
        version_selection="latest" if latest else "exact_one",
    )
    binding = await (
        await start_initial_binding(
            data=data,
            workflow=workflow,
            declarations=(declaration,),
            capabilities=(source_capability,),
            resources=(
                ContextResource(
                    source=SOURCE,
                    capability=source_capability,
                    lease=ResourceLease.create(owner="caller", safe_detachment="forbidden", handle=provider),
                    factory=None,
                ),
            ),
            limits=BindingLimits(
                max_declarations=1,
                max_sources=1,
                max_capabilities=1,
                max_selector_fields=0,
                max_selector_bytes=0,
                max_items=2,
                max_bytes=8,
                max_requests=1,
                max_resources=1,
            ),
        )
    ).wait()
    assert binding.context is not None
    if mode == "omitted":
        assert binding.receipt.sources[0].terminal == "omitted_optional"
        assert not binding.receipt.artifacts
        assert not binding.context.artifacts
    capability = _capability(workflow)
    prepared = prepare(
        data=data,
        workflow=workflow,
        activation_limits=ActivationLimits(max_events=8, max_entries=2, max_parent_depth=2),
        bound_inputs=(),
        configuration=PreparationConfiguration(
            purpose="execution_only",
            required_protection_outcomes=frozenset(),
            hard_request_limit=None,
        ),
        state=StateRevisionView(revisions=frozenset()),
        selections=(
            ImplementationSelection(
                node=implementation_node,
                implementation=capability.implementation,
                configuration=capability.configuration,
            ),
        ),
        capabilities=(capability,),
        limits=_limits(slots=2),
    )
    context = admit_context_plan(
        prepared=prepared,
        bound_context=binding.context,
        adaptive_retrievals=(),
        context_capabilities=(),
    )
    implementation = ExecutionImplementation(
        implementation=capability.implementation,
        configuration=capability.configuration,
        capability=capability,
        request=None,
    )
    admitted = admit_execution_plan(
        context=context,
        capabilities=(capability,),
        policies=(
            OperationExecutionPolicy(
                node=implementation_node,
                kind="local",
                request=None,
                safe_detachment="forbidden",
                implementations=(implementation,),
                result_outcomes=frozenset({"ok"}),
                runtime_outcomes=_runtime_rows(),
            ),
        ),
        decisions=(),
        assessment_productions=(),
        assessment_limits=AssessmentLimits(
            max_productions=0,
            max_findings_per_production=0,
            max_finding_code_bytes=0,
            max_absence_queries=0,
            max_assessment_facts=0,
            max_port_facts=6,
            max_provenance_edges=6,
        ),
    )
    callback = _IdentityCallback()
    result = await (
        await start_execution(
            admitted=admitted,
            capabilities=(capability,),
            services=ExecutionServices(
                handles=(
                    ImplementationHandle(
                        implementation=capability.implementation,
                        operation=capability.operation,
                        configuration=capability.configuration,
                        local=callback,
                        transport=None,
                        resource=None,
                    ),
                ),
                context_resources=(),
                limits=ExecutionLimits(
                    max_local_in_flight=1,
                    max_remote_outstanding=0,
                    max_runtime_artifacts=artifact_limit,
                    max_runtime_artifact_bytes=32,
                    max_collection_items=2,
                ),
                decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
                clock=_Clock(),
            ),
        )
    ).wait()
    assert not any(type(key).__name__ == "RootInputKey" for fact in result.provenance for key in (fact.key,))
    if mode == "omitted":
        assert callback.calls == 0
        assert result.record.terminals[0].category == "blocked"
        assert result.record.terminals[0].attempt is None
        assert not result.final_outputs
        return
    assert callback.calls == 1
    expected_value = ("ab", "cd") if mode in {"collection", "nested", "substitution"} else "ab"
    actual_value = callback.observed[0]
    if isinstance(actual_value, TextCollectionValue):
        assert tuple(item.value.text for item in actual_value.items) == expected_value
    else:
        assert isinstance(actual_value, TextArtifactValue)
        assert actual_value.text == expected_value
    assert result.final_outputs, (
        tuple((item.category, item.reasons) for item in result.record.terminals),
        tuple((item.port, item.role) for item in result.ports),
        tuple(type(item.key).__name__ for item in result.provenance),
    )
    final = result.final_outputs[0]
    materialized = tuple(
        fact for fact in result.provenance if isinstance(fact.key, (BoundInputKey, InitialCollectionKey))
    )
    source_fact = next(fact for fact in materialized if mode == "scalar" or isinstance(fact.key, InitialCollectionKey))
    source_artifact = source_fact.artifact
    assert final.candidate.artifact == source_artifact
    assert (final.outcome, final.port) == ("ok", "output")
    assert isinstance(final.producer, OperationOutputKey)
    output_fact = next(
        fact for fact in result.provenance if isinstance(fact.key, OperationOutputKey) and fact.key == final.producer
    )
    assert output_fact.artifact == source_artifact
    if mode in {"nested", "substitution"}:
        inner_output = next(
            fact
            for fact in result.provenance
            if isinstance(fact.key, OperationOutputKey)
            and fact.key.activation
            == next(item.activation for item in result.ports if item.node == implementation_node)
            and fact.key.port == "output"
        )
        assert inner_output.parents == frozenset({source_fact.key})
        assert output_fact.parents == frozenset({inner_output.key})
    else:
        assert output_fact.parents == frozenset({source_fact.key})
    values = dict(result.artifacts)
    logical_bytes = sum(
        len(value.text.encode())
        if isinstance(value, TextArtifactValue)
        else sum(len(item.value.text.encode()) for item in value.items)
        for value in values.values()
    )
    if mode == "scalar":
        assert len(values) == len(materialized) == 1
        assert logical_bytes == 2
        assert not source_fact.parents
    else:
        bound = frozenset(fact.key for fact in materialized if isinstance(fact.key, BoundInputKey))
        assert len(values) == len(materialized) == 3
        assert logical_bytes == 8
        assert source_fact.parents == bound
        assert len(bound) == 2
    if same_key_versions:
        qualification = admit_qualification(
            execution=admitted, productions=(), limits=_qualification_limits(max_port_facts=32, max_provenance_edges=32)
        )
        latest_refs = {ref.key: ref for ref, _ in sorted(result.artifacts, key=lambda pair: pair[0].version)}
        current = evidence_revision_view(
            admitted=qualification,
            result=result,
            artifacts=tuple(latest_refs.values()),
            absences=(),
            configurations=(),
            state=admitted.context.prepared.state,
        )
        qualified = qualify(admitted=qualification, result=result, current=current, submissions=())
        assert all(status.qualification == "not_assessed" for status in qualified.record.statuses)
        assert all(target.withholding == frozenset({"execution_only"}) for target in qualified.targets)
    if same_key_versions:
        versioned = [fact for fact in materialized if isinstance(fact.key, BoundInputKey)]
        assert len({fact.artifact.key for fact in versioned}) == 1
        assert {fact.artifact.version for fact in versioned} == {1, 2}
        assert all(
            fact.artifact.version == fact.key.binding_artifact.version
            for fact in versioned
            if isinstance(fact.key, BoundInputKey)
        )
        assert source_artifact.key not in {fact.artifact.key for fact in versioned}
    input_fact = next(item for item in result.ports if item.node == implementation_node and item.port == "context")
    output_port_fact = next(item for item in result.ports if item.node == implementation_node and item.port == "output")
    assert input_fact.artifact == output_port_fact.artifact == source_artifact
    if mode in {"nested", "substitution"}:
        assert output_port_fact.role == "artifact"
        wrapper_output = next(item for item in result.ports if item.node == initial_node and item.port == "output")
        assert wrapper_output.artifact == source_artifact
        assert wrapper_output.role == "candidate"
        assert sum(isinstance(fact.key, InitialCollectionKey) for fact in result.provenance) == 1
    else:
        assert output_port_fact.role == "candidate"


@pytest.mark.parametrize("mode", ["collection", "nested", "substitution"])
def test_initial_materialization_retains_versions_of_one_runtime_key(mode: str) -> None:
    asyncio.run(_assert_context_execution(mode, same_key_versions=True, artifact_limit=3))


def test_initial_materialization_counts_versions_against_artifact_limit() -> None:
    with pytest.raises(EffectRejected) as rejected:
        asyncio.run(_assert_context_execution("collection", same_key_versions=True, artifact_limit=2))
    assert rejected.value.code.value == "limit_exceeded"
