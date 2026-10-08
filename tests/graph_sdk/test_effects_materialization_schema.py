# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real binding and execution for context schema unions and distinct sites."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from typing import Any

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectRejected
from anonymizer.engine.graph_sdk.binding import start_initial_binding
from anonymizer.engine.graph_sdk.capabilities import (
    ImplementationCapability,
    ImplementationRef,
    ImplementationSelection,
)
from anonymizer.engine.graph_sdk.context import (
    AdaptiveRetrievalDecl,
    BindingLimits,
    BindingResult,
    ContextMaterialization,
    ContextResource,
    ContextSelector,
    ContextSourceCapability,
    ContextSourceRef,
    InitialContextDecl,
    RetrievalBounds,
    SourceItem,
    SourceResponse,
    admit_context_plan,
)
from anonymizer.engine.graph_sdk.data import ValidatedDataGraph
from anonymizer.engine.graph_sdk.executor import (
    BoundInputKey,
    DecisionLimits,
    ExecutionImplementation,
    ExecutionLimits,
    ExecutionResult,
    ExecutionServices,
    ImplementationHandle,
    InitialCollectionKey,
    LocalCompleted,
    OperationExecutionPolicy,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import (
    BoundInput,
    PreparationConfiguration,
    PreparedPlan,
    StateRevisionView,
    prepare,
)
from anonymizer.engine.graph_sdk.requests import (
    AssociationInput,
    BindingAssociation,
    BindingId,
    ExactUsage,
    ExternalSettlement,
    PhysicalRequestId,
    PhysicalRequestPolicy,
    StopConfirmed,
    TextArtifactValue,
    TextCollectionValue,
)
from anonymizer.engine.graph_sdk.resources import ResourceLease
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
    OperationNode,
    OperationSpec,
    OutcomeBinding,
    OutcomeSpec,
    OutputDependency,
    OutputPort,
    ResourceCeiling,
    SequenceEdge,
    WorkflowId,
    WorkflowInputRef,
    WorkflowLimits,
    WorkflowOutcomeRef,
    admit_activation_workflow,
    admit_static_workflow,
)
from tests.graph_sdk.test_effects_production_conformance import (
    CORPUS,
    _assessment_limits,
    _ContextConsumer,
    _normalize,
    _valid_runtime_rows,
    _ZeroClock,
)
from tests.graph_sdk.test_preparation import _capability, _data, _limits


@dataclass
class _Provider:
    source: ContextSourceRef
    calls: int = 0

    async def retrieve(
        self,
        *,
        request: PhysicalRequestId,
        association: BindingAssociation,
        selector: ContextSelector,
        bounds: RetrievalBounds,
    ) -> SourceResponse:
        del selector, bounds
        value = ("a", "b")[self.calls]
        self.calls += 1
        return SourceResponse(
            source=self.source,
            items=(SourceItem(association=association, key=0, version=1, text=value),),
            settlement=ExternalSettlement(
                request=request,
                disposition="completed",
                remote_stopped=True,
                usage=ExactUsage(input_units=0, output_units=0),
            ),
        )

    async def cancel(self, request: PhysicalRequestId) -> StopConfirmed:
        return StopConfirmed(usage=ExactUsage(input_units=0, output_units=0))


@dataclass(kw_only=True)
class _SiteConsumer(_ContextConsumer):
    expected_text: str

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        assert len(request) == 1 and len(request[0].inputs) == 1
        value = request[0].inputs[0].value
        assert isinstance(value, TextCollectionValue)
        assert [(item.key, item.version, item.value.text) for item in value.items] == [(0, 1, self.expected_text)]
        return await super().run(request)


@pytest.mark.parametrize(
    "case_id",
    [
        "materialization/conflicting_schema",
        "materialization/nested_collection_schema",
        "materialization/same_port_distinct_nodes",
    ],
)
def test_materialization_schema_and_sites(case_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    allocations = []
    original = BindingId.new

    def counted(cls: type[BindingId]) -> BindingId:
        identity = original()
        allocations.append(identity)
        return identity

    monkeypatch.setattr(BindingId, "new", classmethod(counted))
    asyncio.run(_assert_schema_case(case_id, allocations))


@dataclass
class _SchemaFixture:
    case: dict[str, Any]
    workflow: AdmittedActivationWorkflow
    data: ValidatedDataGraph
    nodes: tuple[NodeId, ...]
    types: dict[str, ArtifactType]
    policy: PhysicalRequestPolicy
    source_caps: tuple[ContextSourceCapability, ...]
    initial_caps: tuple[ContextSourceCapability, ...]
    providers: dict[ContextSourceRef, _Provider]
    declarations: tuple[InitialContextDecl, ...]
    capabilities: tuple[ImplementationCapability, ...]


async def _assert_schema_case(case_id: str, allocations: list[BindingId]) -> None:
    case = next(item for item in json.loads(CORPUS.read_bytes()) if item["case_id"] == case_id)
    fixture = _build_schema_fixture(case)
    if case_id == "materialization/conflicting_schema":
        # Distinct sites isolate schema union from duplicate-site precedence.
        with pytest.raises(EffectRejected) as rejected:
            await _bind_schema_fixture(fixture)
        assert rejected.value.code.value == case["expected"]["code"] == "contradictory"
        assert not allocations and all(provider.calls == 0 for provider in fixture.providers.values())
        return
    bound = await _bind_schema_fixture(fixture)
    assert bound.context is not None and len(allocations) == 1
    prepared = _prepare_schema_fixture(fixture)
    if case_id == "materialization/nested_collection_schema":
        _assert_nested_rejection(fixture, bound, prepared)
        return
    result = await _execute_distinct_sites(fixture, bound, prepared)
    _assert_distinct_site_result(fixture, bound, result)


def _build_schema_fixture(case: dict[str, Any]) -> _SchemaFixture:
    raw = case["declaration"]["materializations"]
    nested = case["case_id"] == "materialization/nested_collection_schema"
    conflict = case["case_id"] == "materialization/conflicting_schema"
    types = {
        name: ArtifactType(name=name, revision=1)
        for name in {"text", *(item["item_type"] for item in raw), *(item["output_type"] for item in raw)}
    }
    owner = WorkflowId.new()
    nodes = tuple(NodeId.new(workflow=owner) for _ in raw)
    operations = []
    for index, item in enumerate(raw):
        adaptive = nested and index == 1
        input_type = types["text"] if adaptive else types[item["output_type"]]
        operations.append(
            OperationSpec(
                name=f"context-{index}",
                inputs=(InputPort(name="input" if adaptive else "context", artifact_type=input_type),),
                outputs=(OutputPort(name="context", artifact_type=types[item["output_type"]]),) if adaptive else (),
                output_dependencies=(
                    OutputDependency(output="context", inputs=frozenset({"input"}), identity_input=None),
                )
                if adaptive
                else (),
                outcomes=(
                    OutcomeSpec(
                        name="ok",
                        category="success",
                        produced_ports=frozenset({"context"}) if adaptive else frozenset(),
                        context=frozenset()
                        if adaptive
                        else frozenset({ContextUse(port="context", meaning="retrieved", capture="whole_artifact")}),
                        evidence=frozenset(),
                        state_effects=frozenset(),
                        model_requirements=frozenset(),
                        ceiling=ResourceCeiling(
                            max_activations=1,
                            max_model_requests=2 if adaptive else 0,
                            max_input_bytes=12,
                            max_output_bytes=12,
                        ),
                    ),
                ),
            )
        )
    interface = replace(
        operations[0],
        name="two-context-sites",
        inputs=tuple(
            InputPort(name=f"root-{index}", artifact_type=operation.inputs[0].artifact_type)
            for index, operation in enumerate(operations)
        ),
        outcomes=(
            replace(
                operations[0].outcomes[0],
                context=frozenset(
                    ContextUse(port=f"root-{index}", meaning="retrieved", capture="whole_artifact")
                    for index in range(1 if nested else 2)
                ),
                ceiling=ResourceCeiling(
                    max_activations=2, max_model_requests=2 if nested else 0, max_input_bytes=24, max_output_bytes=24
                ),
            ),
        ),
    )
    static = admit_static_workflow(
        workflow=owner,
        interface=interface,
        nodes=tuple(
            OperationNode(id=node, operation=operation) for node, operation in zip(nodes, operations, strict=True)
        ),
        input_bindings=tuple(
            InputBinding(
                source=WorkflowInputRef(port=f"root-{index}")
                if nested and index == 1
                else ContextInputRef(port=f"root-{index}"),
                destination=NodeInputRef(node=node, port=operation.inputs[0].name),
            )
            for index, (node, operation) in enumerate(zip(nodes, operations, strict=True))
        ),
        output_bindings=(),
        outcome_bindings=(
            OutcomeBinding(
                source=NodeOutcomeRef(node=nodes[1], outcome="ok"), destination=WorkflowOutcomeRef(outcome="ok")
            ),
        ),
        sequence=(SequenceEdge(before=nodes[0], after=nodes[1]),),
        choices=(),
        protection=(),
        limits=WorkflowLimits(
            max_nodes=2,
            max_bindings=3,
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
    data = _data(1)
    target = next(iter(data.targets))
    policy = PhysicalRequestPolicy(
        visibility="dispatch_and_settlement",
        pre_dispatch_control="executor",
        retry_owner="executor",
        replay="idempotent",
        max_attempts=2,
    )
    sources = tuple(ContextSourceRef(name=f"S{index if conflict or nested else 0}", revision=1) for index in range(2))
    source_caps = tuple(
        ContextSourceCapability(
            source=source,
            artifact_type=types[item["item_type"]],
            uses=frozenset({"adaptive_retrieval" if nested and index == 1 else "initial_binding"}),
            execution="async",
            resource_owner="caller",
            cancellation="cooperative_ack",
            settlement="explicit_ack",
            usage="exact",
            request=policy,
            safe_detachment="forbidden",
        )
        for index, (source, item) in enumerate(zip(sources, raw, strict=True))
    )
    initial_caps = tuple(dict.fromkeys(source_caps[:1] if nested else source_caps))
    providers = {cap.source: _Provider(source=cap.source) for cap in initial_caps}
    declarations = tuple(
        InitialContextDecl(
            target=target,
            node=node,
            port="context",
            artifact_type=types[item["output_type"]],
            source=source,
            selector=ContextSelector(fields=()),
            requirement="required",
            bounds=RetrievalBounds(max_items=item["max_items"], max_bytes=item["max_bytes"], max_requests=2),
            materialization=ContextMaterialization(kind=item["kind"], item_type=types[item["item_type"]]),
        )
        for index, (node, source, item) in enumerate(zip(nodes, sources, raw, strict=True))
        if not nested or index == 0
    )
    capabilities = tuple(
        replace(
            _capability(workflow, external=nested and index == 1),
            operation=operation,
            implementation=ImplementationRef(name=f"implementation-{index}", revision=1),
            max_physical_requests_per_activation=2 if nested and index == 1 else 0,
        )
        for index, operation in enumerate(operations)
    )
    return _SchemaFixture(
        case, workflow, data, nodes, types, policy, source_caps, initial_caps, providers, declarations, capabilities
    )


async def _bind_schema_fixture(fixture: _SchemaFixture) -> BindingResult:
    data, workflow, declarations = fixture.data, fixture.workflow, fixture.declarations
    initial_caps, providers = fixture.initial_caps, fixture.providers
    binding = await start_initial_binding(
        data=data,
        workflow=workflow,
        declarations=declarations,
        capabilities=initial_caps,
        resources=tuple(
            ContextResource(
                source=cap.source,
                capability=cap,
                lease=ResourceLease.create(owner="caller", safe_detachment="forbidden", handle=providers[cap.source]),
                factory=None,
            )
            for cap in initial_caps
        ),
        limits=BindingLimits(
            max_declarations=4,
            max_sources=2,
            max_capabilities=2,
            max_selector_fields=0,
            max_selector_bytes=0,
            max_items=3,
            max_bytes=12,
            max_requests=2,
            max_resources=2,
        ),
    )
    return await binding.wait()


def _prepare_schema_fixture(fixture: _SchemaFixture) -> PreparedPlan:
    data, workflow, nodes, types = fixture.data, fixture.workflow, fixture.nodes, fixture.types
    capabilities = fixture.capabilities
    target = next(iter(data.targets))
    nested = fixture.case["case_id"] == "materialization/nested_collection_schema"
    return prepare(
        data=data,
        workflow=workflow,
        activation_limits=ActivationLimits(max_events=6, max_entries=2, max_parent_depth=1),
        bound_inputs=(BoundInput(target=target, source=target, port="root-1", artifact_type=types["text"]),)
        if nested
        else (),
        configuration=PreparationConfiguration(
            purpose="execution_only", required_protection_outcomes=frozenset(), hard_request_limit=None
        ),
        state=StateRevisionView(revisions=frozenset()),
        selections=tuple(
            ImplementationSelection(node=node, implementation=cap.implementation, configuration=cap.configuration)
            for node, cap in zip(nodes, capabilities, strict=True)
        ),
        capabilities=capabilities,
        limits=_limits(capabilities=2, slots=2),
    )


def _assert_nested_rejection(fixture: _SchemaFixture, bound: BindingResult, prepared: PreparedPlan) -> None:
    case, nodes, types = fixture.case, fixture.nodes, fixture.types
    raw = case["declaration"]["materializations"]
    source_caps, providers = fixture.source_caps, fixture.providers
    sources = tuple(cap.source for cap in source_caps)
    adaptive = (
        AdaptiveRetrievalDecl(
            node=nodes[1],
            source=sources[1],
            selector_ports=("input",),
            output_port="context",
            bounds=RetrievalBounds(max_items=raw[1]["max_items"], max_bytes=raw[1]["max_bytes"], max_requests=2),
            materialization=ContextMaterialization(kind=raw[1]["kind"], item_type=types[raw[1]["item_type"]]),
        ),
    )
    with pytest.raises(EffectRejected) as rejected:
        admit_context_plan(
            prepared=prepared,
            bound_context=bound.context,
            adaptive_retrievals=adaptive,
            context_capabilities=(source_caps[1],),
        )
    assert rejected.value.code.value == case["expected"]["code"] == "contradictory"
    assert sum(provider.calls for provider in providers.values()) == 1


async def _execute_distinct_sites(
    fixture: _SchemaFixture, bound: BindingResult, prepared: PreparedPlan
) -> ExecutionResult:
    case, nodes, capabilities = fixture.case, fixture.nodes, fixture.capabilities
    context = admit_context_plan(
        prepared=prepared, bound_context=bound.context, adaptive_retrievals=(), context_capabilities=()
    )
    admitted = admit_execution_plan(
        context=context,
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
                runtime_outcomes=_valid_runtime_rows("local", frozenset({"ok"})),
            )
            for node, cap in zip(nodes, capabilities, strict=True)
        ),
        decisions=(),
        assessment_productions=(),
        assessment_limits=_assessment_limits(),
    )
    consumers = tuple(_SiteConsumer(port="context", expected_text=text) for text in ("a", "b"))
    limits = case["declaration"]["materialization_limits"]
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
                        local=consumer,
                        transport=None,
                        resource=None,
                    )
                    for cap, consumer in zip(capabilities, consumers, strict=True)
                ),
                context_resources=(),
                limits=ExecutionLimits(
                    max_local_in_flight=1,
                    max_remote_outstanding=0,
                    max_runtime_artifacts=limits["max_artifacts"],
                    max_runtime_artifact_bytes=limits["max_artifact_bytes"],
                    max_collection_items=limits["max_collection_items"],
                ),
                decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
                clock=_ZeroClock(),
            ),
        )
    ).wait()
    assert [consumer.calls for consumer in consumers] == [1, 1]
    assert all(state.complete for state in result.states)
    assert all(terminal.category == "success" for terminal in result.record.terminals)
    return result


def _assert_distinct_site_result(fixture: _SchemaFixture, bound: BindingResult, result: ExecutionResult) -> None:
    case, nodes, policy = fixture.case, fixture.nodes, fixture.policy
    expected = case["expected"]["state"]
    bound_artifacts = {item.reference: item for item in bound.receipt.artifacts}
    names = {node: f"N{index}" for index, node in enumerate(nodes)}
    declarations_by_id = {fact.identity: f"D{index}" for index, fact in enumerate(bound.receipt.sources)}
    values = dict(result.artifacts)
    identities: dict[object, str] = {}
    artifacts = []
    for fact in result.provenance:
        key = fact.key
        if isinstance(key, BoundInputKey):
            identity = f"BoundInputKey:T0:{names[key.node]}:context:{declarations_by_id[key.binding_artifact.declaration]}:{key.binding_artifact.key}:{key.binding_artifact.version}"
            value = values[fact.artifact]
            assert isinstance(value, TextArtifactValue)
            artifacts.append(
                {
                    "identity": identity,
                    "artifact_type": "text",
                    "source": bound_artifacts[key.binding_artifact].source.name,
                    "text": value.text,
                }
            )
        else:
            assert isinstance(key, InitialCollectionKey)
            identity = f"InitialCollectionKey:T0:{names[key.node]}:context:{declarations_by_id[key.declaration]}"
            value = values[fact.artifact]
            assert isinstance(value, TextCollectionValue)
            artifacts.append(
                {
                    "identity": identity,
                    "artifact_type": "text_collection",
                    "value": [
                        {"key": item.key, "version": item.version, "value": item.value.text} for item in value.items
                    ],
                }
            )
        identities[key] = identity
    assert sorted(artifacts, key=lambda item: item["identity"]) == sorted(
        expected["artifacts"], key=lambda item: item["identity"]
    )
    assert {
        identities[fact.key]: sorted(identities[parent] for parent in fact.parents) for fact in result.provenance
    } == expected["materialization"]["provenance"]
    assert len(result.artifacts) == expected["materialization"]["artifact_count"]
    assert (
        sum(
            len(value.text.encode())
            if isinstance(value, TextArtifactValue)
            else sum(len(item.value.text.encode()) for item in value.items)
            for value in values.values()
        )
        == expected["materialization"]["artifact_bytes"]
    )
    assert sum(len(fact.parents) for fact in result.provenance) == expected["materialization"]["provenance_edges"]
    assert len(result.ports) == len(expected["materialization"]["ports"]) == 2
    assert {fact.node for fact in result.ports} == set(nodes)
    artifact_facts = {fact.artifact: fact for fact in result.provenance}
    artifact_rows = {item["identity"]: item for item in artifacts}
    actual_ports = {}
    for fact in result.ports:
        key = artifact_facts[fact.artifact].key
        assert isinstance(key, InitialCollectionKey)
        identity = identities[key]
        actual_ports[f"initial:T0:{names[fact.node]}:{fact.port}:{declarations_by_id[key.declaration]}"] = {
            "artifact_type": fact.artifact_type.name,
            "key": identity,
            "value": artifact_rows[identity]["value"],
        }
    assert actual_ports == expected["materialization"]["ports"]
    assert not result.final_outputs and not result.assessments
    request_names = {f"R{index}": dispatch.request for index, dispatch in enumerate(bound.receipt.requests.dispatches)}
    tasks = {name: BindingAssociation(declaration=identity) for identity, name in declarations_by_id.items()}
    normalized = _normalize(bound.receipt.requests, tasks, request_names, {"P0": policy})
    association_names: dict[object, str] = {association: name for name, association in tasks.items()}
    request_labels = {request: name for name, request in request_names.items()}
    normalized["association_terminals"] = {
        association_names[row.association]: {
            "request": request_labels[terminal.request],
            "outcome": row.outcome,
            "policy": "P0",
        }
        for terminal in bound.receipt.requests.terminals
        for row in terminal.results
    }
    for key, value in normalized.items():
        assert value == expected[key], key
    assert bound.receipt.terminal == expected["binding_terminal"]
    assert {declarations_by_id[fact.identity]: fact.terminal for fact in bound.receipt.sources} == expected[
        "binding_sources"
    ]
    assert {
        declarations_by_id[fact.identity]: fact.declaration.source.name for fact in bound.receipt.sources
    } == expected["binding_declarations"]
