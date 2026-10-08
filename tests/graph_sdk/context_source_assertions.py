# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution and storage assertions for context source fixtures."""

from __future__ import annotations

import asyncio
from dataclasses import replace
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
    InitialContextDecl,
    RetrievalBounds,
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
    OperationExecutionPolicy,
    OperationOutputKey,
    RuntimeOutcome,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import BoundInput, PreparationConfiguration, StateRevisionView, prepare
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.requests import (
    BindingId,
    PhysicalRequestPolicy,
    TextArtifactValue,
    TextCollectionValue,
)
from anonymizer.engine.graph_sdk.resources import ResourceLease
from anonymizer.graph.activation import ActivationLimits
from anonymizer.graph.workflow import (
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
    OutcomeSpec,
    OutputBinding,
    OutputDependency,
    OutputPort,
    ResourceCeiling,
    SubgraphNode,
    WorkflowId,
    WorkflowInputRef,
    WorkflowLimits,
    WorkflowOutcomeRef,
    WorkflowOutputRef,
    admit_activation_workflow,
    admit_static_workflow,
)
from tests.graph_sdk.context_source_fixtures import (
    CONTEXT_CASES,
    SOURCE,
    _ContextProvider,
    _FixtureCallback,
    _IdentityCallback,
    _nested_workflow,
    _ProviderFactory,
    _single_workflow,
    _static_fixture,
)
from tests.graph_sdk.evidence_fixtures import _qualification_limits
from tests.graph_sdk.test_local_executor import _Clock, _runtime_rows
from tests.graph_sdk.test_preparation import _capability, _data, _limits


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
