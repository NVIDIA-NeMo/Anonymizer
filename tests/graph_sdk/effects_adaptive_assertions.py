# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adaptive materialization comparisons through the real executor."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

from anonymizer.engine.graph_sdk.context import (
    AdaptiveRetrievalDecl,
    ContextMaterialization,
    ContextResource,
    ContextSourceCapability,
    ContextSourceRef,
    RetrievalBounds,
    admit_context_plan,
)
from anonymizer.engine.graph_sdk.executor import (
    AssessmentLimits,
    DecisionLimits,
    ExecutionImplementation,
    ExecutionLimits,
    ExecutionServices,
    ImplementationHandle,
    OperationExecutionPolicy,
    OperationOutputKey,
    RootInputKey,
    RuntimeOutcome,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import BoundInput, PreparationConfiguration
from anonymizer.engine.graph_sdk.requests import (
    PhysicalRequestId,
    SemanticAssociation,
    TextArtifactValue,
)
from anonymizer.engine.graph_sdk.resources import ResourceLease
from anonymizer.graph.workflow import (
    ArtifactType,
    DynamicScope,
    OperationNode,
    OutputPort,
    admit_activation_workflow,
    admit_static_workflow,
)
from tests.graph_sdk.effects_admission_fixtures import _selector_data, _valid_runtime_rows
from tests.graph_sdk.effects_provider_fixtures import _CaseProvider, _LateAdaptiveProvider
from tests.graph_sdk.effects_request_fixtures import _normalize, _policy
from tests.graph_sdk.test_adaptive_executor import _adaptive_workflow, _Clock, _UnusedTransport
from tests.graph_sdk.test_preparation import _capability, _limits, _prepare


async def _assert_adaptive_materialization_case(case: dict[str, Any], *, late_mode: str | None = None) -> None:
    raw = cast(dict[str, Any], case["declaration"])
    materialization = cast(list[dict[str, Any]], raw["materializations"])[0]
    retrieval_requests = cast(int, raw["retrieval_bounds"]["max_requests"])
    workflow, node, item_type = _adaptive_workflow(requests=retrieval_requests)
    output_type = item_type
    if materialization["kind"] == "collection":
        output_type = ArtifactType(name="text_collection", revision=1)
        static = workflow.workflow
        operation_node = next(item for item in static.nodes if isinstance(item, OperationNode))
        operation = replace(
            operation_node.operation,
            outputs=(OutputPort(name="context", artifact_type=output_type),),
        )
        rebuilt = admit_static_workflow(
            workflow=static.workflow,
            interface=operation,
            nodes=(OperationNode(id=node, operation=operation),),
            input_bindings=tuple(static.input_bindings),
            output_bindings=tuple(static.output_bindings),
            outcome_bindings=tuple(static.outcome_bindings),
            sequence=tuple(static.sequence),
            choices=tuple(static.choices),
            protection=(),
            limits=static.limits,
        )
        workflow = admit_activation_workflow(
            workflow=rebuilt,
            scopes=(DynamicScope(workflow=rebuilt, maps=(), joins=(), loops=()),),
            limits=workflow.limits,
        )
    data = _selector_data()
    target = next(iter(data.targets))
    request_policy = _policy(cast(dict[str, Any], raw["policies"])["P0"])
    capability = replace(
        _capability(workflow, external=True),
        max_physical_requests_per_activation=request_policy.max_attempts,
    )
    prepared = _prepare(
        data=data,
        workflow=workflow,
        capability=capability,
        configuration=PreparationConfiguration(
            purpose="execution_only",
            required_protection_outcomes=frozenset(),
            hard_request_limit=cast(int, raw["hard_limit"]),
        ),
        bound_inputs=(BoundInput(target=target, source=target, port="input", artifact_type=item_type),),
        limits=_limits(capabilities=1),
    )
    source = ContextSourceRef(name="S0", revision=1)
    source_capability = ContextSourceCapability(
        source=source,
        artifact_type=item_type,
        uses=frozenset({"adaptive_retrieval"}),
        execution="async",
        resource_owner="caller",
        cancellation="cooperative_ack",
        settlement="explicit_ack",
        usage="exact",
        request=request_policy,
        safe_detachment="forbidden",
    )
    retrieval = AdaptiveRetrievalDecl(
        node=node,
        source=source,
        selector_ports=("input",),
        output_port="context",
        bounds=RetrievalBounds(
            max_items=cast(int, materialization["max_items"]),
            max_bytes=cast(int, materialization["max_bytes"]),
            max_requests=retrieval_requests,
        ),
        materialization=ContextMaterialization(
            kind=materialization["kind"],
            item_type=item_type,
        ),
    )
    context = admit_context_plan(
        prepared=prepared,
        bound_context=None,
        adaptive_retrievals=(retrieval,),
        context_capabilities=(source_capability,),
    )
    implementation = ExecutionImplementation(
        implementation=capability.implementation,
        configuration=capability.configuration,
        capability=capability,
        request=request_policy,
    )
    runtime_mappings = cast(list[dict[str, Any]], raw["runtime_mappings"])
    policy = OperationExecutionPolicy(
        node=node,
        kind="external",
        request=request_policy,
        safe_detachment="forbidden",
        implementations=(implementation,),
        result_outcomes=frozenset({"ok"}),
        runtime_outcomes=tuple(
            RuntimeOutcome(
                condition=row["condition"],
                reported_outcome=cast(str | None, row["reported_outcome"]),
                failure=row["failure"],
                outcome=cast(str | None, row["outcome"]),
                category=row["category"],
            )
            for row in runtime_mappings
        )
        if len(runtime_mappings) > 1
        else _valid_runtime_rows("external", frozenset({"ok"})),
    )
    materialization_limits = cast(dict[str, int], raw["materialization_limits"])
    admitted = admit_execution_plan(
        context=context,
        capabilities=(capability,),
        policies=(policy,),
        decisions=(),
        assessment_productions=(),
        assessment_limits=AssessmentLimits(
            max_productions=0,
            max_findings_per_production=0,
            max_finding_code_bytes=0,
            max_absence_queries=0,
            max_assessment_facts=0,
            max_port_facts=2,
            max_provenance_edges=materialization_limits["max_provenance_edges"],
        ),
    )
    events = [
        event
        for event in cast(list[dict[str, Any]], case["events"])
        if event["kind"] in {"materialize_result", "source_failure"}
    ]
    association_names: dict[object, str] = {}
    request_names: dict[PhysicalRequestId, str] = {}
    provider: _CaseProvider | _LateAdaptiveProvider
    if late_mode is None:
        provider = _CaseProvider(
            events=events,
            sources={"S0": source},
            association_names=association_names,
            request_names=request_names,
            fallback_associations=["A0"],
            default_source=source,
        )
    else:
        provider = _LateAdaptiveProvider(
            source=source,
            association_names=association_names,
            request_names=request_names,
            acknowledge_stop=late_mode == "cancelled",
        )
    transport = _UnusedTransport()
    running = await start_execution(
        admitted=admitted,
        capabilities=(capability,),
        services=ExecutionServices(
            handles=(
                ImplementationHandle(
                    implementation=capability.implementation,
                    operation=capability.operation,
                    configuration=capability.configuration,
                    local=None,
                    transport=transport,
                    resource=ResourceLease.create(owner="sdk", safe_detachment="forbidden", handle=transport),
                ),
            ),
            context_resources=(
                ContextResource(
                    source=source,
                    capability=source_capability,
                    lease=ResourceLease.create(owner="caller", safe_detachment="forbidden", handle=provider),
                    factory=None,
                ),
            ),
            limits=ExecutionLimits(
                max_local_in_flight=0,
                max_remote_outstanding=1,
                max_runtime_artifacts=materialization_limits["max_artifacts"],
                max_runtime_artifact_bytes=materialization_limits["max_artifact_bytes"],
                max_collection_items=materialization_limits["max_collection_items"],
            ),
            decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
            clock=_Clock(),
        ),
    )
    if isinstance(provider, _LateAdaptiveProvider):
        await provider.started.wait()
        running.request_cancel()
    result = await running.wait()
    if isinstance(provider, _LateAdaptiveProvider):
        assert provider.returned_after_cancel
    expected = cast(dict[str, Any], case["expected"])["state"]
    expected_task = expected["tasks"].get("A0", late_mode)
    assert result.record.terminals[0].category == expected_task, (
        [(entry.status, entry.outcome) for entry in result.states[0].entries],
        [(item.category, item.failure) for item in result.requests.terminals],
    )
    request_actual = _normalize(
        result.requests,
        cast(Any, {name: association for association, name in association_names.items()}),
        {name: request for request, name in request_names.items()},
        {"P0": request_policy},
    )
    dispatch_by_request = {item.request: item for item in result.requests.dispatches}
    request_actual["association_terminals"] = {
        association_names[returned.association]: {
            "request": request_names[terminal.request],
            "outcome": returned.outcome,
            "policy": "P0",
        }
        for terminal in result.requests.terminals
        if terminal.request in dispatch_by_request
        for returned in terminal.results
    } | cast(dict[str, object], request_actual["association_terminals"])
    actual_task_requests: dict[str, str] = {}
    if any(event["kind"] == "bridge_start" for event in cast(list[dict[str, Any]], case["events"])):
        for dispatch in result.requests.dispatches:
            for association in dispatch.associations:
                if not isinstance(association, SemanticAssociation):
                    continue
                name = association_names[association]
                terminal = next(
                    item for item in result.record.terminals if item.activation == association.task.activation
                )
                assert terminal.category == expected["tasks"][name]
                actual_task_requests[name] = request_names[dispatch.request]
    request_actual["task_requests"] = actual_task_requests
    for key in (
        "bindings",
        "dispatched",
        "dispatched_count",
        "denials",
        "terminals",
        "request_facts",
        "request_failures",
        "settlements",
        "association_terminals",
        "association_requests",
        "task_requests",
        "request_associations",
        "request_policies",
        "reservations",
        "reservation_policies",
        "local_in_flight",
        "remote_outstanding",
        "cancel_requested",
        "defects",
        "attempts",
    ):
        assert request_actual[key] == expected[key], (case["case_id"], key)
    values = dict(result.artifacts)
    if late_mode is not None:
        assert not any(isinstance(item.key, OperationOutputKey) for item in result.provenance)
        assert not any(item.port == "context" and item.node == node for item in result.ports)
        assert not result.final_outputs
    identities: dict[object, str] = {}
    actual_artifacts: list[dict[str, Any]] = []
    for fact in result.provenance:
        if isinstance(fact.key, RootInputKey):
            identity = f"RootInputKey:T0:{fact.key.port}"
            artifact_name = item_type.name
        elif isinstance(fact.key, OperationOutputKey):
            identity = f"OperationOutputKey:A0:T0:{fact.key.port}"
            artifact_name = output_type.name
        else:
            continue
        identities[fact.key] = identity
        value = values[fact.artifact]
        actual_artifacts.append(
            {
                "identity": identity,
                "artifact_type": artifact_name,
                "value": (
                    value.text
                    if isinstance(value, TextArtifactValue)
                    else [{"key": item.key, "version": item.version, "value": item.value.text} for item in value.items]
                ),
            }
        )
    assert sorted(actual_artifacts, key=lambda item: item["identity"]) == sorted(
        expected["artifacts"], key=lambda item: item["identity"]
    )
    materialization_ports: dict[str, object] = {}
    for fact in result.provenance:
        if not isinstance(fact.key, OperationOutputKey):
            continue
        value = values[fact.artifact]
        materialization_ports[f"adaptive:T0:N0:{fact.key.port}:None"] = {
            "artifact_type": output_type.name,
            "key": identities[fact.key],
            "value": (
                value.text
                if isinstance(value, TextArtifactValue)
                else [{"key": item.key, "version": item.version, "value": item.value.text} for item in value.items]
            ),
        }
    materialization_actual = {
        "artifact_bytes": sum(
            len(value.text.encode())
            if isinstance(value, TextArtifactValue)
            else sum(len(item.value.text.encode()) for item in value.items)
            for value in values.values()
        ),
        "artifact_count": len(values),
        "ports": materialization_ports,
        "provenance": {
            identities[fact.key]: sorted(identities[parent] for parent in fact.parents)
            for fact in result.provenance
            if fact.key in identities
        },
        "provenance_edges": sum(len(fact.parents) for fact in result.provenance),
    }
    assert materialization_actual == expected["materialization"]
    root_fact = next(fact for fact in result.provenance if isinstance(fact.key, RootInputKey))
    output_facts = [fact for fact in result.provenance if isinstance(fact.key, OperationOutputKey)]
    input_port = next(fact for fact in result.ports if fact.port == "input")
    assert (
        input_port.node,
        input_port.target,
        input_port.artifact,
        input_port.artifact_type,
        input_port.role,
    ) == (node, target, root_fact.artifact, item_type, "artifact")
    if output_facts:
        assert len(output_facts) == 1
        output_fact = output_facts[0]
        output_port = next(fact for fact in result.ports if fact.port == "context")
        assert len(result.ports) == 2
        assert (
            output_port.node,
            output_port.target,
            output_port.artifact,
            output_port.artifact_type,
            output_port.role,
        ) == (node, target, output_fact.artifact, output_type, "candidate")
        assert len(result.final_outputs) == 1
        final = result.final_outputs[0]
        assert (final.target, final.outcome, final.port, final.candidate.target) == (
            target,
            "ok",
            "context",
            target,
        )
        assert final.candidate.artifact == output_fact.artifact
        assert final.producer == output_fact.key
    else:
        assert len(result.ports) == 1
        assert not result.final_outputs
