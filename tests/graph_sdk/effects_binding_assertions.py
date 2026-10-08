# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Initial binding and execution projection comparisons."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
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
    admit_context_plan,
)
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
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.requests import (
    BindingDeclarationId,
    PhysicalRequestId,
    PortArtifact,
    TextArtifactValue,
)
from anonymizer.engine.graph_sdk.resources import ResourceLease
from anonymizer.graph.workflow import (
    ArtifactType,
    ContextInputRef,
    DynamicScope,
    InputPort,
    NodeId,
    NodeOutputRef,
    OperationNode,
    OutputBinding,
    OutputDependency,
    OutputPort,
    WorkflowOutputRef,
    admit_activation_workflow,
    admit_static_workflow,
)
from tests.graph_sdk.effects_admission_fixtures import _valid_runtime_rows
from tests.graph_sdk.effects_late_binding_assertions import (
    _assert_latest_publication,
)
from tests.graph_sdk.effects_provider_fixtures import (
    _CaseProvider,
    _CloseFailureProvider,
    _ContextConsumer,
    _ProviderWithoutClose,
)
from tests.graph_sdk.effects_request_fixtures import _normalize, _policy
from tests.graph_sdk.test_adaptive_executor import _Clock
from tests.graph_sdk.test_binding import _context_workflow
from tests.graph_sdk.test_preparation import _capability, _data, _limits, _prepare


async def _assert_binding_corpus_case(case: dict[str, Any]) -> None:
    raw = cast(dict[str, Any], case["declaration"])
    event_values = cast(list[dict[str, Any]], case["events"])
    workflow, node, artifact_type = _context_workflow()
    max_response_items = max(
        1,
        max(
            (
                len(cast(list[object], event.get("items", [])))
                for event in event_values
                if event["kind"] in {"source_result", "materialize_result"}
            ),
            default=1,
        ),
    )
    materializations = cast(list[dict[str, Any]], raw.get("materializations", []))
    publication = materializations[0].get("publication") if materializations else None
    latest = bool(materializations and materializations[0].get("version_selection") == "latest")
    materialization_kind = cast(str, materializations[0]["kind"]) if materializations else None
    declared_max_items = cast(int, materializations[0]["max_items"]) if materializations else max_response_items
    item_type = artifact_type
    context_port = cast(str, materializations[0]["port"]) if materializations else "input"
    if materializations:
        input_type = (
            ArtifactType(name="text_collection", revision=1) if materialization_kind == "collection" else item_type
        )
        static = workflow.workflow
        operation_node = next(item for item in static.nodes if isinstance(item, OperationNode))
        operation = replace(
            operation_node.operation,
            inputs=(InputPort(name=context_port, artifact_type=input_type),),
            outcomes=tuple(
                replace(
                    outcome,
                    context=frozenset(replace(use, port=context_port) for use in outcome.context),
                )
                for outcome in operation_node.operation.outcomes
            ),
        )
        if publication is not None:
            assert materializations[0]["output_type"] == item_type.name
            assert publication["artifact_type"] == item_type.name
            assert publication["activation"] == f"OP:{materializations[0]['association']}"
            assert publication["node"] == materializations[0]["node"] == "N0"
            (published_outcome,) = operation.outcomes
            assert publication["outcome"] == published_outcome.name == "ok"
            operation = replace(
                operation,
                outputs=(OutputPort(name=publication["output_port"], artifact_type=item_type),),
                output_dependencies=(
                    OutputDependency(
                        output=publication["output_port"],
                        inputs=frozenset(publication["inputs"]),
                        identity_input=None,
                    ),
                ),
                outcomes=tuple(
                    replace(outcome, produced_ports=frozenset({publication["output_port"]}))
                    for outcome in operation.outcomes
                ),
            )
        rebuilt = admit_static_workflow(
            workflow=static.workflow,
            interface=replace(operation, inputs=(InputPort(name=context_port, artifact_type=input_type),)),
            nodes=(OperationNode(id=node, operation=operation),),
            input_bindings=tuple(
                replace(
                    binding,
                    source=ContextInputRef(port=context_port),
                    destination=replace(binding.destination, port=context_port),
                )
                for binding in static.input_bindings
            ),
            output_bindings=(
                OutputBinding(
                    source=NodeOutputRef(node=node, port=publication["output_port"]),
                    destination=WorkflowOutputRef(port=publication["output_port"]),
                ),
            )
            if publication is not None
            else tuple(static.output_bindings),
            outcome_bindings=tuple(static.outcome_bindings),
            sequence=tuple(static.sequence),
            choices=tuple(static.choices),
            protection=(),
            limits=replace(static.limits, max_bindings=static.limits.max_bindings + int(publication is not None)),
        )
        workflow = admit_activation_workflow(
            workflow=rebuilt,
            scopes=(DynamicScope(workflow=rebuilt, maps=(), joins=(), loops=()),),
            limits=workflow.limits,
        )
        artifact_type = input_type
    elif materialization_kind is None and max_response_items > 1:
        artifact_type = ArtifactType(name="text_collection", revision=1)
        static = workflow.workflow
        operation_node = next(item for item in static.nodes if isinstance(item, OperationNode))
        operation = replace(
            operation_node.operation,
            inputs=(InputPort(name="input", artifact_type=artifact_type),),
        )
        rebuilt = admit_static_workflow(
            workflow=static.workflow,
            interface=replace(static.interface, inputs=(InputPort(name="input", artifact_type=artifact_type),)),
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
    binding_sources = cast(dict[str, str], raw["binding_declarations"])
    order = list(dict.fromkeys(event["association"] for event in event_values if event["kind"] == "bind_policy"))
    assert set(order) == set(binding_sources)
    data = _data(len(order))
    targets = sorted(data.targets, key=repr)
    result_sources = {
        cast(str, event["source"])
        for event in event_values
        if event["kind"] in {"source_result", "source_failure"} and "source" in event
    }
    source_refs = {
        name: ContextSourceRef(name=name, revision=1) for name in sorted(set(binding_sources.values()) | result_sources)
    }
    policies = {name: _policy(value) for name, value in cast(dict[str, dict[str, Any]], raw["policies"]).items()}
    response_events: dict[str, list[dict[str, Any]]] = {name: [] for name in source_refs}
    for event in event_values:
        if event["kind"] not in {"source_result", "source_failure", "materialize_result"}:
            continue
        source_name = cast(str | None, event.get("source"))
        if source_name is None:
            source_name = binding_sources[cast(str, event["association"])]
        response_events[source_name].append(event)
    if case["case_id"] == "binding/wrong_source":
        declared_source = next(iter(binding_sources.values()))
        response_events[declared_source] = [
            event for event in event_values if event["kind"] in {"source_result", "source_failure"}
        ]
    association_names: dict[object, str] = {}
    request_names: dict[PhysicalRequestId, str] = {}
    cleanup_dispositions = {
        event["resource"]: event["disposition"] for event in event_values if event["kind"] == "binding_cleanup"
    }
    provider_class = _CloseFailureProvider if "close_failed" in cleanup_dispositions.values() else _CaseProvider
    providers = {
        name: provider_class(
            events=response_events[name],
            sources=source_refs,
            association_names=association_names,
            request_names=request_names,
            fallback_associations=[label for label in order if binding_sources[label] == name],
            default_source=source_refs[name],
        )
        for name in source_refs
    }
    declaration_limits = cast(dict[str, int], raw["binding_limits"])
    requirements = cast(dict[str, str], raw.get("binding_requirements", {}))
    owners = {
        binding_sources[event["association"]]: event["owner"]
        for event in event_values
        if event["kind"] == "binding_cleanup_association"
    }
    declarations = tuple(
        InitialContextDecl(
            target=target,
            node=node,
            port=context_port,
            artifact_type=artifact_type,
            source=source_refs[binding_sources[label]],
            selector=ContextSelector(fields=()),
            requirement=cast(Any, requirements.get(label, "required")),
            bounds=RetrievalBounds(
                max_items=declared_max_items,
                max_bytes=max(1, declaration_limits["max_bytes"]),
                max_requests=policies["P0"].max_attempts,
            ),
            version_selection=materializations[0].get("version_selection", "exact_one")
            if materializations
            else "exact_one",
            materialization=ContextMaterialization(
                kind=cast(Any, materialization_kind or ("single" if max_response_items == 1 else "collection")),
                item_type=item_type,
            ),
        )
        for label, target in zip(order, targets, strict=True)
    )
    declared_source_names = set(binding_sources.values())
    capabilities = tuple(
        ContextSourceCapability(
            source=source_refs[name],
            artifact_type=item_type,
            uses=frozenset({"initial_binding"}),
            execution="async",
            resource_owner=owners.get(name, "caller"),
            cancellation="cooperative_ack",
            settlement="explicit_ack",
            usage="exact",
            request=policies["P0"],
            safe_detachment="forbidden",
        )
        for name in source_refs
        if name in declared_source_names
    )
    handles = {
        name: _ProviderWithoutClose(provider)
        if any(
            cleanup_dispositions.get(f"Q:{label}") == "close_unknown"
            for label in order
            if binding_sources[label] == name
        )
        else provider
        for name, provider in providers.items()
    }
    resources = tuple(
        ContextResource(
            source=capability.source,
            capability=capability,
            lease=ResourceLease.create(
                owner="caller", safe_detachment="forbidden", handle=handles[capability.source.name]
            )
            if capability.resource_owner == "caller"
            else None,
            factory=(lambda source=capability.source: handles[source.name])
            if capability.resource_owner == "sdk"
            else None,
        )
        for capability in capabilities
    )
    result = await (
        await start_initial_binding(
            data=data,
            workflow=workflow,
            declarations=declarations,
            capabilities=capabilities,
            resources=resources,
            limits=BindingLimits(
                max_declarations=len(declarations),
                max_sources=len(source_refs),
                max_capabilities=len(capabilities),
                max_selector_fields=0,
                max_selector_bytes=0,
                max_items=declaration_limits["max_items"],
                max_bytes=declaration_limits["max_bytes"],
                max_requests=cast(int, raw.get("hard_limit", 2)),
                max_resources=len(resources),
            ),
        )
    ).wait()
    preflight_rejection = case["boundary"] == "execution_preflight"
    expected = cast(dict[str, Any], case["expected"])["binding" if preflight_rejection else "state"]
    source_names = {fact.identity: label for label, fact in zip(order, result.receipt.sources, strict=True)}
    assert {source_names[fact.identity]: fact.terminal for fact in result.receipt.sources} == expected[
        "binding_sources"
    ]
    assert result.receipt.terminal == expected["binding_terminal"]
    if latest:
        if result.context is None:
            # Binding failure prevents execution and all publication allocation.
            assert not result.receipt.artifacts
            assert expected["binding_artifacts"] == expected["artifacts"] == []
            assert expected["allocator_next"] == 0
            assert expected["lineage_allocations"] == expected["operation_occurrences"] == {}
            assert "materialization" not in expected
        resource_names = {
            association.resource: f"Q:{label}"
            for association in result.receipt.cleanup_associations
            for label, target in zip(order, targets, strict=True)
            if association.targets == frozenset({target})
        }
        assert len(resource_names) == len(resources)
        assert len(result.receipt.cleanup_associations) == len(expected["binding_cleanup_associations"])
        assert len(result.receipt.cleanup) == len(expected["binding_cleanup"])
        assert {resource_names[fact.resource]: fact.disposition for fact in result.receipt.cleanup} == expected[
            "binding_cleanup"
        ]
        target_names = {target: f"T{index}" for index, target in enumerate(targets)}
        actual_cleanup = {}
        cleanup_owners = {fact.resource: fact.owner for fact in result.receipt.cleanup}
        for association in result.receipt.cleanup_associations:
            (target,) = association.targets
            name = resource_names[association.resource]
            label = name.removeprefix("Q:")
            actual_cleanup[name] = {
                "association": label,
                "target": target_names[target],
                "owner": cleanup_owners[association.resource],
            }
        assert actual_cleanup == expected["binding_cleanup_associations"]
        assert all(
            provider.closes == int(owners[name] == "sdk" and not isinstance(handles[name], _ProviderWithoutClose))
            for name, provider in providers.items()
        )
    if latest and not preflight_rejection:
        expected_artifacts = expected["binding_artifacts"]
        actual_artifacts = [
            {
                "association": source_names[item.reference.declaration],
                "key": item.reference.key,
                "version": item.reference.version,
                "source": item.source.name,
                "text": item.text,
            }
            for item in result.receipt.artifacts
        ]
    elif materializations and not preflight_rejection:
        target_names = {target: f"T{index}" for index, target in enumerate(targets)}
        expected_artifacts = [
            item for item in expected["artifacts"] if cast(str, item["identity"]).startswith("BoundInputKey:")
        ]
        expected_by_identity = {item["identity"]: item for item in expected_artifacts}
        actual_artifacts = []
        for item in result.receipt.artifacts:
            identity = (
                f"BoundInputKey:{target_names[item.target]}:N0:context:"
                f"{source_names[item.reference.declaration]}:{item.reference.key}:{item.reference.version}"
            )
            expected_item = expected_by_identity[identity]
            actual_item = {"identity": identity, "artifact_type": item_type.name}
            if "source" in expected_item:
                actual_item["source"] = item.source.name
            if "text" in expected_item:
                actual_item["text"] = item.text
            if "value" in expected_item:
                actual_item["value"] = item.text
            actual_artifacts.append(actual_item)
    else:
        expected_artifacts = expected["artifacts"]
        actual_artifacts = [
            {
                "identity": f"{source_names[item.reference.declaration]}:{item.reference.key}:{item.reference.version}",
                "source": item.source.name,
                "text": item.text,
            }
            for item in result.receipt.artifacts
        ]
    assert sorted(actual_artifacts, key=lambda item: json.dumps(item, sort_keys=True)) == sorted(
        expected_artifacts, key=lambda item: json.dumps(item, sort_keys=True)
    )
    request_actual = _normalize(
        result.receipt.requests,
        cast(Any, {name: association for association, name in association_names.items()}),
        {name: request for request, name in request_names.items()},
        policies,
    )
    dispatch_by_request = {item.request: item for item in result.receipt.requests.dispatches}
    terminal_associations: dict[str, object] = {}
    for terminal in result.receipt.requests.terminals:
        dispatch = dispatch_by_request[terminal.request]
        policy_name = next(name for name, policy in policies.items() if policy is dispatch.policy)
        for returned in terminal.results:
            terminal_associations[association_names[returned.association]] = {
                "request": request_names[terminal.request],
                "outcome": returned.outcome,
                "policy": policy_name,
            }
        if terminal.failure is not None:
            for association in dispatch.associations:
                terminal_associations[association_names[association]] = {
                    "request": request_names[terminal.request],
                    "failure": terminal.failure,
                    "policy": policy_name,
                }
    request_actual["association_terminals"] = terminal_associations
    for key in (
        "bindings",
        "dispatched",
        "dispatched_count",
        "terminals",
        "settlements",
        "request_facts",
        "attempts",
        "request_failures",
        "association_terminals",
        "request_associations",
        "request_policies",
        "cancel_requested",
        "local_in_flight",
        "remote_outstanding",
        "defects",
        "denials",
        "association_requests",
    ):
        actual_value = sorted(cast(list[str], request_actual[key])) if key == "dispatched" else request_actual[key]
        expected_value = sorted(cast(list[str], expected[key])) if key == "dispatched" else expected[key]
        assert actual_value == expected_value, (case["case_id"], key)
    if preflight_rejection:
        assert result.context is not None
        before = result.receipt
        await _assert_initial_execution_projection(
            data=data,
            workflow=workflow,
            node=node,
            item_type=item_type,
            input_type=artifact_type,
            bound_context=result.context,
            expected=expected,
            declaration_names=source_names,
            target_names={target: f"T{index}" for index, target in enumerate(targets)},
            limits=cast(dict[str, int], raw["materialization_limits"]),
            expected_rejection=case["expected"].get("code"),
            admission_only=latest,
        )
        assert result.receipt is before
    elif latest and all(fact.terminal == "omitted_optional" for fact in result.receipt.sources):
        assert result.context is not None and result.context.receipt is result.receipt
        assert not result.receipt.artifacts
        assert expected["binding_finished"] is True
        assert expected["materialization"] == {
            "artifact_bytes": 0,
            "artifact_count": 0,
            "ports": {},
            "provenance": {},
            "provenance_edges": 0,
        }
        assert expected["allocator_next"] == 0
        assert expected["lineage_allocations"] == expected["operation_occurrences"] == {}
        assert expected["publication_attempts"] == expected["publication_failures"] == []
    elif materializations and result.context is not None and expected.get("materialization") is not None:
        await _assert_initial_execution_projection(
            data=data,
            workflow=workflow,
            node=node,
            item_type=item_type,
            input_type=artifact_type,
            bound_context=result.context,
            expected=expected,
            declaration_names=source_names,
            target_names=target_names,
            limits=cast(dict[str, int], raw["materialization_limits"]),
            publication={
                **publication,
                "value": next(event["value"] for event in event_values if event["kind"] == "operation_publish"),
            }
            if publication is not None
            else None,
        )


async def _assert_initial_execution_projection(
    *,
    data: Any,
    workflow: Any,
    node: NodeId,
    item_type: ArtifactType,
    input_type: ArtifactType,
    bound_context: Any,
    expected: dict[str, Any],
    declaration_names: dict[BindingDeclarationId, str],
    target_names: Mapping[Any, str],
    limits: dict[str, int],
    expected_rejection: str | None = None,
    admission_only: bool = False,
    publication: dict[str, Any] | None = None,
) -> None:
    capability = _capability(workflow)
    prepared = _prepare(
        data=data,
        workflow=workflow,
        capability=capability,
        bound_inputs=(),
        limits=_limits(capabilities=1),
    )
    context = admit_context_plan(
        prepared=prepared,
        bound_context=bound_context,
        adaptive_retrievals=(),
        context_capabilities=(),
    )
    policy = OperationExecutionPolicy(
        node=node,
        kind="local",
        request=None,
        safe_detachment="forbidden",
        implementations=(
            ExecutionImplementation(
                implementation=capability.implementation,
                configuration=capability.configuration,
                capability=capability,
                request=None,
            ),
        ),
        result_outcomes=frozenset({"ok"}),
        runtime_outcomes=_valid_runtime_rows("local", frozenset({"ok"})),
    )
    context_port = bound_context.receipt.sources[0].declaration.port
    consumer = _ContextConsumer(
        port=context_port,
        outputs=()
        if publication is None
        else (
            PortArtifact(
                port=publication["output_port"],
                artifact_type=item_type,
                artifact=None,
                value=TextArtifactValue(text=publication["value"]),
            ),
        ),
        expected_input=None
        if publication is None
        else next(iter(expected["materialization"]["ports"].values()))["value"],
    )
    try:
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
                max_port_facts=16,
                max_provenance_edges=limits["max_provenance_edges"],
            ),
        )
    except EffectRejected as exc:
        assert admission_only and exc.code.value == expected_rejection
        assert consumer.calls == 0
        return
    if admission_only:
        assert expected_rejection is None
        assert consumer.calls == 0
        return
    try:
        running = await start_execution(
            admitted=admitted,
            capabilities=(capability,),
            services=ExecutionServices(
                handles=(
                    ImplementationHandle(
                        implementation=capability.implementation,
                        operation=capability.operation,
                        configuration=capability.configuration,
                        local=consumer,
                        transport=None,
                        resource=None,
                    ),
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
                clock=_Clock(),
            ),
        )
    except EffectRejected as exc:
        assert expected_rejection is not None
        assert exc.code.value == expected_rejection
        assert consumer.calls == 0
        return
    assert expected_rejection is None, "execution preflight unexpectedly admitted"
    execution = await running.wait()
    if publication is not None:
        assert consumer.calls == 1
        _assert_latest_publication(execution, bound_context.receipt, expected, declaration_names, target_names)
        return
    values = dict(execution.artifacts)
    bound_receipts = {item.reference: item for item in bound_context.receipt.artifacts}
    expected_by_identity = {item["identity"]: item for item in expected["artifacts"]}
    identities: dict[object, str] = {}
    actual_artifacts: list[dict[str, object]] = []
    for fact in execution.provenance:
        key = fact.key
        if isinstance(key, BoundInputKey):
            declaration = declaration_names[key.binding_artifact.declaration]
            identity = (
                f"BoundInputKey:{target_names[key.target]}:N0:{key.port}:"
                f"{declaration}:{key.binding_artifact.key}:{key.binding_artifact.version}"
            )
            artifact_type = item_type
        elif isinstance(key, InitialCollectionKey):
            declaration = declaration_names[key.declaration]
            identity = f"InitialCollectionKey:{target_names[key.target]}:N0:{key.port}:{declaration}"
            artifact_type = input_type
        else:
            continue
        identities[key] = identity
        value = values[fact.artifact]
        expected_item = expected_by_identity[identity]
        actual_item: dict[str, object] = {"identity": identity, "artifact_type": artifact_type.name}
        if isinstance(key, BoundInputKey):
            receipt = bound_receipts[key.binding_artifact]
            if "source" in expected_item:
                actual_item["source"] = receipt.source.name
            if "text" in expected_item:
                actual_item["text"] = receipt.text
            if "value" in expected_item:
                actual_item["value"] = receipt.text
        elif "value" in expected_item:
            actual_item["value"] = (
                value.text
                if isinstance(value, TextArtifactValue)
                else [{"key": item.key, "version": item.version, "value": item.value.text} for item in value.items]
            )
        actual_artifacts.append(actual_item)
    assert sorted(actual_artifacts, key=lambda item: cast(str, item["identity"])) == sorted(
        expected["artifacts"], key=lambda item: item["identity"]
    )
    aggregate = next(fact for fact in execution.provenance if isinstance(fact.key, BoundInputKey))
    if any(isinstance(fact.key, InitialCollectionKey) for fact in execution.provenance):
        aggregate = next(fact for fact in execution.provenance if isinstance(fact.key, InitialCollectionKey))
    aggregate_value = values[aggregate.artifact]
    declaration = declaration_names[bound_context.receipt.sources[0].identity]
    materialization = {
        "artifact_bytes": sum(
            len(value.text.encode())
            if isinstance(value, TextArtifactValue)
            else sum(len(item.value.text.encode()) for item in value.items)
            for value in values.values()
        ),
        "artifact_count": len(values),
        "ports": {
            f"initial:T0:N0:{context_port}:{declaration}": {
                "artifact_type": input_type.name,
                "key": identities[aggregate.key],
                "value": (
                    aggregate_value.text
                    if isinstance(aggregate_value, TextArtifactValue)
                    else [
                        {"key": item.key, "version": item.version, "value": item.value.text}
                        for item in aggregate_value.items
                    ]
                ),
            }
        },
        "provenance": {
            identities[fact.key]: sorted(identities[parent] for parent in fact.parents)
            for fact in execution.provenance
            if fact.key in identities
        },
        "provenance_edges": sum(len(fact.parents) for fact in execution.provenance),
    }
    assert materialization == expected["materialization"]
    assert len(execution.ports) == 1
    port = execution.ports[0]
    assert (port.node, port.target, port.port, port.artifact, port.artifact_type, port.role) == (
        node,
        next(iter(target_names)),
        context_port,
        aggregate.artifact,
        input_type,
        "artifact",
    )
    assert not execution.final_outputs
