# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Latest publication and late binding result comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

from anonymizer.engine.graph_sdk._effect_values import EffectRejected
from anonymizer.engine.graph_sdk.binding import start_initial_binding
from anonymizer.engine.graph_sdk.context import (
    BindingLimits,
    BindingReceipt,
    ContextMaterialization,
    ContextResource,
    ContextSelector,
    ContextSourceCapability,
    ContextSourceRef,
    InitialContextDecl,
    RetrievalBounds,
)
from anonymizer.engine.graph_sdk.executor import (
    BoundInputKey,
    ExecutionResult,
    OperationOutputKey,
    ProvenanceKey,
)
from anonymizer.engine.graph_sdk.requests import (
    BindingDeclarationId,
    PhysicalRequestId,
    PhysicalRequestPolicy,
    TextArtifactValue,
)
from anonymizer.engine.graph_sdk.resources import ResourceLease
from anonymizer.graph._values import ArtifactRef
from anonymizer.graph.workflow import (
    ContextInputRef,
    DynamicScope,
    InputPort,
    NodeOutputRef,
    OperationNode,
    OutputBinding,
    OutputDependency,
    OutputPort,
    WorkflowOutputRef,
    admit_activation_workflow,
    admit_static_workflow,
)
from tests.graph_sdk.effects_provider_fixtures import (
    _CaseProvider,
    _LateBindingCaseProvider,
)
from tests.graph_sdk.effects_request_fixtures import _normalize, _policy
from tests.graph_sdk.test_binding import _context_workflow
from tests.graph_sdk.test_preparation import _data


def _assert_latest_publication(
    execution: ExecutionResult,
    receipt: BindingReceipt,
    expected: dict[str, Any],
    declaration_names: dict[BindingDeclarationId, str],
    target_names: Mapping[Any, str],
) -> None:
    """Compare version retention, selected input and fresh output from a real run."""
    (state,) = execution.states
    (entry,) = tuple(state.entries)
    (source,) = receipt.sources
    declaration = declaration_names[source.identity]
    target = source.declaration.target
    target_name = target_names[target]
    node = source.declaration.node
    context_port = source.declaration.port
    assert entry.template == node and state.complete
    assert execution.record.targets == frozenset({target})
    assert not execution.assessments and not execution.pending_decisions
    assert not execution.requests.dispatches and not execution.cleanup
    values = dict(execution.artifacts)
    assert len(values) == len(execution.artifacts) == len(execution.provenance)
    assert execution.record.artifacts == frozenset(values)
    assert {fact.artifact for fact in execution.provenance} == set(values)
    bound_artifacts = {item.reference: item for item in receipt.artifacts}
    identities: dict[ProvenanceKey, str] = {}
    artifacts = []
    lineages: dict[str, set[int]] = {}

    def artifact_name(reference: ArtifactRef) -> str:
        assert reference.invocation == execution.record.invocation
        return f"ArtifactRef:I0:K{reference.key}:{reference.version}"

    for fact in execution.provenance:
        key = fact.key
        value = values[fact.artifact]
        assert isinstance(value, TextArtifactValue)
        if isinstance(key, BoundInputKey):
            assert key.node == node and key.target == target and key.port == context_port
            binding = key.binding_artifact
            assert binding.declaration == source.identity
            assert fact.artifact.version == binding.version
            identity = f"BoundInputKey:{target_name}:N0:{context_port}:{declaration}:{binding.key}:{binding.version}"
            source_name = bound_artifacts[binding].source.name
            assert value.text == bound_artifacts[binding].text
            lineages.setdefault(f"{declaration}:{binding.key}", set()).add(fact.artifact.key)
        else:
            assert isinstance(key, OperationOutputKey)
            assert key.target == target and key.activation == entry.activation
            assert entry.outcome == "ok"
            identity = f"OperationOutputKey:OP:{declaration}:{target_name}:N0:{entry.outcome}:{key.port}"
            source_name = "operation:N0"
            assert fact.artifact.version == 1
        identities[key] = identity
        artifacts.append(
            {
                "identity": artifact_name(fact.artifact),
                "artifact_type": source.declaration.artifact_type.name,
                "source": source_name,
                "text": value.text,
            }
        )
    assert sorted(artifacts, key=lambda item: item["identity"]) == sorted(
        expected["artifacts"], key=lambda item: item["identity"]
    )
    assert all(len(keys) == 1 for keys in lineages.values())
    assert {name: f"K{next(iter(keys))}" for name, keys in lineages.items()} == expected["lineage_allocations"]
    runtime_keys = {reference.key for reference in values}
    assert runtime_keys == set(range(expected["allocator_next"]))
    assert len(execution.ports) == 2
    (input_port,) = tuple(port for port in execution.ports if port.port == context_port)
    (output_port,) = tuple(port for port in execution.ports if port.port != context_port)
    assert input_port.node == output_port.node == node
    assert input_port.target == output_port.target == target
    assert input_port.activation == output_port.activation == entry.activation
    assert input_port.role == "artifact" and output_port.role == "candidate"
    assert input_port.artifact_type == output_port.artifact_type == source.declaration.artifact_type
    (selected,) = tuple(fact for fact in execution.provenance if fact.artifact == input_port.artifact)
    assert isinstance(selected.key, BoundInputKey)
    assert input_port.artifact.version == max(item.reference.version for item in receipt.artifacts)
    assert execution._input_parents == ((target, entry.activation, context_port, selected.key),)
    materialization = {
        "artifact_bytes": sum(
            len(value.text.encode()) for value in values.values() if isinstance(value, TextArtifactValue)
        ),
        "artifact_count": len(values),
        "ports": {
            f"initial:{target_name}:N0:{context_port}:{declaration}": {
                "artifact_type": input_port.artifact_type.name,
                "key": artifact_name(input_port.artifact),
                "value": next(
                    item["text"] for item in artifacts if item["identity"] == artifact_name(input_port.artifact)
                ),
            }
        },
        "provenance": {
            identities[fact.key]: {
                "artifact": artifact_name(fact.artifact),
                "parents": sorted(identities[parent] for parent in fact.parents),
            }
            for fact in execution.provenance
        },
        "provenance_edges": sum(len(fact.parents) for fact in execution.provenance),
    }
    assert materialization == expected["materialization"]
    (terminal,) = execution.record.terminals
    assert terminal.activation == entry.activation and terminal.attempt is not None
    assert terminal.attempt.activation == entry.activation
    assert terminal.category == entry.status == "success"
    assert not terminal.structural and not terminal.reasons
    assert {
        f"OP:{declaration}:TASK:OP:{declaration}": {
            "activation": f"OP:{declaration}",
            "attempt": f"TASK:OP:{declaration}",
            "binding_declaration": declaration,
            "node": "N0",
            "target": target_name,
            "published_ports": [output_port.port],
            "terminal": {"category": terminal.category, "outcome": entry.outcome},
        }
    } == expected["operation_occurrences"]
    (final,) = execution.final_outputs
    assert final.target == target and final.outcome == entry.outcome and final.port == output_port.port
    assert final.candidate.artifact == output_port.artifact
    assert final.producer in identities and identities[final.producer].startswith("OperationOutputKey:")


async def _assert_late_initial_provider_case(case: dict[str, Any]) -> None:
    raw = cast(dict[str, Any], case["declaration"])
    workflow, node, artifact_type = _context_workflow()
    materializations = cast(list[dict[str, Any]], raw.get("materializations", []))
    latest = bool(materializations and materializations[0].get("version_selection") == "latest")
    owner = "sdk" if latest else "caller"
    publication = materializations[0].get("publication") if materializations else None
    context_port = cast(str, materializations[0]["port"]) if materializations else "input"
    if context_port != "input":
        static = workflow.workflow
        operation_node = next(item for item in static.nodes if isinstance(item, OperationNode))
        operation = replace(
            operation_node.operation,
            inputs=(InputPort(name=context_port, artifact_type=artifact_type),),
            outcomes=tuple(
                replace(
                    outcome,
                    context=frozenset(replace(use, port=context_port) for use in outcome.context),
                )
                for outcome in operation_node.operation.outcomes
            ),
        )
        if publication is not None:
            assert materializations[0]["output_type"] == artifact_type.name
            assert publication["artifact_type"] == artifact_type.name
            assert publication["activation"] == f"OP:{materializations[0]['association']}"
            assert publication["node"] == materializations[0]["node"] == "N0"
            (published_outcome,) = operation.outcomes
            assert publication["outcome"] == published_outcome.name == "ok"
            operation = replace(
                operation,
                outputs=(OutputPort(name=publication["output_port"], artifact_type=artifact_type),),
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
            interface=replace(operation, inputs=(InputPort(name=context_port, artifact_type=artifact_type),)),
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
    data = _data(1)
    target = next(iter(data.targets))
    source = ContextSourceRef(name="S0", revision=1)
    policy = _policy(cast(dict[str, dict[str, Any]], raw["policies"])["P0"])
    association_names: dict[object, str] = {}
    request_names: dict[PhysicalRequestId, str] = {}
    response_events = [
        item
        for item in cast(list[dict[str, Any]], case["events"])
        if item["kind"] in {"source_result", "source_failure", "materialize_result"}
    ]
    delegate = _CaseProvider(
        events=response_events,
        sources={"S0": source},
        association_names=association_names,
        request_names=request_names,
        fallback_associations=["D0"],
        default_source=source,
    )
    acknowledge_stop = any(item["kind"] == "stop" for item in cast(list[dict[str, Any]], case["events"]))
    provider = _LateBindingCaseProvider(delegate=delegate, acknowledge_stop=acknowledge_stop)
    capability = ContextSourceCapability(
        source=source,
        artifact_type=artifact_type,
        uses=frozenset({"initial_binding"}),
        execution="async",
        resource_owner=cast(Any, owner),
        cancellation="cooperative_ack",
        settlement="explicit_ack",
        usage="exact",
        request=policy,
        safe_detachment="forbidden",
    )
    declaration = InitialContextDecl(
        target=target,
        node=node,
        port=context_port,
        artifact_type=artifact_type,
        source=source,
        selector=ContextSelector(fields=()),
        requirement="required",
        bounds=RetrievalBounds(
            max_items=materializations[0]["max_items"] if materializations else 1,
            max_bytes=materializations[0]["max_bytes"] if materializations else 12,
            max_requests=policy.max_attempts,
        ),
        version_selection="latest" if latest else "exact_one",
        materialization=ContextMaterialization(kind="single", item_type=artifact_type),
    )
    binding_limits = cast(dict[str, int], raw["binding_limits"])
    running = await start_initial_binding(
        data=data,
        workflow=workflow,
        declarations=(declaration,),
        capabilities=(capability,),
        resources=(
            ContextResource(
                source=source,
                capability=capability,
                lease=None
                if latest
                else ResourceLease.create(owner="caller", safe_detachment="forbidden", handle=provider),
                factory=(lambda: provider) if latest else None,
            ),
        ),
        limits=BindingLimits(
            max_declarations=1,
            max_sources=1,
            max_capabilities=1,
            max_selector_fields=0,
            max_selector_bytes=0,
            max_items=binding_limits["max_items"],
            max_bytes=binding_limits["max_bytes"],
            max_requests=cast(int, raw["hard_limit"]),
            max_resources=1,
        ),
    )
    await provider.started.wait()
    running.request_cancel()
    result = await running.wait()
    expected = cast(dict[str, Any], case["expected"])["state"]
    actual = _normalize(
        result.receipt.requests,
        cast(Any, {name: association for association, name in association_names.items()}),
        {name: request for request, name in request_names.items()},
        {"P0": policy},
    )
    for key in (
        "bindings",
        "dispatched",
        "dispatched_count",
        "terminals",
        "settlements",
        "request_facts",
        "attempts",
        "request_failures",
        "request_associations",
        "request_policies",
        "cancel_requested",
        "local_in_flight",
        "remote_outstanding",
        "defects",
        "denials",
        "association_requests",
    ):
        assert actual[key] == expected[key], (case["case_id"], key)
    expected_source = "cancelled" if acknowledge_stop else "lost"
    assert result.receipt.sources[0].terminal == expected_source
    assert result.receipt.terminal == ("failed" if acknowledge_stop else "lost")
    assert result.context is None
    assert not result.receipt.artifacts

    if latest:
        conflicts = []
        for defect in result.receipt.requests.defects:
            assert defect.association is None and defect.settlement is None
            terminal = defect.terminal
            assert terminal is not None
            rows = []
            for returned in terminal.results:
                assert not returned.outputs and not returned.consumed_context_ports
                rows.append(
                    {
                        "association": association_names[returned.association],
                        "outcome": returned.outcome,
                        "outputs": [],
                        "consumed_context_ports": [],
                    }
                )
            conflicts.append(
                {
                    "request": request_names[defect.request],
                    "code": defect.code,
                    "association": None,
                    "terminal": {
                        "request": request_names[terminal.request],
                        "category": terminal.category,
                        "failure": terminal.failure,
                        "results": rows,
                    },
                    "settlement": None,
                }
            )
        assert conflicts == expected["conflicting_terminal_facts"]
        # No BoundContext exists, so this driver cannot enter the execution owner.
        # Authenticate the reference's downstream zero projection at that boundary.
        assert expected["artifacts"] == []
        assert expected["lineage_allocations"] == {}
        assert expected["allocator_next"] == 0
        assert expected["operation_occurrences"] == {}
        assert expected["publication_attempts"] == []
        assert expected["publication_failures"] == []
        assert expected["binding_sources"] == {"D0": result.receipt.sources[0].terminal}
        assert expected["binding_terminal"] == result.receipt.terminal
        assert not expected["binding_artifacts"]
        assert not expected["association_terminals"]
        assert all(not terminal.results and terminal.failure is None for terminal in result.receipt.requests.terminals)
        (cleanup,) = result.receipt.cleanup
        (association,) = result.receipt.cleanup_associations
        assert cleanup.resource == association.resource
        assert association.targets == frozenset({target})
        assert association.purpose == "accounting"
        assert expected["binding_cleanup"] == {"Q:D0": cleanup.disposition}
        assert expected["binding_cleanup_associations"] == {
            "Q:D0": {"association": "D0", "target": "T0", "owner": cleanup.owner}
        }
        assert cleanup.owner == "sdk" and delegate.closes == 1


async def _reject_foreign_binding_target() -> str | None:
    workflow, node, artifact_type = _context_workflow()
    data = _data(1)
    foreign_target = next(iter(_data(1).targets))
    source = ContextSourceRef(name="S0", revision=1)
    policy = PhysicalRequestPolicy(
        visibility="dispatch_and_settlement",
        pre_dispatch_control="executor",
        retry_owner="executor",
        replay="idempotent",
        max_attempts=1,
    )
    capability = ContextSourceCapability(
        source=source,
        artifact_type=artifact_type,
        uses=frozenset({"initial_binding"}),
        execution="async",
        resource_owner="caller",
        cancellation="cooperative_ack",
        settlement="explicit_ack",
        usage="exact",
        request=policy,
        safe_detachment="forbidden",
    )
    try:
        await start_initial_binding(
            data=data,
            workflow=workflow,
            declarations=(
                InitialContextDecl(
                    target=foreign_target,
                    node=node,
                    port="input",
                    artifact_type=artifact_type,
                    source=source,
                    selector=ContextSelector(fields=()),
                    requirement="required",
                    bounds=RetrievalBounds(max_items=1, max_bytes=1, max_requests=1),
                    materialization=ContextMaterialization(kind="single", item_type=artifact_type),
                ),
            ),
            capabilities=(capability,),
            resources=(
                ContextResource(
                    source=source,
                    capability=capability,
                    lease=ResourceLease.create(owner="caller", safe_detachment="forbidden", handle=object()),
                    factory=None,
                ),
            ),
            limits=BindingLimits(
                max_declarations=1,
                max_sources=1,
                max_capabilities=1,
                max_selector_fields=0,
                max_selector_bytes=0,
                max_items=1,
                max_bytes=1,
                max_requests=1,
                max_resources=1,
            ),
        )
    except EffectRejected as exc:
        return exc.code.value
    return None
