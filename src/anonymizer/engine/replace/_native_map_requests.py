# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre-admitted native map requests inside an otherwise opaque invocation."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import replace
from typing import Any

from anonymizer.engine.graph_sdk.capabilities import FrozenConfig, ImplementationCapability, ImplementationRef
from anonymizer.engine.graph_sdk.executor import DelegatedReadiness, ExecutionImplementation, OperationExecutionPolicy
from anonymizer.engine.graph_sdk.requests import ChildRoute, PhysicalRequestPolicy
from anonymizer.engine.ndd._graph_requests import NddRequestBridge
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow
from anonymizer.graph.workflow import (
    NodeId,
    OperationNode,
    OperationSpec,
    OutcomeSpec,
    ResourceCeiling,
    SequenceEdge,
    compile_graph,
)

_BRIDGE: ContextVar[NddRequestBridge | None] = ContextVar("native_map_request_bridge", default=None)


def native_map_capacity(receiver: Any) -> int | None:
    if (
        getattr(getattr(receiver, "_adapter", None), "run_workflow", None) is None
        or getattr(receiver._adapter.run_workflow, "__func__", None) is not NddAdapter.run_workflow
        or getattr(receiver._generate_columns, "__func__", None) is not LlmReplaceWorkflow._generate_columns
    ):
        return None
    config = receiver._adapter._data_designer.run_config
    return (config.max_conversation_correction_steps + 1) * (config.max_conversation_restarts + 1)


def extend_native_map_requests(
    raw: Any, capacity: int
) -> tuple[Any, tuple[Any, ...], tuple[Any, ...], dict[str, NodeId]]:
    from anonymizer.engine.detection._graph_profile import _runtime_rows

    ids = {
        name: NodeId.new(workflow=raw.workflow)
        for name in ("readiness", "request", "probe", *(f"leaf-{i}" for i in range(capacity)))
    }
    nodes, capabilities, policies = [], [], []
    for name, node in ids.items():
        local = name == "probe"
        outcome = "async_required" if local else "ready" if name in {"readiness", "request"} else "ok"
        operation = OperationSpec(
            name="native-map-" + name,
            inputs=(),
            outputs=(),
            output_dependencies=(),
            outcomes=(
                OutcomeSpec(
                    name=outcome,
                    category="success",
                    produced_ports=frozenset(),
                    context=frozenset(),
                    evidence=frozenset(),
                    model_requirements=frozenset(),
                    ceiling=ResourceCeiling(
                        max_activations=1,
                        max_model_requests=0 if local else 1,
                        max_input_bytes=None,
                        max_output_bytes=None,
                    ),
                ),
            ),
        )
        request = (
            None
            if local
            else PhysicalRequestPolicy(
                visibility="dispatch_and_settlement",
                pre_dispatch_control="executor",
                retry_owner="executor",
                replay="idempotent",
                max_attempts=1,
            )
        )
        cap = ImplementationCapability(
            implementation=ImplementationRef(name=operation.name, revision=1),
            operation=operation,
            configuration=FrozenConfig(fields=()),
            effect="local" if local else "external",
            attribution="per_task",
            request_visibility="none" if local else "dispatch_and_settlement",
            pre_dispatch_control="none" if local else "executor",
            retry_owner="none" if local else "executor",
            error_reporting="typed_terminal",
            cancellation="cooperative_ack",
            settlement="synchronous" if local else "uncertain_possible",
            usage="exact",
            resource_lifetime="stateless" if name in {"readiness", "probe"} else "executor_owned",
            max_physical_requests_per_activation=0 if local else 1,
        )
        policy = OperationExecutionPolicy(
            node=node,
            kind="local" if local else "external",
            request=request,
            safe_detachment="forbidden",
            implementations=(
                ExecutionImplementation(
                    implementation=cap.implementation, configuration=cap.configuration, capability=cap, request=request
                ),
            ),
            result_outcomes=frozenset({outcome}),
            runtime_outcomes=_runtime_rows((outcome,), not local),
        )
        nodes.append(OperationNode(id=node, operation=operation))
        capabilities.append(cap)
        policies.append(policy)
    count = len(nodes)
    terminals = {binding.source.node for binding in raw.outcome_bindings}
    edges = tuple(SequenceEdge(before=node, after=terminal) for node in ids.values() for terminal in terminals) + (
        SequenceEdge(before=ids["readiness"], after=ids["probe"]),
    )
    graph = compile_graph(
        workflow=raw.workflow,
        interface=replace(
            raw.interface,
            outcomes=tuple(
                replace(o, ceiling=replace(o.ceiling, max_activations=o.ceiling.max_activations + count))
                for o in raw.interface.outcomes
            ),
        ),
        nodes=(*raw.nodes, *nodes),
        input_bindings=tuple(raw.input_bindings),
        output_bindings=tuple(raw.output_bindings),
        outcome_bindings=tuple(raw.outcome_bindings),
        sequence=(*raw.sequence, *edges),
        protection=tuple(raw.protection_requirements),
        limits=replace(
            raw.limits,
            max_nodes=raw.limits.max_nodes + count,
            max_sequence_edges=raw.limits.max_sequence_edges + len(edges),
        ),
        dynamic_limits=replace(
            raw.dynamic_limits, max_activation_occurrences=raw.dynamic_limits.max_activation_occurrences + count
        ),
    )
    return graph, tuple(capabilities), tuple(policies), ids


def bind_native_map_policy(prepared: Any, policies: tuple[Any, ...], ids: dict[str, NodeId]) -> tuple[Any, ...]:
    slots = {
        name: next(slot.index for slot in prepared.reservation_recipe if slot.template == node)
        for name, node in ids.items()
    }
    leaves = tuple(slot for name, slot in slots.items() if name.startswith("leaf-"))
    request = PhysicalRequestPolicy(
        visibility="dispatch_and_settlement",
        pre_dispatch_control="executor",
        retry_owner="implementation",
        replay="idempotent",
        max_attempts=len(leaves),
    )
    return tuple(
        replace(
            policy,
            delegated=replace(
                policy.delegated,
                observed_request=request,
                child_slots=leaves,
                generation_groups=(leaves,),
                routes=(ChildRoute(name="primary", parse_attempts=len(leaves)),),
                followups=frozenset({"correction", "conversation_restart"}),
                readiness=DelegatedReadiness(
                    owner_slot=slots["readiness"],
                    request_slot=slots["request"],
                    probe_slot=slots["probe"],
                    route=ChildRoute(name="readiness", parse_attempts=1),
                    ready_outcome="ready",
                    probe_outcomes=frozenset({"async_required"}),
                ),
            ),
        )
        if policy.delegated is not None
        else policy
        for policy in policies
    )
