# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre-admitted native map requests inside an otherwise opaque invocation."""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import replace
from typing import Any

from anonymizer.engine.graph_sdk._authoring_defaults import _compile_graph, _operation_spec, _outcome_spec
from anonymizer.engine.graph_sdk._workflow_values import (
    NodeId,
    OperationNode,
    ResourceCeiling,
    SequenceEdge,
)
from anonymizer.engine.graph_sdk.capabilities import FrozenConfig, ImplementationCapability, ImplementationRef
from anonymizer.engine.graph_sdk.executor import DelegatedReadiness, ExecutionImplementation, OperationExecutionPolicy
from anonymizer.engine.graph_sdk.requests import ChildRoute, PhysicalRequestPolicy
from anonymizer.engine.ndd._graph_requests import NddRequestBridge, scheduler_salvage_rounds, skip_health_checks_env
from anonymizer.engine.ndd._route import native_map_columns
from anonymizer.engine.ndd.model_loader import resolve_model_alias

_BRIDGE: ContextVar[NddRequestBridge | None] = ContextVar("native_map_request_bridge", default=None)


def native_map_capacity(receiver: Any, arguments: Mapping[str, Any]) -> int | None:
    """Return the map's physical request capacity per record when DD can run as native graph requests, else None.

    The capacity is DD's parser attempts plus its bounded transport retries (see ``dd_retry_allowance``); each
    physical attempt, including a retry, is its own leaf with its own dispatch and settlement.

    None keeps the map call opaque (the 0.4.0 shape). The native contract has a readiness slot that only DD's
    model health check satisfies (``NddRequestBridge`` routes the non-source request to ``readiness()``), so a
    replacement model that DD will not health-check cannot use it.
    """
    if not native_map_columns(receiver) or not _health_checked(arguments):
        return None
    config = receiver._adapter._data_designer.run_config
    return (config.max_conversation_correction_steps + 1) * (
        config.max_conversation_restarts + 1
    ) + dd_retry_allowance()


def dd_retry_allowance() -> int:
    """Retries admitted as accounted graph leaves for one record, beyond its parser attempts.

    DD 0.9.4 retries 502/503/504 and connection errors up to ``RetryConfig.max_retries`` times within one model
    call (``ModelRequestExecutor``), and its async scheduler re-runs a task that failed with a retryable model error
    (5xx, 429, timeout, connection) in up to ``salvage_max_rounds`` salvage rounds before dropping that record. This
    allowance covers that bounded policy exactly, so 5xx/timeout/connection exhaustion drops the record as in 0.4.0.
    429 (and request-admission timeouts) are "preserved retryable": DD never drops them and keeps re-salvaging. A
    record whose retries outrun the admitted leaves keeps retrying as DD-owned requests outside graph request
    accounting (``NddRequestBridge`` logs the count); the invocation declares no request upper bound and admits no
    hard request budget, so no bound is claimed for them.
    """
    from data_designer.engine.models.clients.retry import RetryConfig

    rounds = scheduler_salvage_rounds()
    return (RetryConfig().max_retries + 1) * (rounds + 1) - 1


def _health_checked(arguments: Mapping[str, Any]) -> bool:
    """Mirror DD's readiness skip rules (``data_designer.engine.readiness`` and ``ModelRegistry``) for the map model."""
    if os.environ.get(skip_health_checks_env()) == "1":
        return False
    alias = resolve_model_alias("replacement_generator", arguments["selected_models"])
    return not any(model.alias == alias and model.skip_health_check for model in arguments["model_configs"])


def extend_native_map_requests(
    raw: Any, capacity: int
) -> tuple[Any, tuple[Any, ...], tuple[Any, ...], dict[str, NodeId]]:
    from anonymizer.engine.harness.kernel import _runtime_rows

    ids = {
        name: NodeId.new(workflow=raw.workflow)
        for name in ("readiness", "request", "probe", *(f"leaf-{i}" for i in range(capacity)))
    }
    nodes, capabilities, policies = [], [], []
    for name, node in ids.items():
        local = name == "probe"
        outcome = "async_required" if local else "ready" if name in {"readiness", "request"} else "ok"
        operation = _operation_spec(
            name="native-map-" + name,
            outcomes=(
                _outcome_spec(
                    name=outcome,
                    category="success",
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
    graph = _compile_graph(
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
                followups=frozenset({"retry", "correction", "conversation_restart"}),
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
