# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure execution and service admission."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from inspect import iscoroutinefunction
from itertools import product

from anonymizer.engine.detection.artifacts import ENTITY_SPANS_TYPE
from anonymizer.engine.graph_sdk._effect_values import (
    EffectCode,
    reject,
    require_instance,
    require_literal,
)
from anonymizer.engine.graph_sdk._execution_delegated import delegated_submitter_of
from anonymizer.engine.graph_sdk._execution_frame import _invocation_workflows
from anonymizer.engine.graph_sdk._execution_topology import _node_operation, _workflow_owners
from anonymizer.engine.graph_sdk._execution_values import (
    _PLAN_KEY,
    AdmittedExecutionPlan,
    AssessmentLimits,
    DelegatedCallable,
    DelegatedGenerationSelection,
    DelegatedReadiness,
    DelegatedRequestContract,
    EvidenceProductionDecl,
    ExecutionKind,
    ExecutionLimits,
    ExecutionServices,
    ImplementationHandle,
    LocalCallable,
    MapExpansionDecl,
    OperationExecutionPolicy,
    RequestTransport,
)
from anonymizer.engine.graph_sdk._foreign_transport import ForeignLoopTransport
from anonymizer.engine.graph_sdk._graph_values import ContractViolation, ValidationCode
from anonymizer.engine.graph_sdk._ingress_values import IngressLimits
from anonymizer.engine.graph_sdk._workflow_composition import validate_dynamic_input_summaries
from anonymizer.engine.graph_sdk._workflow_values import (
    Graph,
    InputBinding,
    NodeId,
    NodeInputRef,
    NodeOutputRef,
    OperationNode,
    OperationSpec,
)
from anonymizer.engine.graph_sdk.blocking import BlockingTransport
from anonymizer.engine.graph_sdk.capabilities import (
    ImplementationCapability,
    validate_capability,
)
from anonymizer.engine.graph_sdk.context import (
    AdmittedContextPlan,
    BoundTextArtifact,
    MaterializationSchema,
)
from anonymizer.engine.graph_sdk.preparation import PreparedPlan
from anonymizer.engine.graph_sdk.requests import (
    BindingDeclarationId,
    ChildRoute,
    PhysicalRequestPolicy,
)


def admit_execution_plan(
    *,
    context: AdmittedContextPlan,
    capabilities: tuple[ImplementationCapability, ...],
    policies: tuple[OperationExecutionPolicy, ...],
    assessment_productions: tuple[EvidenceProductionDecl, ...],
    assessment_limits: AssessmentLimits,
    map_expansions: tuple[MapExpansionDecl, ...] = (),
    ingress_limits: IngressLimits = IngressLimits(),
) -> AdmittedExecutionPlan:
    """Admit exact execution policies over one immutable prepared plan."""
    require_instance(context, AdmittedContextPlan)
    # Exact type, then a detached copy: the issued response policy never aliases the caller's object.
    if type(ingress_limits) is not IngressLimits:
        reject(EffectCode.INVALID_TYPE)
    ingress_limits = replace(ingress_limits)
    prepared = context.prepared
    if not isinstance(capabilities, tuple):
        reject(EffectCode.INVALID_TYPE)
    if len(capabilities) > prepared.limits.max_capabilities:
        reject(EffectCode.LIMIT_EXCEEDED)
    if any(not isinstance(item, ImplementationCapability) for item in capabilities):
        reject(EffectCode.INVALID_TYPE)
    if len(capabilities) != len(set(capabilities)):
        reject(EffectCode.DUPLICATE)
    for capability in capabilities:
        validate_capability(capability)
    if not isinstance(policies, tuple) or any(not isinstance(item, OperationExecutionPolicy) for item in policies):
        reject(EffectCode.INVALID_TYPE)
    # Candidate spans are transient values, never text context or map membership.
    prepared_workflows = prepared.workflow.bodies()
    for body in prepared_workflows:
        for node in body.operations:
            for port in (*node.operation.inputs, *node.operation.outputs):
                if port.artifact_type.name == ENTITY_SPANS_TYPE.name and port.artifact_type != ENTITY_SPANS_TYPE:
                    reject(EffectCode.UNSUPPORTED)
    if any(item.artifact_type == ENTITY_SPANS_TYPE for item in prepared.bound_inputs):
        reject(EffectCode.UNSUPPORTED)
    materialization = _validate_map_expansions(context, map_expansions)
    workflow_owners = _workflow_owners(prepared.workflow)
    if any(item.node.workflow not in workflow_owners for item in policies):
        reject(EffectCode.FOREIGN_OWNER)
    selected = {item.node: item.capability for item in prepared.implementations}
    if {item.node for item in policies} != set(selected):
        reject(EffectCode.MISSING)
    if len(policies) != len({item.node for item in policies}):
        reject(EffectCode.DUPLICATE)
    for policy in policies:
        if not policy.implementations:
            reject(EffectCode.IMPLEMENTATION_COUNT)
        primary = policy.implementations[0]
        if primary.capability != selected[policy.node]:
            reject(EffectCode.UNSUPPORTED)
        if any(item.capability not in capabilities for item in policy.implementations):
            reject(EffectCode.UNSUPPORTED)
        _validate_policy(policy)
    _validate_delegated_recipes(prepared, policies)
    _validate_assessments(prepared, policies, assessment_productions, assessment_limits)
    _validate_execution_fact_capacity(prepared, assessment_productions, assessment_limits)
    target_workflow, invocation_workflow = _invocation_workflows(prepared, policies)
    return AdmittedExecutionPlan(
        _key=_PLAN_KEY,
        context=context,
        target_workflow=target_workflow,
        invocation_workflow=invocation_workflow,
        capabilities=capabilities,
        policies=frozenset(policies),
        map_expansions=map_expansions,
        assessment_productions=assessment_productions,
        assessment_limits=assessment_limits,
        ingress_limits=ingress_limits,
        materialization=materialization,
    )


def _validate_map_expansions(
    context: AdmittedContextPlan,
    declarations: tuple[MapExpansionDecl, ...],
) -> MaterializationSchema:
    if not isinstance(declarations, tuple) or any(not isinstance(item, MapExpansionDecl) for item in declarations):
        reject(EffectCode.INVALID_TYPE)
    workflow = context.prepared.workflow
    maps = [item for scope in workflow.bodies() for item in scope.maps]
    expected = {(item.expander, outcome) for item in maps for outcome in item.expansion_outcomes}
    keys = [(item.expander, item.outcome) for item in declarations]
    if len(keys) != len(set(keys)):
        reject(EffectCode.DUPLICATE)
    if set(keys) != expected:
        reject(EffectCode.MISSING if set(keys) < expected else EffectCode.EXTRA)
    # Context admission merged the initial entries; binding already rejected candidate spans there by name.
    schema = context.materialization.extended()
    bound = context.bound_context
    replacement_choices: list[tuple[InputBinding, ...]] = []
    by_key = {(item.expander, item.outcome): item for item in declarations}
    for scope in workflow.bodies():
        for endpoint in scope.map_item_ports:
            declaration = by_key.get((endpoint.expander, endpoint.expansion_outcome))
            if declaration is None:
                reject(EffectCode.MISSING)
            if declaration.membership_port != endpoint.membership_port:
                reject(EffectCode.CONTRADICTORY)
    for dynamic_map in maps:
        member_operation = _node_operation(workflow, dynamic_map.member)
        member_inputs = {item.name: item.artifact_type for item in member_operation.inputs}
        if dynamic_map.item_input is not None and any(
            fact.declaration.node == dynamic_map.member and fact.declaration.port == dynamic_map.item_input
            for fact in (() if bound is None else bound.receipt.sources)
        ):
            reject(EffectCode.CONTRADICTORY)
        map_replacements: list[InputBinding] = []
        for outcome_name in dynamic_map.expansion_outcomes:
            declaration = by_key[(dynamic_map.expander, outcome_name)]
            operation = _node_operation(workflow, dynamic_map.expander)
            outcome = next(item for item in operation.outcomes if item.name == outcome_name)
            output_types = {item.name: item.artifact_type for item in operation.outputs}
            if declaration.membership_port not in outcome.produced_ports:
                reject(EffectCode.UNSUPPORTED)
            collection_type = output_types.get(declaration.membership_port)
            if collection_type is None or collection_type == declaration.item_type:
                reject(EffectCode.CONTRADICTORY)
            if dynamic_map.item_input is not None:
                if member_inputs.get(dynamic_map.item_input) != declaration.item_type:
                    reject(EffectCode.CONTRADICTORY)
                map_replacements.append(
                    InputBinding(
                        source=NodeOutputRef(node=dynamic_map.expander, port=declaration.membership_port),
                        destination=NodeInputRef(node=dynamic_map.member, port=dynamic_map.item_input),
                    )
                )
            if ENTITY_SPANS_TYPE in (collection_type, declaration.item_type):
                reject(EffectCode.UNSUPPORTED)
            schema.merge(collection_type, declaration.item_type, "collection", 0)
        if map_replacements:
            replacement_choices.append(tuple(map_replacements))
    # Owners validate independently, so each owner segment's combinations are checked once (sum, not
    # product). The reported failure is the one the global product would hit first in lexicographic
    # order: the first segment failing on its first combination, else the last segment that fails at all.
    failure: ContractViolation | None = None
    for segment in _owner_segments(replacement_choices):
        combinations = product(*segment)
        first = _dynamic_input_failure(workflow, next(combinations))
        if first is not None:
            failure = first
            break
        later = (_dynamic_input_failure(workflow, item) for item in combinations)
        failure = next((item for item in later if item is not None), failure)
    if failure is not None:
        if failure.code is ValidationCode.MISSING:
            reject(EffectCode.MISSING)
        if failure.code is ValidationCode.CONTRADICTORY:
            reject(EffectCode.CONTRADICTORY)
        raise failure
    return schema


def _owner_segments(choices: list[tuple[InputBinding, ...]]) -> list[list[tuple[InputBinding, ...]]]:
    """Split per-map choices into contiguous runs that no owner workflow spans."""
    last = {choice[0].destination.node.workflow: index for index, choice in enumerate(choices)}
    segments: list[list[tuple[InputBinding, ...]]] = []
    start = 0
    end = -1
    for index, choice in enumerate(choices):
        end = max(end, last[choice[0].destination.node.workflow])
        if index == end:
            segments.append(choices[start : index + 1])
            start = index + 1
    return segments


def _dynamic_input_failure(workflow: Graph, replacements: tuple[InputBinding, ...]) -> ContractViolation | None:
    try:
        validate_dynamic_input_summaries(workflow, replacements)
    except ContractViolation as error:
        return error
    return None


def _validate_execution_fact_capacity(
    prepared: PreparedPlan,
    productions: tuple[EvidenceProductionDecl, ...],
    limits: AssessmentLimits,
) -> None:
    """Prove structural and operation fact storage from the finite reservation recipe."""
    root = prepared.workflow
    operation_nodes: dict[NodeId, OperationSpec] = {}
    subgraph_nodes: dict[NodeId, OperationSpec] = {}
    for current in root.bodies():
        for node in current.operations:
            if isinstance(node, OperationNode):
                operation_nodes[node.id] = node.operation
            else:
                subgraph_nodes[node.id] = node.operation
    port_facts = 0
    provenance_edges = 0
    assessment_facts = 0
    mapped_expanders = {
        declaration.expander: declaration.max_children
        for scope in prepared.workflow.bodies()
        for declaration in scope.maps
        if declaration.item_input is not None
    }
    for slot in prepared.reservation_recipe:
        provenance_edges += mapped_expanders.get(slot.template, 0)
        operation = operation_nodes.get(slot.template)
        if operation is not None:
            port_facts += len(operation.inputs)
            port_facts += max(len(outcome.produced_ports) for outcome in operation.outcomes)
            dependencies = {item.output: item for item in operation.output_dependencies}
            provenance_edges += max(
                sum(len(dependencies[port].inputs) for port in outcome.produced_ports) for outcome in operation.outcomes
            )
            assessment_facts += max(
                sum(item.node == slot.template and item.outcome == outcome.name for item in productions)
                for outcome in operation.outcomes
            )
            continue
        interface = subgraph_nodes[slot.template]
        projected_count = max(len(outcome.produced_ports) for outcome in interface.outcomes)
        port_facts += projected_count
        provenance_edges += projected_count
    target_count = len(prepared.target_occurrences)
    if port_facts * target_count > limits.max_port_facts:
        reject(EffectCode.LIMIT_EXCEEDED)
    if provenance_edges * target_count > limits.max_provenance_edges:
        reject(EffectCode.LIMIT_EXCEEDED)
    if assessment_facts * target_count > limits.max_assessment_facts:
        reject(EffectCode.LIMIT_EXCEEDED)


def _validate_policy(policy: OperationExecutionPolicy) -> None:
    require_literal(policy.initial_request_purpose, frozenset({"initial", "repair"}))
    if policy.initial_request_purpose == "repair" and (policy.kind != "external" or policy.request is None):
        reject(EffectCode.CONTRADICTORY)
    if policy.request is not None and policy.request.retry_owner == "implementation" and policy.kind != "delegated":
        reject(EffectCode.UNSUPPORTED)
    capabilities = [item.capability for item in policy.implementations]
    operation = capabilities[0].operation
    if any(item.operation != operation for item in capabilities):
        reject(EffectCode.CONTRADICTORY)
    if (
        policy.submission_owner is not None
        and policy.kind == "delegated"
        or policy.delegated is not None
        and policy.delegated.mode == "opaque"
    ):
        if (
            policy.kind != "delegated"
            or policy.request is not None
            or len(capabilities) != 1
            or capabilities[0].effect != "external"
            or capabilities[0].request_visibility != "none"
            or capabilities[0].max_physical_requests_per_activation is not None
            or policy.implementations[0].request is not None
        ):
            reject(EffectCode.CONTRADICTORY)
    elif policy.kind == "local":
        if len(policy.implementations) != 1 or policy.request is not None:
            reject(EffectCode.IMPLEMENTATION_COUNT)
        if capabilities[0].effect != "local" or policy.implementations[0].request is not None:
            reject(EffectCode.CONTRADICTORY)
    else:
        if policy.request is None or any(
            item.capability.effect != "external" or item.request != policy.request for item in policy.implementations
        ):
            reject(EffectCode.CONTRADICTORY)
        if any(item.max_physical_requests_per_activation != policy.request.max_attempts for item in capabilities):
            reject(EffectCode.CONTRADICTORY)
    if policy.safe_detachment == "independent_after_dispatch" and (
        policy.kind != "external" or any(item.resource_lifetime != "executor_owned" for item in capabilities)
    ):
        reject(EffectCode.CONTRADICTORY)
    if policy.kind != "external" and policy.safe_detachment != "forbidden":
        reject(EffectCode.CONTRADICTORY)
    outcomes = {item.name: item.category for item in operation.outcomes}
    if not policy.result_outcomes or not policy.result_outcomes <= outcomes.keys():
        reject(EffectCode.UNSUPPORTED)
    if any(item.outcome is not None and item.outcome not in outcomes for item in policy.runtime_outcomes):
        reject(EffectCode.UNSUPPORTED)
    keys = [(item.condition, item.reported_outcome, item.failure) for item in policy.runtime_outcomes]
    if len(keys) != len(set(keys)):
        reject(EffectCode.DUPLICATE)
    expected = _expected_runtime_keys(policy.kind, policy.result_outcomes)
    if set(keys) != expected:
        reject(EffectCode.MISSING if set(keys) < expected else EffectCode.EXTRA)
    for item in policy.runtime_outcomes:
        if item.condition == "result" and item.outcome != item.reported_outcome:
            reject(EffectCode.CONTRADICTORY)
        if item.outcome is not None and outcomes.get(item.outcome) != item.category:
            reject(EffectCode.CONTRADICTORY)
        if item.condition != "result" and item.outcome is not None and item.category == "success":
            reject(EffectCode.CONTRADICTORY)
        if item.outcome is None and item.category == "success":
            reject(EffectCode.CONTRADICTORY)
        if item.condition == "cancel_before_start" and (item.outcome is not None or item.category != "blocked"):
            reject(EffectCode.CONTRADICTORY)
        if item.condition == "request_inconsistent" and item.category != "inconsistent":
            reject(EffectCode.CONTRADICTORY)


def _validate_delegated_recipes(prepared: PreparedPlan, policies: tuple[OperationExecutionPolicy, ...]) -> None:
    """Select existing fixed sibling slots without compiling another inventory."""
    claimed: set[int] = set()
    recipe = prepared.reservation_recipe
    by_node = {policy.node: policy for policy in policies}
    for child in policies:
        if child.submission_owner is None:
            continue
        owner = by_node.get(child.submission_owner)
        if (
            owner is None
            or owner.delegated is None
            or (
                owner.delegated.mode != "opaque"
                and not (
                    owner.delegated.mode == "qualified"
                    and child.kind == "local"
                    and child.implementations[0].capability.max_physical_requests_per_activation == 0
                )
            )
            or child.implementations[0].capability.resource_lifetime != "stateless"
        ):
            reject(EffectCode.UNSUPPORTED)
        owner_slots = [slot for slot in recipe if slot.template == owner.node]
        if len(owner_slots) != 1 or owner_slots[0].parent_index is not None or owner_slots[0].iteration is not None:
            reject(EffectCode.UNSUPPORTED)
    for policy in policies:
        contract = policy.delegated
        if contract is None:
            continue
        _require_exact_contract(contract)
        request = contract.observed_request or policy.request
        if contract.mode == "opaque":
            if (
                prepared.configuration.purpose != "execution_only"
                or prepared.configuration.hard_request_limit is not None
                or any(
                    slot.template != policy.node
                    and slot.template in by_node
                    and by_node[slot.template].submission_owner != policy.node
                    and slot.index not in contract.owned_slots
                    for slot in recipe
                )
                or contract.usage_requirement != "unknown_allowed"
                or contract.observed_request is None
                and (
                    contract.child_slots
                    or contract.routes
                    or contract.followups
                    or contract.readiness is not None
                    or contract.selection is not None
                )
            ):
                reject(EffectCode.UNSUPPORTED)
            if contract.observed_request is None:
                continue
        source_free = (
            prepared.configuration.invocation_owned and not contract.child_slots and contract.readiness is not None
        )
        if request is None or (not contract.child_slots and not source_free) or not contract.routes:
            reject(EffectCode.MISSING)
        if (
            request.visibility != "dispatch_and_settlement"
            or request.pre_dispatch_control != "executor"
            or request.retry_owner != "implementation"
            or policy.safe_detachment != "forbidden"
        ):
            reject(EffectCode.UNSUPPORTED)
        if contract.mode != "opaque" and any(
            item.capability.request_visibility != request.visibility
            or item.capability.pre_dispatch_control != request.pre_dispatch_control
            or item.capability.retry_owner != request.retry_owner
            or (contract.usage_requirement == "exact" and item.capability.usage != "exact")
            for item in policy.implementations
        ):
            reject(EffectCode.UNSUPPORTED)
        if len(contract.child_slots) > len(recipe):
            reject(EffectCode.LIMIT_EXCEEDED)
        # A shared route bounds parser attempts; each group's physical inventory
        # can impose a tighter bound (for example, a validator without restarts).
        capacity = 1 if source_free else max(map(len, contract.generations))
        if (1 if source_free else len(contract.child_slots)) != request.max_attempts or sum(
            route.parse_attempts for route in contract.routes
        ) > capacity:
            reject(EffectCode.LIMIT_EXCEEDED)
        if contract.selection is not None and any(
            contract.selection.port not in {port.name for port in implementation.capability.operation.outputs}
            or any(
                contract.selection.port not in outcome.produced_ports
                for outcome in implementation.capability.operation.outcomes
                if outcome.category == "success"
            )
            for implementation in policy.implementations
        ):
            reject(EffectCode.MISSING)
        if contract.optional_tail and (
            len(contract.child_slots) != 2
            or contract.optional_tail != contract.child_slots[1:]
            or len(contract.routes) != 1
            or contract.routes[0].parse_attempts != 2
            or contract.followups != {"correction"}
            or not all(item.capability.operation.outputs for item in policy.implementations)
        ):
            reject(EffectCode.UNSUPPORTED)
        owners = [slot for slot in recipe if slot.template == policy.node]
        if len(owners) != 1 or owners[0].iteration is not None:
            reject(EffectCode.UNSUPPORTED)
        for index in contract.owned_slots:
            if index >= len(recipe) or index in claimed:
                reject(EffectCode.FOREIGN_OWNER)
            slot = recipe[index]
            leaf = by_node.get(slot.template)
            if (
                slot.index == owners[0].index
                or slot.parent_index != owners[0].parent_index
                or slot.iteration is not None
                or leaf is None
                or (
                    leaf.kind != "local"
                    if contract.readiness is not None and index == contract.readiness.probe_slot
                    else leaf.kind != "external" or leaf.request is None or leaf.request.max_attempts != 1
                )
                or sum(item.template == slot.template for item in recipe) != 1
            ):
                reject(EffectCode.UNSUPPORTED)
            claimed.add(index)
        readiness = contract.readiness
        if readiness is not None:
            semantic, physical, probe = (by_node[recipe[index].template] for index in readiness.slots)
            if (
                semantic.request != physical.request
                or semantic.result_outcomes != physical.result_outcomes
                or readiness.ready_outcome not in semantic.result_outcomes
                or not readiness.probe_outcomes <= probe.result_outcomes
                or any(
                    not any(
                        row.name == name and row.category == "success"
                        for row in item.implementations[0].capability.operation.outcomes
                    )
                    for item, names in ((semantic, {readiness.ready_outcome}), (probe, readiness.probe_outcomes))
                    for name in names
                )
                or any(
                    item.capability.usage != "exact" for item in (*semantic.implementations, *physical.implementations)
                )
                or semantic.request is None
                or semantic.request.visibility != "dispatch_and_settlement"
                or semantic.request.pre_dispatch_control != "executor"
                or any(
                    item.capability.request_visibility != semantic.request.visibility
                    or item.capability.pre_dispatch_control != semantic.request.pre_dispatch_control
                    or item.capability.retry_owner != semantic.request.retry_owner
                    for item in (*semantic.implementations, *physical.implementations)
                )
            ):
                reject(EffectCode.UNSUPPORTED)


def _expected_runtime_keys(kind: ExecutionKind, results: frozenset[str]) -> set[tuple[str, str | None, str | None]]:
    keys: set[tuple[str, str | None, str | None]] = {("result", item, None) for item in results}
    failures = {
        "rejected_before_acceptance",
        "retryable",
        "malformed_response",
        "permanent",
        "transport_unknown",
        "implementation_exception",
    }
    keys.update(("failure", None, item) for item in failures)
    conditions = {"cancel_before_start", "cancel_after_start", "artifact_limit_exhausted", "deadline_exhausted"}
    if kind in {"external", "delegated"}:
        conditions |= {
            "cancel_after_dispatch",
            "lost",
            "request_inconsistent",
            "budget_exhausted",
            "request_limit_exhausted",
        }
    keys.update((item, None, None) for item in conditions)
    return keys


def _validate_assessments(
    prepared: object,
    policies: tuple[OperationExecutionPolicy, ...],
    productions: tuple[EvidenceProductionDecl, ...],
    limits: AssessmentLimits,
) -> None:
    del prepared
    if not isinstance(productions, tuple) or any(not isinstance(item, EvidenceProductionDecl) for item in productions):
        reject(EffectCode.INVALID_TYPE)
    require_instance(limits, AssessmentLimits)
    if len(productions) > limits.max_productions:
        reject(EffectCode.LIMIT_EXCEEDED)
    production_keys = [(item.node, item.outcome, item.promise) for item in productions]
    if len(production_keys) != len(set(production_keys)):
        reject(EffectCode.DUPLICATE)
    policy_by_node = {item.node: item for item in policies}
    for item in productions:
        policy = policy_by_node.get(item.node)
        if policy is None or policy.kind != "local":
            reject(EffectCode.UNSUPPORTED)
        operation = policy.implementations[0].capability.operation
        outcome = next((value for value in operation.outcomes if value.name == item.outcome), None)
        if outcome is None or item.evidence_port not in outcome.produced_ports:
            reject(EffectCode.UNSUPPORTED)
        promise = next((value for value in outcome.evidence if value.name == item.promise), None)
        if promise is None:
            reject(EffectCode.UNSUPPORTED)
        dependency = next(
            (value for value in operation.output_dependencies if value.output == item.evidence_port),
            None,
        )
        if dependency is None or dependency.inputs != promise.consumed_ports:
            reject(EffectCode.CONTRADICTORY)
        if len(item.supported_findings) > limits.max_findings_per_production:
            reject(EffectCode.LIMIT_EXCEEDED)
        if any(len(value.code.encode()) > limits.max_finding_code_bytes for value in item.supported_findings):
            reject(EffectCode.LIMIT_EXCEEDED)


def _validate_delegated_services(admitted: AdmittedExecutionPlan, services: ExecutionServices) -> None:
    """Admit fixed generation and its declared optional correction with owned resources."""
    prepared = admitted.context.prepared
    delegated = [policy for policy in admitted.policies if policy.delegated is not None]
    if len(delegated) > 1:
        reject(EffectCode.UNSUPPORTED)
    for policy in delegated:
        contract = policy.delegated
        if contract is None:
            continue
        if contract.mode == "opaque":
            handles = [
                handle
                for handle in services.handles
                if handle.implementation == policy.implementations[0].implementation
            ]
            if len(handles) != 1:
                reject(EffectCode.MISSING)
            handle = handles[0]
            worker = delegated_submitter_of(handle.delegated)
            if (
                worker is None
                or worker.released
                or worker.borrowed_loop is not asyncio.get_running_loop()
                or handle.resource is None
                or handle.resource.owner != "caller"
                or handle.resource.handle is not worker
            ):
                reject(EffectCode.UNSUPPORTED)
            if contract.observed_request is None:
                continue
        if len(contract.routes) != 1:
            reject(EffectCode.UNSUPPORTED)
        # A transient-failure retry replays the same request, so only an idempotent observed request may declare it.
        policy_request = contract.observed_request or policy.request
        if not contract.followups <= {"retry", "correction", "conversation_restart"} or (
            "retry" in contract.followups and (policy_request is None or policy_request.replay != "idempotent")
        ):
            reject(EffectCode.UNSUPPORTED)
        foreign = [
            handle.transport for handle in services.handles if isinstance(handle.transport, ForeignLoopTransport)
        ]
        if foreign and (
            contract.readiness is None
            or not (
                (prepared.configuration.invocation_owned and not contract.child_slots)
                or (
                    contract.generation_groups and contract.followups <= {"retry", "correction", "conversation_restart"}
                )
                or all(len(group) == 1 for group in contract.generations)
                or (len(contract.child_slots) == 2 and contract.optional_tail == contract.child_slots[1:])
            )
            or any(
                type(handle.transport) is not ForeignLoopTransport
                for handle in services.handles
                if handle.transport is not None
            )
        ):
            reject(EffectCode.UNSUPPORTED)
        nodes = {policy.node, *(prepared.reservation_recipe[index].template for index in contract.owned_slots)}
        if any(item.kind == "external" and item.node not in nodes for item in admitted.policies):
            reject(EffectCode.UNSUPPORTED)
        for owned in (item for item in admitted.policies if item.node in nodes):
            readiness = contract.readiness
            readiness_nodes = (
                set()
                if readiness is None
                else {prepared.reservation_recipe[index].template for index in readiness.slots}
            )
            if len(owned.implementations) != 1 or (
                owned.node not in readiness_nodes and owned.result_outcomes != policy.result_outcomes
            ):
                reject(EffectCode.UNSUPPORTED)
            implementation = owned.implementations[0]
            operation = implementation.capability.operation
            if owned.node != policy.node and (operation.inputs or operation.outputs):
                reject(EffectCode.UNSUPPORTED)
            if readiness is not None and owned.node == prepared.reservation_recipe[readiness.owner_slot].template:
                if implementation.capability.resource_lifetime != "stateless":
                    reject(EffectCode.UNSUPPORTED)
                if (
                    sum(
                        item.implementation == implementation.implementation
                        for row in admitted.policies
                        for item in row.implementations
                    )
                    != 1
                ):
                    reject(EffectCode.UNSUPPORTED)
                continue
            handle = next(
                (
                    item
                    for item in services.handles
                    if item.implementation == implementation.implementation
                    and item.operation == operation
                    and item.configuration == implementation.configuration
                ),
                None,
            )
            if handle is None:
                reject(EffectCode.UNSUPPORTED)
            if delegated_submitter_of(handle.delegated) is not None:
                if (
                    implementation.capability.resource_lifetime != "caller_owned"
                    or handle.resource is None
                    or handle.resource.owner != "caller"
                    or handle.resource.handle is not handle.delegated
                ):
                    reject(EffectCode.UNSUPPORTED)
            elif type(handle.transport) is ForeignLoopTransport:
                parents = [
                    parent
                    for item in services.handles
                    if (parent := delegated_submitter_of(item.delegated)) is not None
                ]
                if len(parents) != 1 or owned.node == policy.node or owned.kind != "external":
                    reject(EffectCode.UNSUPPORTED)
                if implementation.capability.resource_lifetime == "stateless" or owned.safe_detachment != "forbidden":
                    reject(EffectCode.UNSUPPORTED)
                handle.transport.validate(
                    handle.resource,
                    handle.transport.owning_loop if contract.observed_request is not None else parents[0].borrowed_loop,
                )
            elif handle.resource is not None or implementation.capability.resource_lifetime != "stateless":
                reject(EffectCode.UNSUPPORTED)
            callback = (
                handle.delegated.submit
                if handle.delegated is not None
                else (
                    handle.transport.dispatch
                    if handle.transport is not None
                    else handle.local.run
                    if handle.local is not None
                    else None
                )
            )
            borrowing = delegated_submitter_of(handle.delegated)
            if borrowing is not None:
                loop = borrowing.borrowed_loop
                # Pool/loop are live caller-owned execution resources, not stateless
                # provider handles. Their unrelated users do not belong to this invocation.
                if (
                    borrowing.released
                    or not loop.is_running()
                    or loop.is_closed()
                    or loop is asyncio.get_running_loop()
                    and contract.observed_request is None
                ):
                    reject(EffectCode.UNSUPPORTED)
            elif handle.delegated is not None and getattr(handle.delegated, "execution", None) != "same_loop":
                reject(EffectCode.UNSUPPORTED)
            if callback is None or not iscoroutinefunction(callback) or isinstance(handle.transport, BlockingTransport):
                reject(EffectCode.UNSUPPORTED)


def _require_exact_contract(contract: DelegatedRequestContract) -> None:
    """Accept only exact, immutable recipe types, so the admitted recipe cannot change after admission.

    Every member was validated when it was constructed; exact types (no subclass properties) plus frozen
    dataclasses over tuples/frozensets of primitives make the caller's object a detached snapshot.
    """
    readiness, selection = contract.readiness, contract.selection
    if (
        type(contract) is not DelegatedRequestContract
        or type(contract.observed_request) not in (type(None), PhysicalRequestPolicy)
        or type(readiness) not in (type(None), DelegatedReadiness)
        or (readiness is not None and type(readiness.route) is not ChildRoute)
        or type(selection) not in (type(None), DelegatedGenerationSelection)
        or (selection is not None and type(selection.slots) is not tuple)
        or any(type(item) is not tuple for item in (contract.child_slots, contract.routes, contract.optional_tail))
        or type(contract.generation_groups) is not tuple
        or any(type(group) is not tuple for group in contract.generation_groups)
        or any(type(route) is not ChildRoute for route in contract.routes)
    ):
        reject(EffectCode.INVALID_TYPE)


def _validate_services(admitted: AdmittedExecutionPlan, services: ExecutionServices) -> None:
    if any(isinstance(handle.transport, ForeignLoopTransport) for handle in services.handles) and not any(
        policy.delegated is not None for policy in admitted.policies
    ):
        reject(EffectCode.UNSUPPORTED)
    require_instance(services.limits, ExecutionLimits)
    if (
        services.limits.max_remote_outstanding is not None or services.limits.max_runtime_artifact_bytes is not None
    ) and any(policy.delegated is not None and policy.delegated.mode == "opaque" for policy in admitted.policies):
        reject(EffectCode.UNSUPPORTED)
    bound_context = admitted.context.bound_context
    if bound_context is not None and any(
        services.limits.max_collection_items is not None
        and item.declaration.materialization.kind == "collection"
        and item.declaration.bounds.max_items > services.limits.max_collection_items
        for item in bound_context.receipt.sources
    ):
        reject(EffectCode.LIMIT_EXCEEDED)
    _validate_initial_materialization_capacity(admitted, services.limits)
    if not isinstance(services.handles, tuple) or any(
        not isinstance(item, ImplementationHandle) for item in services.handles
    ):
        reject(EffectCode.INVALID_TYPE)
    _validate_delegated_services(admitted, services)
    retained = [item for _, item in admitted._service_implementations()]
    if len(services.handles) > admitted.context.prepared.limits.max_capabilities:
        reject(EffectCode.LIMIT_EXCEEDED)
    handle_keys = [(item.implementation, item.operation, item.configuration) for item in services.handles]
    expected = [(item.implementation, item.capability.operation, item.configuration) for item in retained]
    if len(handle_keys) != len(set(handle_keys)):
        reject(EffectCode.DUPLICATE)
    if set(handle_keys) != set(expected):
        reject(EffectCode.MISSING)
    capability_by_key = {
        (item.implementation, item.capability.operation, item.configuration): item.capability for item in retained
    }
    policy_by_key = {
        (item.implementation, item.capability.operation, item.configuration): policy
        for policy in admitted.policies
        for item in policy.implementations
    }
    for handle in services.handles:
        key = (handle.implementation, handle.operation, handle.configuration)
        capability = capability_by_key[key]
        policy = policy_by_key[key]
        if policy.submission_owner is not None:
            if any(item is not None for item in (handle.local, handle.transport, handle.delegated, handle.resource)):
                reject(EffectCode.CONTRADICTORY)
            continue
        if sum(item is not None for item in (handle.local, handle.transport, handle.delegated)) != 1:
            reject(EffectCode.CONTRADICTORY)
        if policy.kind == "external":
            if handle.transport is None or not isinstance(handle.transport, RequestTransport):
                reject(EffectCode.INVALID_TYPE)
        elif policy.kind == "delegated":
            if handle.delegated is None or not isinstance(handle.delegated, DelegatedCallable):
                reject(EffectCode.INVALID_TYPE)
        elif handle.local is None or not isinstance(handle.local, LocalCallable):
            reject(EffectCode.INVALID_TYPE)
        if isinstance(handle.transport, BlockingTransport) and handle.resource is not None:
            if handle.resource.handle is not handle.transport:
                reject(EffectCode.CONTRADICTORY)
        if capability.resource_lifetime == "stateless" and handle.resource is not None:
            reject(EffectCode.CONTRADICTORY)
        if capability.resource_lifetime != "stateless":
            if handle.resource is None:
                reject(EffectCode.MISSING)
            expected_owner = "caller" if capability.resource_lifetime == "caller_owned" else "sdk"
            if handle.resource.owner != expected_owner or handle.resource.safe_detachment != policy.safe_detachment:
                reject(EffectCode.CONTRADICTORY)


def _validate_initial_materialization_capacity(admitted: AdmittedExecutionPlan, limits: ExecutionLimits) -> None:
    prepared = admitted.context.prepared
    datum_text = {item.id: item.text for item in prepared.data.datums}
    root_lineages = {(item.source, item.artifact_type) for item in prepared.bound_inputs}
    artifact_count = len(root_lineages)
    artifact_bytes = sum(len(datum_text[source].encode()) for source, _ in root_lineages)
    provenance_edges = 0
    context = admitted.context.bound_context
    if context is not None:
        declarations = {item.identity: item.declaration for item in context.receipt.sources}
        groups: dict[BindingDeclarationId, list[BoundTextArtifact]] = {}
        for artifact in context.artifacts:
            groups.setdefault(artifact.reference.declaration, []).append(artifact)
        for identity, items in groups.items():
            declaration = declarations[identity]
            if (
                limits.max_collection_items is not None
                and declaration.materialization.kind == "collection"
                and len(items) > limits.max_collection_items
            ):
                reject(EffectCode.LIMIT_EXCEEDED)
            item_bytes = sum(len(item.text.encode()) for item in items)
            artifact_count += len(items)
            artifact_bytes += item_bytes
            if declaration.materialization.kind == "collection":
                artifact_count += 1
                artifact_bytes += item_bytes
                provenance_edges += len(items)
    if artifact_count > limits.max_runtime_artifacts or (
        limits.max_runtime_artifact_bytes is not None and artifact_bytes > limits.max_runtime_artifact_bytes
    ):
        reject(EffectCode.LIMIT_EXCEEDED)
    if provenance_edges > admitted.assessment_limits.max_provenance_edges:
        reject(EffectCode.LIMIT_EXCEEDED)
