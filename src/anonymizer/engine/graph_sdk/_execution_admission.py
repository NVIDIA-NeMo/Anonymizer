# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure execution and service admission."""

from __future__ import annotations

from itertools import product
from typing import Literal

from anonymizer.engine.graph_sdk._effect_values import (
    EffectCode,
    reject,
    require_count,
    require_instance,
)
from anonymizer.engine.graph_sdk._execution_topology import _node_operation, _workflow_owners
from anonymizer.engine.graph_sdk._execution_values import (
    _PLAN_KEY,
    AdmittedExecutionPlan,
    AssessmentLimits,
    DecisionDeclaration,
    DecisionLimits,
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
from anonymizer.engine.graph_sdk.capabilities import (
    ImplementationCapability,
    validate_capability,
)
from anonymizer.engine.graph_sdk.context import (
    AdmittedContextPlan,
    BoundTextArtifact,
    ContextResource,
)
from anonymizer.engine.graph_sdk.preparation import PreparedPlan
from anonymizer.engine.graph_sdk.requests import (
    BindingDeclarationId,
)
from anonymizer.graph._values import (
    ContractViolation,
    ValidationCode,
)
from anonymizer.graph.workflow import (
    ArtifactType,
    InputBinding,
    NodeId,
    NodeInputRef,
    NodeOutputRef,
    OperationNode,
    OperationSpec,
    validate_dynamic_input_summaries,
)


def admit_execution_plan(
    *,
    context: AdmittedContextPlan,
    capabilities: tuple[ImplementationCapability, ...],
    policies: tuple[OperationExecutionPolicy, ...],
    decisions: tuple[DecisionDeclaration, ...],
    assessment_productions: tuple[EvidenceProductionDecl, ...],
    assessment_limits: AssessmentLimits,
    map_expansions: tuple[MapExpansionDecl, ...] = (),
) -> AdmittedExecutionPlan:
    """Admit exact execution policies over one immutable prepared plan."""
    require_instance(context, AdmittedContextPlan)
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
    if not isinstance(decisions, tuple) or any(not isinstance(item, DecisionDeclaration) for item in decisions):
        reject(EffectCode.INVALID_TYPE)
    _validate_map_expansions(context, map_expansions)
    workflow_owners = _workflow_owners(prepared.workflow.workflow)
    if any(item.node.workflow not in workflow_owners for item in policies):
        reject(EffectCode.FOREIGN_OWNER)
    selected = {item.node: item.capability for item in prepared.implementations}
    if {item.node for item in policies} != set(selected):
        reject(EffectCode.MISSING)
    if len(policies) != len({item.node for item in policies}):
        reject(EffectCode.DUPLICATE)
    decision_nodes = {item.node for item in decisions}
    if len(decisions) != len(decision_nodes):
        reject(EffectCode.DUPLICATE)
    policy_by_node = {item.node: item for item in policies}
    for declaration in decisions:
        policy = policy_by_node.get(declaration.node)
        if policy is None or policy.kind != "decision":
            reject(EffectCode.UNSUPPORTED)
        operation = policy.implementations[0].capability.operation
        if declaration.artifact_port not in {item.name for item in operation.inputs}:
            reject(EffectCode.MISSING)
        declared_outcomes = {item.name for item in operation.outcomes}
        if any(item.outcome not in declared_outcomes for item in declaration.outcomes):
            reject(EffectCode.UNSUPPORTED)
        dependencies = {item.output: item for item in operation.output_dependencies}
        for decision_outcome in declaration.outcomes:
            outcome = next(item for item in operation.outcomes if item.name == decision_outcome.outcome)
            if any(
                dependencies.get(port) is None or dependencies[port].identity_input != declaration.artifact_port
                for port in outcome.produced_ports
            ):
                reject(EffectCode.UNSUPPORTED)
    for policy in policies:
        if not policy.implementations:
            reject(EffectCode.IMPLEMENTATION_COUNT)
        primary = policy.implementations[0]
        if primary.capability != selected[policy.node]:
            reject(EffectCode.UNSUPPORTED)
        if any(item.capability not in capabilities for item in policy.implementations):
            reject(EffectCode.UNSUPPORTED)
        _validate_policy(policy, policy.node in decision_nodes)
        adaptive = next((item for item in context.adaptive_retrievals if item.node == policy.node), None)
        if adaptive is not None:
            source = next(
                item
                for item in context.context_capabilities
                if item.source == adaptive.source and "adaptive_retrieval" in item.uses
            )
            if policy.kind != "external" or policy.request != source.request:
                reject(EffectCode.CONTRADICTORY)
            if any(item.capability.attribution != "per_task" for item in policy.implementations):
                reject(EffectCode.UNSUPPORTED)
    _validate_assessments(prepared, policies, assessment_productions, assessment_limits)
    _validate_execution_fact_capacity(prepared, assessment_productions, assessment_limits)
    return AdmittedExecutionPlan(
        _key=_PLAN_KEY,
        context=context,
        capabilities=capabilities,
        policies=frozenset(policies),
        decisions=frozenset(decisions),
        map_expansions=map_expansions,
        assessment_productions=assessment_productions,
        assessment_limits=assessment_limits,
    )


def _validate_map_expansions(
    context: AdmittedContextPlan,
    declarations: tuple[MapExpansionDecl, ...],
) -> None:
    if not isinstance(declarations, tuple) or any(not isinstance(item, MapExpansionDecl) for item in declarations):
        reject(EffectCode.INVALID_TYPE)
    workflow = context.prepared.workflow
    maps = [item for scope in workflow.scopes for item in scope.maps]
    expected = {(item.expander, outcome) for item in maps for outcome in item.expansion_outcomes}
    keys = [(item.expander, item.outcome) for item in declarations]
    if len(keys) != len(set(keys)):
        reject(EffectCode.DUPLICATE)
    if set(keys) != expected:
        reject(EffectCode.MISSING if set(keys) < expected else EffectCode.EXTRA)
    schemas: dict[ArtifactType, tuple[Literal["scalar", "collection"], ArtifactType, int]] = {}
    bound = context.bound_context
    if bound is not None:
        for fact in bound.receipt.sources:
            materialization = fact.declaration.materialization
            _add_materialization_schema(
                schemas,
                fact.declaration.artifact_type,
                materialization.item_type,
                materialization.kind,
                1,
            )
    for adaptive in context.adaptive_retrievals:
        operation = _node_operation(workflow.workflow, adaptive.node)
        output_type = next(item.artifact_type for item in operation.outputs if item.name == adaptive.output_port)
        _add_materialization_schema(
            schemas,
            output_type,
            adaptive.materialization.item_type,
            adaptive.materialization.kind,
            1,
        )
    replacement_choices: list[tuple[InputBinding, ...]] = []
    by_key = {(item.expander, item.outcome): item for item in declarations}
    for scope in workflow.scopes:
        for endpoint in scope.workflow.map_item_ports:
            declaration = by_key.get((endpoint.expander, endpoint.expansion_outcome))
            if declaration is None:
                reject(EffectCode.MISSING)
            if declaration.membership_port != endpoint.membership_port:
                reject(EffectCode.CONTRADICTORY)
    for dynamic_map in maps:
        member_operation = _node_operation(workflow.workflow, dynamic_map.member)
        member_inputs = {item.name: item.artifact_type for item in member_operation.inputs}
        if dynamic_map.item_input is not None and any(
            fact.declaration.node == dynamic_map.member and fact.declaration.port == dynamic_map.item_input
            for fact in (() if bound is None else bound.receipt.sources)
        ):
            reject(EffectCode.CONTRADICTORY)
        map_replacements: list[InputBinding] = []
        for outcome_name in dynamic_map.expansion_outcomes:
            declaration = by_key[(dynamic_map.expander, outcome_name)]
            operation = _node_operation(workflow.workflow, dynamic_map.expander)
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
            _add_materialization_schema(schemas, collection_type, declaration.item_type, "collection", 0)
        if map_replacements:
            replacement_choices.append(tuple(map_replacements))
    for replacements in product(*replacement_choices):
        try:
            validate_dynamic_input_summaries(workflow.workflow, replacements)
        except ContractViolation as error:
            if error.code is ValidationCode.MISSING:
                reject(EffectCode.MISSING)
            if error.code is ValidationCode.CONTRADICTORY:
                reject(EffectCode.CONTRADICTORY)
            raise


def _validate_execution_fact_capacity(
    prepared: PreparedPlan,
    productions: tuple[EvidenceProductionDecl, ...],
    limits: AssessmentLimits,
) -> None:
    """Prove structural and operation fact storage from the finite reservation recipe."""
    root = prepared.workflow.workflow
    operation_nodes: dict[NodeId, OperationSpec] = {}
    subgraph_nodes: dict[NodeId, OperationSpec] = {}
    pending = [root]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        for node in current.nodes:
            if isinstance(node, OperationNode):
                operation_nodes[node.id] = node.operation
            else:
                subgraph_nodes[node.id] = node.operation
                pending.append(node.body)
    port_facts = 0
    provenance_edges = 0
    assessment_facts = 0
    mapped_expanders = {
        declaration.expander: declaration.max_children
        for scope in prepared.workflow.scopes
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


def _add_materialization_schema(
    schemas: dict[ArtifactType, tuple[Literal["scalar", "collection"], ArtifactType, int]],
    output_type: ArtifactType,
    item_type: ArtifactType,
    kind: Literal["single", "collection"],
    minimum: int,
) -> None:
    scalar = ("scalar", item_type, 1)
    existing_item = schemas.get(item_type)
    if existing_item is not None and existing_item[0] == "collection":
        reject(EffectCode.CONTRADICTORY)
    schemas.setdefault(item_type, scalar)
    candidate = ("scalar", item_type, 1) if kind == "single" else ("collection", item_type, minimum)
    existing = schemas.get(output_type)
    if existing is not None and existing != candidate:
        reject(EffectCode.CONTRADICTORY)
    schemas[output_type] = candidate


def _validate_policy(policy: OperationExecutionPolicy, has_decision: bool) -> None:
    if policy.request is not None and policy.request.retry_owner == "implementation":
        reject(EffectCode.UNSUPPORTED)
    capabilities = [item.capability for item in policy.implementations]
    operation = capabilities[0].operation
    if any(item.operation != operation for item in capabilities):
        reject(EffectCode.CONTRADICTORY)
    if policy.kind in {"local", "decision"}:
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
    if (policy.kind == "decision") != has_decision:
        reject(EffectCode.CONTRADICTORY)
    outcomes = {item.name: item.category for item in operation.outcomes}
    if policy.kind != "decision" and (not policy.result_outcomes or not policy.result_outcomes <= outcomes.keys()):
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


def _expected_runtime_keys(kind: ExecutionKind, results: frozenset[str]) -> set[tuple[str, str | None, str | None]]:
    keys: set[tuple[str, str | None, str | None]] = {("result", item, None) for item in results}
    failures = (
        {"permanent", "implementation_exception"}
        if kind == "decision"
        else {
            "rejected_before_acceptance",
            "retryable",
            "malformed_response",
            "permanent",
            "transport_unknown",
            "implementation_exception",
        }
    )
    keys.update(("failure", None, item) for item in failures)
    conditions = {"cancel_before_start", "cancel_after_start", "artifact_limit_exhausted", "deadline_exhausted"}
    if kind == "external":
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
        if (
            len(item.supported_findings) > limits.max_findings_per_production
            or len(item.absence_queries) > limits.max_absence_queries
        ):
            reject(EffectCode.LIMIT_EXCEEDED)
        if any(len(value.code.encode()) > limits.max_finding_code_bytes for value in item.supported_findings):
            reject(EffectCode.LIMIT_EXCEEDED)


def _validate_services(admitted: AdmittedExecutionPlan, services: ExecutionServices) -> None:
    require_instance(services.limits, ExecutionLimits)
    require_instance(services.decision_limits, DecisionLimits)
    if any(item.max_lifetime_ns > services.decision_limits.max_lifetime_ns for item in admitted.decisions):
        reject(EffectCode.LIMIT_EXCEEDED)
    if any(
        item.materialization.kind == "collection" and item.bounds.max_items > services.limits.max_collection_items
        for item in admitted.context.adaptive_retrievals
    ):
        reject(EffectCode.LIMIT_EXCEEDED)
    bound_context = admitted.context.bound_context
    if bound_context is not None and any(
        item.declaration.materialization.kind == "collection"
        and item.declaration.bounds.max_items > services.limits.max_collection_items
        for item in bound_context.receipt.sources
    ):
        reject(EffectCode.LIMIT_EXCEEDED)
    _validate_initial_materialization_capacity(admitted, services.limits)
    if not isinstance(services.handles, tuple) or any(
        not isinstance(item, ImplementationHandle) for item in services.handles
    ):
        reject(EffectCode.INVALID_TYPE)
    retained = [item for policy in admitted.policies for item in policy.implementations]
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
        if (handle.local is None) == (handle.transport is None):
            reject(EffectCode.CONTRADICTORY)
        if policy.kind == "external":
            if handle.transport is None or not isinstance(handle.transport, RequestTransport):
                reject(EffectCode.INVALID_TYPE)
        elif handle.local is None or not isinstance(handle.local, LocalCallable):
            reject(EffectCode.INVALID_TYPE)
        if capability.resource_lifetime == "stateless" and handle.resource is not None:
            reject(EffectCode.CONTRADICTORY)
        if capability.resource_lifetime != "stateless":
            if handle.resource is None:
                reject(EffectCode.MISSING)
            expected_owner = "caller" if capability.resource_lifetime == "caller_owned" else "sdk"
            if handle.resource.owner != expected_owner or handle.resource.safe_detachment != policy.safe_detachment:
                reject(EffectCode.CONTRADICTORY)
    if not isinstance(services.context_resources, tuple) or any(
        not isinstance(item, ContextResource) for item in services.context_resources
    ):
        reject(EffectCode.INVALID_TYPE)
    expected_context = {
        (item.source, item) for item in admitted.context.context_capabilities if "adaptive_retrieval" in item.uses
    }
    actual_context = {(item.source, item.capability) for item in services.context_resources}
    if actual_context != expected_context or len(actual_context) != len(services.context_resources):
        reject(EffectCode.MISSING if actual_context < expected_context else EffectCode.EXTRA)


def _validate_initial_materialization_capacity(admitted: AdmittedExecutionPlan, limits: ExecutionLimits) -> None:
    prepared = admitted.context.prepared
    datum_text = {item.id: item.text for item in prepared.data.datums}
    artifact_count = len(prepared.bound_inputs)
    artifact_bytes = sum(len(datum_text[item.source].encode()) for item in prepared.bound_inputs)
    provenance_edges = 0
    context = admitted.context.bound_context
    if context is not None:
        declarations = {item.identity: item.declaration for item in context.receipt.sources}
        groups: dict[BindingDeclarationId, list[BoundTextArtifact]] = {}
        for artifact in context.artifacts:
            groups.setdefault(artifact.reference.declaration, []).append(artifact)
        for identity, items in groups.items():
            declaration = declarations[identity]
            if declaration.materialization.kind == "collection" and len(items) > limits.max_collection_items:
                reject(EffectCode.LIMIT_EXCEEDED)
            item_bytes = sum(len(item.text.encode()) for item in items)
            artifact_count += len(items)
            artifact_bytes += item_bytes
            if declaration.materialization.kind == "collection":
                artifact_count += 1
                artifact_bytes += item_bytes
                provenance_edges += len(items)
    if artifact_count > limits.max_runtime_artifacts or artifact_bytes > limits.max_runtime_artifact_bytes:
        reject(EffectCode.LIMIT_EXCEEDED)
    if provenance_edges > admitted.assessment_limits.max_provenance_edges:
        reject(EffectCode.LIMIT_EXCEEDED)


def _validate_absences(admitted: AdmittedExecutionPlan, services: ExecutionServices) -> None:
    if not isinstance(services.absence_revisions, tuple) or any(
        not isinstance(item, tuple) or len(item) != 2 for item in services.absence_revisions
    ):
        reject(EffectCode.INVALID_TYPE)
    for query, revision in services.absence_revisions:
        require_count(query)
        require_count(revision, positive=True)
    queries = [item[0] for item in services.absence_revisions]
    if len(queries) != len(set(queries)):
        reject(EffectCode.DUPLICATE)
    expected = {query for item in admitted.assessment_productions for query in item.absence_queries}
    if set(queries) != expected:
        reject(EffectCode.MISSING)
