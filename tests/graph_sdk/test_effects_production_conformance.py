# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate the frozen effects corpus through the production request reducer."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectRejected
from anonymizer.engine.graph_sdk.binding import start_initial_binding
from anonymizer.engine.graph_sdk.capabilities import ImplementationRef, PreparationRejected
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
from anonymizer.engine.graph_sdk.executor import (
    AssessmentLimits,
    DecisionDeclaration,
    DecisionLimits,
    DecisionOutcome,
    DecisionResponse,
    DecisionWaitId,
    ExecutionImplementation,
    ExecutionLimits,
    ExecutionServices,
    ImplementationHandle,
    LocalDecisionWait,
    OperationExecutionPolicy,
    RuntimeOutcome,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import BoundInput
from anonymizer.engine.graph_sdk.requests import (
    AcceptFailure,
    AcceptResult,
    AssociationInput,
    AssociationResult,
    BindingAssociation,
    BindingDeclarationId,
    BindingId,
    Dispatch,
    ExactUsage,
    ExternalSettlement,
    InvocationRequestScope,
    MarkLost,
    ObserveSettlement,
    PhysicalRequestId,
    PhysicalRequestPolicy,
    PortArtifact,
    RequestCancel,
    RequestPolicyBinding,
    RequestState,
    Reserve,
    ScopeCancel,
    SemanticAssociation,
    StopAcknowledged,
    StopConfirmed,
    TextArtifactValue,
    UnknownUsage,
    advance_requests,
    bind_request_policies,
    initialize_requests,
)
from anonymizer.engine.graph_sdk.resources import ResourceLease, close_resource
from anonymizer.graph._values import ActivationKey, ArtifactRef, InvocationId, PlanId, TaskAttemptId
from anonymizer.graph.workflow import (
    ArtifactType,
    DynamicScope,
    InputPort,
    NodeId,
    OperationNode,
    WorkflowId,
    admit_activation_workflow,
    admit_static_workflow,
)
from tests.graph_sdk.test_binding import _context_workflow
from tests.graph_sdk.test_preparation import _capability, _data, _limits, _prepare, _workflow

CORPUS = Path(__file__).parent / "reference" / "effects_v1_cases.json"
REQUEST_FAMILIES = {"budgets", "keyed", "retry", "races", "inflight"}
_BASE_CASES = tuple(
    case
    for case in json.loads(CORPUS.read_bytes())
    if (case["family"] in REQUEST_FAMILIES or case["case_id"] == "binding/adaptive_semantic_outcome_independent")
    and case["boundary"] == "runtime"
)
CASES = tuple(
    [*(_BASE_CASES)]
    + [
        {
            **case,
            "case_id": f"{case['case_id']}::{trace['name']}",
            "events": trace["events"],
            "expected": trace["expected"],
        }
        for case in _BASE_CASES
        for trace in case["traces"]
    ]
)
RESOURCE_CASES = tuple(
    case
    for case in json.loads(CORPUS.read_bytes())
    if case["case_id"]
    in {
        "resources/caller_left_open",
        "resources/sdk_closed",
        "resources/sdk_close_failed",
        "resources/sdk_close_unknown",
    }
)
BINDING_RESULT_SHAPE_CASES = tuple(
    case
    for case in json.loads(CORPUS.read_bytes())
    if case["case_id"]
    in {
        "binding/source_result_wrong_outcome",
        "binding/source_result_outputs_present",
        "binding/source_result_consumed_present",
    }
)
ADMISSION_CASES = tuple(case for case in json.loads(CORPUS.read_bytes()) if case["family"] == "admission")
REAL_BINDING_CASE_IDS = {
    "binding/two_sources_same_key",
    "binding/one_source_two_declarations",
    "binding/required_failure_preserves_prior",
    "binding/optional_failure_partial",
    "binding/omitted_optional",
    "binding/response_reordered",
    "binding/oversize_no_truncation",
    "binding/exact_item_byte_bounds",
    "binding/one_over_byte_bound",
    "binding/one_over_item_bound",
    "binding/explicit_omission_preserves_prior",
    "binding/oversize_retrieved_known_usage",
    "binding/optional_oversize_partial",
    "binding/source_failure_explicit_no_settlement",
}
REAL_BINDING_CASES = tuple(case for case in json.loads(CORPUS.read_bytes()) if case["case_id"] in REAL_BINDING_CASE_IDS)


def _assessment_limits() -> AssessmentLimits:
    return AssessmentLimits(
        max_productions=0,
        max_findings_per_production=0,
        max_finding_code_bytes=0,
        max_absence_queries=0,
        max_assessment_facts=0,
        max_port_facts=2,
        max_provenance_edges=2,
    )


def _runtime_row(value: dict[str, Any]) -> RuntimeOutcome:
    return RuntimeOutcome(
        condition=value["condition"],
        reported_outcome=value.get("reported_outcome"),
        failure=value.get("failure"),
        outcome=value.get("outcome"),
        category=value["category"],
    )


def _valid_runtime_rows(kind: str, outcomes: frozenset[str]) -> tuple[RuntimeOutcome, ...]:
    rows = [
        RuntimeOutcome(
            condition="result",
            reported_outcome=outcome,
            failure=None,
            outcome=outcome,
            category="success",
        )
        for outcome in outcomes
    ]
    failures = (
        ("permanent", "implementation_exception")
        if kind == "decision"
        else (
            "rejected_before_acceptance",
            "retryable",
            "malformed_response",
            "permanent",
            "transport_unknown",
            "implementation_exception",
        )
    )
    rows.extend(
        RuntimeOutcome(
            condition="failure",
            reported_outcome=None,
            failure=failure,
            outcome=None,
            category="failure",
        )
        for failure in failures
    )
    categories = {
        "cancel_before_start": "blocked",
        "cancel_after_start": "cancelled",
        "artifact_limit_exhausted": "blocked",
        "deadline_exhausted": "failure",
    }
    if kind == "external":
        categories.update(
            {
                "cancel_after_dispatch": "cancelled",
                "lost": "lost",
                "request_inconsistent": "inconsistent",
                "budget_exhausted": "blocked",
                "request_limit_exhausted": "blocked",
            }
        )
    rows.extend(
        RuntimeOutcome(
            condition=cast(Any, condition),
            reported_outcome=None,
            failure=None,
            outcome=None,
            category=cast(Any, category),
        )
        for condition, category in categories.items()
    )
    return tuple(rows)


def _admit_case(case: dict[str, Any]) -> None:
    declaration = cast(dict[str, Any], case["declaration"])
    admitted_declaration = cast(dict[str, Any], declaration.get("admitted", declaration))
    execution_policies = admitted_declaration.get("execution_policies")
    kind = "local"
    if isinstance(execution_policies, list) and execution_policies and isinstance(execution_policies[0], dict):
        kind = cast(str, execution_policies[0].get("kind", "local"))
    external = kind == "external"
    workflow, node, artifact_type = _workflow(requests=3 if external else 0, with_input=kind == "decision")
    primary = _capability(workflow, external=external)
    if external:
        primary = replace(primary, max_physical_requests_per_activation=2)
    bound_inputs = ()
    data = _data(1 if kind == "decision" else 2)
    if kind == "decision":
        target = next(iter(data.targets))
        bound_inputs = (BoundInput(target=target, source=target, port="input", artifact_type=artifact_type),)
    capability_maximum = cast(dict[str, int], declaration.get("admission_limits", {})).get("max_capabilities", 2)
    prepared = _prepare(
        data=data,
        workflow=workflow,
        capability=primary,
        bound_inputs=bound_inputs,
        limits=_limits(capabilities=capability_maximum),
    )
    context = admit_context_plan(
        prepared=prepared,
        bound_context=None,
        adaptive_retrievals=(),
        context_capabilities=(),
    )
    physical = PhysicalRequestPolicy(
        visibility="dispatch_and_settlement",
        pre_dispatch_control="executor",
        retry_owner="executor",
        replay="idempotent",
        max_attempts=2,
    )
    if isinstance(execution_policies, list) and any(
        isinstance(item, dict) and item.get("retry_owner") == "implementation" for item in execution_policies
    ):
        physical = replace(physical, retry_owner="implementation")
    capabilities = [primary]
    catalog = cast(list[dict[str, Any]], admitted_declaration.get("capability_catalog", []))
    if not execution_policies and catalog:
        catalog_implementations: dict[str, ImplementationRef] = {
            cast(str, catalog[0]["implementation"]): primary.implementation
        }
        for item in catalog[1:]:
            name = cast(str, item["implementation"])
            implementation = catalog_implementations.setdefault(
                name,
                ImplementationRef(name=f"catalog-{len(catalog_implementations)}", revision=1),
            )
            capabilities.append(replace(primary, implementation=implementation))

    if execution_policies == "invalid":
        policies: Any = execution_policies
    elif execution_policies == [0]:
        policies = (0,)
    else:
        raw_policies = cast(list[dict[str, Any]], execution_policies or [{}])
        built: list[OperationExecutionPolicy] = []
        for raw_index, raw in enumerate(raw_policies):
            policy_kind = cast(str, raw.get("kind", kind))
            policy_node = node if raw.get("owner") != "I1" else NodeId.new(workflow=WorkflowId.new())
            default_physical = "P0" if policy_kind == "external" else None
            implementation_values = cast(
                list[dict[str, Any]], raw.get("implementations", [{"physical_policy": default_physical}])
            )
            implementations: list[ExecutionImplementation] = []
            for index, item in enumerate(implementation_values):
                capability = primary
                if index:
                    capability = replace(
                        primary,
                        implementation=ImplementationRef(name=f"alternate-{raw_index}-{index}", revision=1),
                    )
                    capabilities.append(capability)
                request = physical if item.get("physical_policy") == "P0" else None
                if item.get("physical_policy") == "P1":
                    request = replace(physical, max_attempts=3)
                    capability = replace(capability, max_physical_requests_per_activation=3)
                    capabilities[-1 if index else 0] = capability
                implementations.append(
                    ExecutionImplementation(
                        implementation=capability.implementation,
                        configuration=capability.configuration,
                        capability=capability,
                        request=request,
                    )
                )
            result_values = frozenset(
                cast(list[str], raw.get("result_outcomes", admitted_declaration.get("declared_outcomes", [])))
            )
            declared_rows = tuple(
                _runtime_row(item)
                for item in cast(list[dict[str, Any]], admitted_declaration.get("runtime_mappings", []))
            )
            rows = declared_rows
            if not declared_rows and "required_runtime_conditions" not in admitted_declaration:
                rows = _valid_runtime_rows(policy_kind, result_values)
            elif "execution_policies" not in admitted_declaration and case["case_id"] in {
                "admission/contradictory",
                "admission/wrong_category",
                "admission/unknown_outcome",
                "admission/duplicate_runtime_mapping",
            }:
                replacement_keys = {(item.condition, item.reported_outcome, item.failure) for item in declared_rows}
                rows = (
                    tuple(
                        item
                        for item in _valid_runtime_rows(policy_kind, result_values)
                        if (item.condition, item.reported_outcome, item.failure) not in replacement_keys
                    )
                    + declared_rows
                )
            built.append(
                OperationExecutionPolicy(
                    node=policy_node,
                    kind=cast(Any, policy_kind),
                    request=physical if raw.get("physical_policy") == "P0" else None,
                    safe_detachment="forbidden",
                    implementations=tuple(implementations),
                    result_outcomes=result_values,
                    runtime_outcomes=rows,
                )
            )
        policies = tuple(built)
    decisions = ()
    if kind == "decision":
        decisions = (
            DecisionDeclaration(
                node=node,
                artifact_port="input",
                outcomes=(DecisionOutcome(decision="approve", outcome="ok"),),
                max_lifetime_ns=1,
            ),
        )
    admitted = admit_execution_plan(
        context=context,
        capabilities=tuple(capabilities),
        policies=cast(Any, policies),
        decisions=decisions,
        assessment_productions=(),
        assessment_limits=_assessment_limits(),
    )
    if case["boundary"] == "pre_execution":
        changed = replace(capabilities[-1], max_physical_requests_per_activation=3)
        asyncio.run(
            start_execution(
                admitted=admitted,
                capabilities=tuple([*capabilities[:-1], changed]),
                services=ExecutionServices(
                    handles=(),
                    context_resources=(),
                    limits=ExecutionLimits(
                        max_local_in_flight=1,
                        max_remote_outstanding=1,
                        max_runtime_artifacts=1,
                        max_runtime_artifact_bytes=1,
                        max_collection_items=1,
                    ),
                    decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
                    clock=_ZeroClock(),
                ),
            )
        )


class _ZeroClock:
    def now_ns(self) -> int:
        return 0


class _DecisionProvider:
    async def run(self, request: tuple[AssociationInput, ...]) -> LocalDecisionWait:
        artifact = request[0].inputs[0].artifact
        assert artifact is not None
        assert isinstance(request[0].association, SemanticAssociation)
        return LocalDecisionWait(association=request[0].association, artifact=artifact)


async def _assert_decision_submission(case_id: str, expected: str) -> None:
    workflow, node, artifact_type = _workflow(with_input=True)
    data = _data(1)
    target = next(iter(data.targets))
    capability = _capability(workflow)
    prepared = _prepare(
        data=data,
        workflow=workflow,
        capability=capability,
        bound_inputs=(BoundInput(target=target, source=target, port="input", artifact_type=artifact_type),),
    )
    context = admit_context_plan(
        prepared=prepared,
        bound_context=None,
        adaptive_retrievals=(),
        context_capabilities=(),
    )
    implementation = ExecutionImplementation(
        implementation=capability.implementation,
        configuration=capability.configuration,
        capability=capability,
        request=None,
    )
    policy = OperationExecutionPolicy(
        node=node,
        kind="decision",
        request=None,
        safe_detachment="forbidden",
        implementations=(implementation,),
        result_outcomes=frozenset(),
        runtime_outcomes=tuple(
            RuntimeOutcome(
                condition=condition,
                reported_outcome=None,
                failure=failure,
                outcome=None,
                category=category,
            )
            for condition, failure, category in (
                ("failure", "permanent", "failure"),
                ("failure", "implementation_exception", "failure"),
                ("cancel_before_start", None, "blocked"),
                ("cancel_after_start", None, "cancelled"),
                ("artifact_limit_exhausted", None, "blocked"),
                ("deadline_exhausted", None, "blocked"),
            )
        ),
    )
    admitted = admit_execution_plan(
        context=context,
        capabilities=(capability,),
        policies=(policy,),
        decisions=(
            DecisionDeclaration(
                node=node,
                artifact_port="input",
                outcomes=(DecisionOutcome(decision="approve", outcome="ok"),),
                max_lifetime_ns=10,
            ),
        ),
        assessment_productions=(),
        assessment_limits=_assessment_limits(),
    )
    running = await start_execution(
        admitted=admitted,
        capabilities=(capability,),
        services=ExecutionServices(
            handles=(
                ImplementationHandle(
                    implementation=capability.implementation,
                    operation=capability.operation,
                    configuration=capability.configuration,
                    local=_DecisionProvider(),
                    transport=None,
                    resource=None,
                ),
            ),
            context_resources=(),
            limits=ExecutionLimits(
                max_local_in_flight=1,
                max_remote_outstanding=0,
                max_runtime_artifacts=1,
                max_runtime_artifact_bytes=100,
                max_collection_items=1,
            ),
            decision_limits=DecisionLimits(max_pending=1, max_lifetime_ns=10),
            clock=_ZeroClock(),
        ),
    )
    while not running.pending_decisions():
        await asyncio.sleep(0)
    wait = running.pending_decisions()[0]
    response = DecisionResponse(
        wait=wait.wait,
        workflow=wait.workflow,
        artifact=wait.artifact,
        decision="approve",
    )
    if case_id == "decisions/stale_artifact":
        response = replace(response, artifact=ArtifactRef(invocation=wait.artifact.invocation, key=1, version=1))
    elif case_id == "decisions/foreign_workflow":
        response = replace(response, workflow=WorkflowId.new())
    elif case_id == "decisions/foreign_wait":
        response = replace(
            response,
            wait=DecisionWaitId.new(invocation=InvocationId.new(plan=wait.activation.invocation.plan)),
        )
    elif case_id == "decisions/unknown_decision":
        response = replace(response, decision="deny")
    elif case_id == "decisions/duplicate_response":
        running.submit_decision(response)
    before = running.pending_decisions()
    with pytest.raises(EffectRejected) as rejected:
        running.submit_decision(response)
    assert rejected.value.code.value == expected
    assert running.pending_decisions() == before
    if case_id != "decisions/duplicate_response":
        running.submit_decision(
            DecisionResponse(
                wait=wait.wait,
                workflow=wait.workflow,
                artifact=wait.artifact,
                decision="approve",
            )
        )
    result = await running.wait()
    assert result.record.terminals[0].category == "success"
    assert not result.pending_decisions


@pytest.mark.parametrize(
    ("case_id", "expected"),
    (
        ("decisions/stale_artifact", "foreign_owner"),
        ("decisions/foreign_workflow", "foreign_owner"),
        ("decisions/foreign_wait", "foreign_owner"),
        ("decisions/unknown_decision", "unsupported"),
        ("decisions/duplicate_response", "duplicate"),
    ),
)
def test_decision_submission_rejections_use_public_running_execution(case_id: str, expected: str) -> None:
    asyncio.run(_assert_decision_submission(case_id, expected))


@pytest.mark.parametrize("case", ADMISSION_CASES, ids=lambda case: cast(str, case["case_id"]))
def test_admission_corpus_case_through_production(case: dict[str, Any]) -> None:
    rejected: str | None = None
    try:
        _admit_case(case)
    except (EffectRejected, PreparationRejected) as exc:
        rejected = exc.code.value
    expected = cast(dict[str, str], case["expected"])
    assert ("rejected" if rejected else "accepted") == expected["status"]
    assert rejected == expected.get("code")


@pytest.mark.parametrize(
    ("duplicate", "expected"),
    ((False, None), (True, "duplicate")),
)
def test_admission_capability_identity_survives_a_raised_limit(duplicate: bool, expected: str | None) -> None:
    workflow, node, _ = _workflow(requests=3)
    primary = replace(_capability(workflow, external=True), max_physical_requests_per_activation=2)
    alternate = (
        primary
        if duplicate
        else replace(
            primary,
            implementation=ImplementationRef(name="alternate", revision=1),
        )
    )
    prepared = _prepare(
        data=_data(2),
        workflow=workflow,
        capability=primary,
        limits=_limits(capabilities=2),
    )
    context = admit_context_plan(
        prepared=prepared,
        bound_context=None,
        adaptive_retrievals=(),
        context_capabilities=(),
    )
    request = PhysicalRequestPolicy(
        visibility="dispatch_and_settlement",
        pre_dispatch_control="executor",
        retry_owner="executor",
        replay="idempotent",
        max_attempts=2,
    )
    policy = OperationExecutionPolicy(
        node=node,
        kind="external",
        request=request,
        safe_detachment="forbidden",
        implementations=tuple(
            ExecutionImplementation(
                implementation=item.implementation,
                configuration=item.configuration,
                capability=item,
                request=request,
            )
            for item in (primary, alternate)
        ),
        result_outcomes=frozenset({"ok"}),
        runtime_outcomes=_valid_runtime_rows("external", frozenset({"ok"})),
    )
    rejected: str | None = None
    try:
        admit_execution_plan(
            context=context,
            capabilities=(primary, alternate),
            policies=(policy,),
            decisions=(),
            assessment_productions=(),
            assessment_limits=_assessment_limits(),
        )
    except EffectRejected as exc:
        rejected = exc.code.value
    assert rejected == expected


class _CaseProvider:
    def __init__(
        self,
        *,
        events: list[dict[str, Any]],
        sources: dict[str, ContextSourceRef],
        association_names: dict[object, str],
        request_names: dict[PhysicalRequestId, str],
    ) -> None:
        self.events = events
        self.sources = sources
        self.association_names = association_names
        self.request_names = request_names

    async def retrieve(
        self,
        *,
        request: PhysicalRequestId,
        association: BindingAssociation,
        selector: ContextSelector,
        bounds: RetrievalBounds,
    ) -> SourceResponse | SourceFailure:
        del selector, bounds
        event = self.events.pop(0)
        label = cast(str, event.get("association", event.get("items", [{}])[0].get("association")))
        self.association_names[association] = label
        self.request_names[request] = cast(str, event["request"])
        settlement_value = cast(dict[str, Any] | None, event.get("settlement"))
        settlement = None
        if settlement_value is not None:
            settlement = ExternalSettlement(
                request=request,
                disposition=settlement_value["disposition"],
                usage=_usage(settlement_value["usage"]),
                remote_stopped=settlement_value["remote_stopped"],
            )
        source = self.sources[cast(str, event["source"])]
        if event["kind"] == "source_failure":
            return SourceFailure(
                source=source,
                failure=event["failure"],
                settlement=settlement,
                disposition=event.get("disposition", "failed"),
            )
        assert settlement is not None
        return SourceResponse(
            source=source,
            items=tuple(
                SourceItem(
                    association=association,
                    key=item["key"],
                    version=item["version"],
                    text=item["text"],
                )
                for item in cast(list[dict[str, Any]], event["items"])
            ),
            settlement=settlement,
        )

    async def cancel(self, request: PhysicalRequestId) -> StopConfirmed:
        del request
        return StopConfirmed(usage=ExactUsage(input_units=0, output_units=0))


@pytest.mark.parametrize("case", REAL_BINDING_CASES, ids=lambda case: cast(str, case["case_id"]))
def test_binding_corpus_case_through_real_provider(case: dict[str, Any]) -> None:
    asyncio.run(_assert_binding_corpus_case(case))


async def _assert_binding_corpus_case(case: dict[str, Any]) -> None:
    raw = cast(dict[str, Any], case["declaration"])
    event_values = cast(list[dict[str, Any]], case["events"])
    workflow, node, artifact_type = _context_workflow()
    max_response_items = max(
        (len(cast(list[object], event.get("items", []))) for event in event_values if event["kind"] == "source_result"),
        default=1,
    )
    item_type = artifact_type
    if max_response_items > 1:
        collection_type = ArtifactType(name="text_collection", revision=1)
        static = workflow.workflow
        operation_node = next(item for item in static.nodes if isinstance(item, OperationNode))
        operation = replace(
            operation_node.operation,
            inputs=(InputPort(name="input", artifact_type=collection_type),),
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
        artifact_type = collection_type
    binding_sources = cast(dict[str, str], raw["binding_declarations"])
    order = list(dict.fromkeys(event["association"] for event in event_values if event["kind"] == "bind_policy"))
    assert set(order) == set(binding_sources)
    data = _data(len(order))
    targets = sorted(data.targets, key=repr)
    source_refs = {name: ContextSourceRef(name=name, revision=1) for name in sorted(set(binding_sources.values()))}
    policies = {name: _policy(value) for name, value in cast(dict[str, dict[str, Any]], raw["policies"]).items()}
    response_events: dict[str, list[dict[str, Any]]] = {
        name: [
            event
            for event in event_values
            if event["kind"] in {"source_result", "source_failure"} and event["source"] == name
        ]
        for name in source_refs
    }
    association_names: dict[object, str] = {}
    request_names: dict[PhysicalRequestId, str] = {}
    providers = {
        name: _CaseProvider(
            events=response_events[name],
            sources=source_refs,
            association_names=association_names,
            request_names=request_names,
        )
        for name in source_refs
    }
    declaration_limits = cast(dict[str, int], raw["binding_limits"])
    requirements = cast(dict[str, str], raw.get("binding_requirements", {}))
    declarations = tuple(
        InitialContextDecl(
            target=target,
            node=node,
            port="input",
            artifact_type=artifact_type,
            source=source_refs[binding_sources[label]],
            selector=ContextSelector(fields=()),
            requirement=cast(Any, requirements.get(label, "required")),
            bounds=RetrievalBounds(
                max_items=max_response_items,
                max_bytes=max(1, declaration_limits["max_bytes"]),
                max_requests=policies["P0"].max_attempts,
            ),
            materialization=ContextMaterialization(
                kind="single" if max_response_items == 1 else "collection",
                item_type=item_type,
            ),
        )
        for label, target in zip(order, targets, strict=True)
    )
    capabilities = tuple(
        ContextSourceCapability(
            source=source_refs[name],
            artifact_type=item_type,
            uses=frozenset({"initial_binding"}),
            execution="async",
            resource_owner="caller",
            cancellation="cooperative_ack",
            settlement="explicit_ack",
            usage="exact",
            request=policies["P0"],
            safe_detachment="forbidden",
        )
        for name in source_refs
    )
    resources = tuple(
        ContextResource(
            source=capability.source,
            capability=capability,
            lease=ResourceLease.create(
                owner="caller", safe_detachment="forbidden", handle=providers[capability.source.name]
            ),
            factory=None,
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
    expected = cast(dict[str, Any], case["expected"])["state"]
    source_names = {fact.identity: label for label, fact in zip(order, result.receipt.sources, strict=True)}
    assert {source_names[fact.identity]: fact.terminal for fact in result.receipt.sources} == expected[
        "binding_sources"
    ]
    assert result.receipt.terminal == expected["binding_terminal"]
    actual_artifacts = [
        {
            "identity": f"{source_names[item.reference.declaration]}:{item.reference.key}:{item.reference.version}",
            "source": item.source.name,
            "text": item.text,
        }
        for item in result.receipt.artifacts
    ]
    assert sorted(actual_artifacts, key=lambda item: item["identity"]) == sorted(
        expected["artifacts"], key=lambda item: item["identity"]
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


def test_binding_foreign_target_admission_case_through_production() -> None:
    case = next(
        item for item in json.loads(CORPUS.read_bytes()) if item["case_id"] == "binding/invalid_local_zero_effects"
    )
    rejected = asyncio.run(_reject_foreign_binding_target())
    assert rejected == cast(dict[str, str], case["expected"])["code"]


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


class _Closable:
    def __init__(self, disposition: str) -> None:
        self.disposition = disposition

    async def close(self) -> None:
        if self.disposition == "close_failed":
            raise RuntimeError("close failed")


@pytest.mark.parametrize("case", RESOURCE_CASES, ids=lambda case: cast(str, case["case_id"]))
def test_resource_corpus_case_through_production(case: dict[str, Any]) -> None:
    events = cast(list[dict[str, Any]], case["events"])
    resource = events[0]
    close_event = events[1]
    disposition = cast(str, close_event.get("disposition", "closed"))
    lease = ResourceLease.create(
        owner=resource["owner"],
        safe_detachment=resource["safe_detachment"],
        handle=object() if disposition == "close_unknown" else _Closable(disposition),
    )
    fact = asyncio.run(close_resource(lease))
    expected = cast(dict[str, Any], case["expected"])["state"]["cleanup"]["Q0"]
    assert fact.disposition == expected


@pytest.mark.parametrize("case", BINDING_RESULT_SHAPE_CASES, ids=lambda case: cast(str, case["case_id"]))
def test_binding_result_shape_case_through_production(case: dict[str, Any]) -> None:
    event = cast(list[dict[str, Any]], case["events"])[-1]
    binding = BindingId.new()
    association = BindingAssociation(declaration=BindingDeclarationId.new(binding=binding, ordinal=0))
    outputs = tuple(
        PortArtifact(
            port=cast(str, item["port"]),
            artifact_type=ArtifactType(name="text", revision=1),
            artifact=None,
            value=TextArtifactValue(text="unexpected"),
        )
        for item in cast(list[dict[str, object]], event.get("outputs", []))
    )
    rejected: str | None = None
    try:
        AssociationResult(
            association=association,
            outcome=cast(str, event.get("outcome", "retrieved")),
            outputs=outputs,
            consumed_context_ports=frozenset(cast(list[str], event.get("consumed_context_ports", []))),
        )
    except EffectRejected as exc:
        rejected = exc.code.value
    expected = cast(dict[str, str], case["expected"])
    assert ("rejected" if rejected else "accepted") == expected["status"]
    assert rejected == expected["code"]


def _policy(value: dict[str, Any]) -> PhysicalRequestPolicy:
    return PhysicalRequestPolicy(
        visibility="dispatch_and_settlement",
        pre_dispatch_control="executor",
        retry_owner=value["retry_owner"],
        replay=value["replay"],
        max_attempts=cast(int, value["max_attempts"]),
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: cast(str, case["case_id"]))
def test_request_corpus_case_through_production(case: dict[str, Any]) -> None:
    declaration = cast(dict[str, Any], case["declaration"])
    invocation = InvocationId.new(plan=PlanId.new())
    scope = InvocationRequestScope(invocation=invocation)
    policies = {
        name: _policy(value) for name, value in cast(dict[str, dict[str, Any]], declaration["policies"]).items()
    }
    association_names = {
        item
        for event in cast(list[dict[str, Any]], case["events"])
        for field in ("associations", "returned")
        for item in cast(list[str], event.get(field, ()))
    } | {"T0", "T1"}
    foreign_invocation = InvocationId.new(plan=invocation.plan)
    tasks = {
        name: SemanticAssociation(
            task=TaskAttemptId.new(
                activation=ActivationKey(
                    invocation=foreign_invocation if name.startswith("X") else invocation,
                    occurrence=index,
                    parent=None,
                    iteration=None,
                )
            )
        )
        for index, name in enumerate(sorted(association_names))
    }
    requests = {name: PhysicalRequestId.new(scope=scope) for name in ("R0", "R1", "R2")}
    state = initialize_requests(
        scope=scope,
        hard_limit=cast(int | None, declaration.get("hard_limit")),
        policies=frozenset(policies.values()),
    )
    rejected: str | None = None
    try:
        for event in cast(list[dict[str, Any]], case["events"]):
            state = _apply(state, event, tasks, requests, policies)
    except EffectRejected as exc:
        rejected = exc.code.value
    expected = cast(dict[str, Any], case["expected"])
    assert ("rejected" if rejected else "accepted") == expected["status"]
    if rejected:
        assert rejected == expected["code"]
        return
    expected_state = cast(dict[str, Any], expected["state"])
    actual = _normalize(state, tasks, requests, policies)
    for key in (
        "bindings",
        "dispatched",
        "dispatched_count",
        "denials",
        "terminals",
        "defects",
        "local_in_flight",
        "remote_outstanding",
        "attempts",
        "request_failures",
        "association_terminals",
        "reservations",
        "request_associations",
        "request_policies",
        "reservation_policies",
        "settlements",
        "request_facts",
        "cancel_requested",
        "association_requests",
    ):
        assert actual[key] == expected_state[key], (case["case_id"], key)


def _apply(
    state: RequestState,
    event: dict[str, Any],
    tasks: dict[str, SemanticAssociation],
    requests: dict[str, PhysicalRequestId],
    policies: dict[str, PhysicalRequestPolicy],
) -> RequestState:
    kind = event["kind"]
    if kind == "bind_policy":
        return bind_request_policies(
            state=state,
            binding=RequestPolicyBinding.create(
                association=tasks[event["association"]],
                policies=frozenset(policies[item] for item in event["policies"]),
            ),
        )
    if kind == "reserve":
        return advance_requests(
            state=state,
            event=Reserve(
                request=requests[event["request"]],
                purpose=event["purpose"],
                associations=frozenset(tasks[item] for item in event["associations"]),
                policy=policies[event["policy"]],
            ),
        )
    if kind == "dispatch":
        value = Dispatch(request=requests[event["request"]])
    elif kind == "result":
        value = AcceptResult(
            request=requests[event["request"]],
            results=tuple(
                AssociationResult(
                    association=tasks[item],
                    outcome=event["outcomes"].get(item, "ok"),
                    outputs=(),
                    consumed_context_ports=frozenset(),
                )
                for item in event["returned"]
            ),
        )
    elif kind == "failure":
        value = AcceptFailure(request=requests[event["request"]], failure=event["failure"])
    elif kind == "cancel":
        value = RequestCancel(request=requests[event["request"]])
    elif kind == "scope_cancel":
        value = ScopeCancel()
    elif kind == "stop":
        usage = _usage(event["usage"]) if "usage" in event else object()
        value = StopAcknowledged(request=requests[event["request"]], usage=cast(Any, usage))
    elif kind == "lost":
        value = MarkLost(request=requests[event["request"]])
    elif kind == "settlement":
        value = ObserveSettlement(
            settlement=ExternalSettlement(
                request=requests[event["request"]],
                disposition=event["disposition"],
                usage=_usage(event["usage"]),
                remote_stopped=event["remote_stopped"],
            )
        )
    else:
        raise AssertionError(kind)
    return advance_requests(state=state, event=value)


def _usage(value: object):
    if value == "unknown":
        return UnknownUsage()
    usage = cast(dict[str, int], value)
    return ExactUsage(input_units=usage["input"], output_units=usage["output"])


def _normalize(
    state: Any,
    tasks: dict[str, SemanticAssociation],
    requests: dict[str, PhysicalRequestId],
    policies: dict[str, PhysicalRequestPolicy],
) -> dict[str, object]:
    task_names: dict[object, str] = {value: key for key, value in tasks.items()}
    request_names = {value: key for key, value in requests.items()}
    attempts = {
        name: sum(association in item.associations for item in state.dispatches)
        for association, name in task_names.items()
        if any(association in item.associations for item in state.dispatches)
    }
    request_failures = {
        request_names[item.request]: item.failure for item in state.terminals if item.failure is not None
    }
    association_terminals: dict[str, object] = {}
    for terminal in state.terminals:
        dispatch = next((item for item in state.dispatches if item.request == terminal.request), None)
        if dispatch is not None and terminal.failure is not None:
            association_terminals.update(
                {
                    task_names[item]: {
                        "request": request_names[terminal.request],
                        "failure": terminal.failure,
                        "policy": next(key for key, policy in policies.items() if policy is dispatch.policy),
                    }
                    for item in dispatch.associations
                }
            )
    bindings = {
        task_names[item.association]: sorted(
            key for key, policy in policies.items() if any(policy is retained for retained in item.policies)
        )
        for item in state.bindings
    }
    dispatches = {item.request: item for item in state.dispatches}
    dispatched_ids = frozenset(dispatches)
    settlements = {
        request_names[item.request]: {
            "disposition": item.disposition,
            "usage": (
                "unknown"
                if isinstance(item.usage, UnknownUsage)
                else {"input": item.usage.input_units, "output": item.usage.output_units}
            ),
            "remote_stopped": item.remote_stopped,
        }
        for item in state.settlements
    }
    request_facts: dict[str, object] = {}
    for terminal in state.terminals:
        fact: dict[str, object] = {
            "condition": "request_inconsistent" if terminal.category == "inconsistent" else terminal.category
        }
        if terminal.category == "success":
            fact = {
                "condition": "result",
                "outcomes": {task_names[item.association]: item.outcome for item in terminal.results},
            }
        elif terminal.category == "failure" and terminal.failure is not None:
            fact = {"condition": "failure", "failure": terminal.failure}
        elif terminal.category == "cancelled" and terminal.request in dispatched_ids:
            fact = {"condition": "cancel_after_dispatch"}
        if terminal.request in dispatched_ids:
            request_facts[request_names[terminal.request]] = fact
    return {
        "bindings": bindings,
        "dispatched": [request_names[item.request] for item in state.dispatches],
        "dispatched_count": len(state.dispatches),
        "denials": {task_names[item]: denial.category for denial in state.denials for item in denial.associations},
        "terminals": {request_names[item.request]: item.category for item in state.terminals},
        "defects": [item.code for item in state.defects],
        "local_in_flight": sorted(request_names[item] for item in state.local_in_flight),
        "remote_outstanding": sorted(request_names[item] for item in state.remote_outstanding),
        "attempts": attempts,
        "request_failures": request_failures,
        "association_terminals": association_terminals,
        "reservations": {
            request_names[item.request]: sorted(task_names[value] for value in item.associations)
            for item in getattr(state, "reserved", ())
        },
        "request_associations": {
            request_names[request]: sorted(task_names[item] for item in reservation.associations)
            for request, reservation in dispatches.items()
        },
        "request_policies": {
            request_names[request]: next(key for key, policy in policies.items() if policy is reservation.policy)
            for request, reservation in dispatches.items()
        },
        "reservation_policies": {
            request_names[item.request]: next(key for key, policy in policies.items() if policy is item.policy)
            for item in getattr(state, "reserved", ())
        },
        "settlements": settlements,
        "request_facts": request_facts,
        "cancel_requested": sorted(request_names[item] for item in state.cancel_requested),
        "association_requests": {
            task_names[association]: request_names[item.request]
            for item in state.dispatches
            for association in item.associations
        },
    }
