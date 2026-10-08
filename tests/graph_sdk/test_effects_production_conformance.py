# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate the frozen effects corpus through the production request reducer."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectRejected
from anonymizer.engine.graph_sdk.binding import start_initial_binding
from anonymizer.engine.graph_sdk.capabilities import ImplementationRef, PreparationRejected
from anonymizer.engine.graph_sdk.context import (
    AdaptiveRetrievalDecl,
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
from anonymizer.engine.graph_sdk.data import DataGraph, DataLimits
from anonymizer.engine.graph_sdk.executor import (
    AssessmentLimits,
    BoundInputKey,
    DecisionDeclaration,
    DecisionLimits,
    DecisionOutcome,
    DecisionResponse,
    DecisionWaitId,
    ExecutionImplementation,
    ExecutionLimits,
    ExecutionServices,
    ImplementationHandle,
    InitialCollectionKey,
    LocalCompleted,
    LocalDecisionWait,
    LocalFailure,
    OperationExecutionPolicy,
    OperationOutputKey,
    RootInputKey,
    RuntimeOutcome,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import BoundInput, PreparationConfiguration
from anonymizer.engine.graph_sdk.requests import (
    AcceptFailure,
    AcceptResult,
    AssociationInput,
    AssociationResult,
    BindingAssociation,
    BindingDeclarationId,
    BindingId,
    BindingRequestScope,
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
    TextCollectionItem,
    TextCollectionValue,
    UnknownUsage,
    advance_requests,
    bind_request_policies,
    initialize_requests,
)
from anonymizer.engine.graph_sdk.resources import ResourceLease, close_resource
from anonymizer.graph._values import ActivationKey, ArtifactRef, InvocationId, PlanId, TaskAttemptId
from anonymizer.graph.workflow import (
    ArtifactType,
    ContextInputRef,
    DynamicScope,
    InputPort,
    NodeId,
    OperationNode,
    OutputPort,
    WorkflowId,
    admit_activation_workflow,
    admit_static_workflow,
)
from tests.graph_sdk.test_adaptive_executor import _adaptive_workflow, _Clock, _UnusedTransport
from tests.graph_sdk.test_binding import _context_workflow
from tests.graph_sdk.test_preparation import _capability, _data, _limits, _prepare, _workflow

CORPUS = Path(__file__).parent / "reference" / "effects_v1_cases.json"
REQUEST_FAMILIES = {"budgets", "keyed", "retry", "races", "inflight"}
_BASE_CASES = tuple(
    case
    for case in json.loads(CORPUS.read_bytes())
    if (
        case["family"] in REQUEST_FAMILIES
        or case["case_id"]
        in {
            "binding/adaptive_semantic_outcome_independent",
            "binding/cancelled_late_source_failure",
            "binding/cancelled_late_source_result",
            "binding/cancel_after_dispatch_lost",
            "binding/cancel_before_dispatch",
            "binding/lost_late_source_failure",
            "binding/lost_late_source_result",
            "binding/source_failure_correction_authority",
            "binding/source_failure_retry_authority",
            "binding/success_after_failure_preserves_authority",
            "binding/unsolicited_source_result",
        }
    )
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
LOCAL_BRIDGE_CASES = tuple(
    case
    for case in json.loads(CORPUS.read_bytes())
    if case["case_id"]
    in {
        "bridges/result",
        "bridges/cancel_before_start",
        "bridges/cancel_after_start",
        "bridges/failure_rejected_before_acceptance",
        "bridges/failure_retryable",
        "bridges/failure_malformed_response",
        "bridges/failure_permanent",
        "bridges/failure_transport_unknown",
        "bridges/failure_implementation_exception",
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
ADMISSION_CASES = tuple(
    case
    for case in json.loads(CORPUS.read_bytes())
    if case["family"] == "admission" or case["case_id"] == "retry/implementation_owner_rejected"
)
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
    "binding/wrong_source",
    "binding/missing_result",
    "binding/duplicate_result",
    "binding/foreign_result_association",
    "binding/required_omission_misuse",
    "binding/empty_optional_response_malformed",
    "materialization/initial_single_exact",
    "materialization/initial_single_multiple",
    "materialization/initial_collection_1",
    "materialization/initial_collection_2",
    "materialization/initial_collection_3",
    "materialization/initial_collection_one_over",
    "materialization/initial_collection_0",
    "materialization/initial_canonical_reorder",
    "materialization/initial_duplicate",
    "materialization/initial_outer_count_precedence",
    "materialization/initial_declared_bytes_exact",
    "materialization/initial_declared_bytes_one_over",
    "materialization/initial_single_zero_collection_limit",
    "materialization/initial_unmaterialized_source_result",
}
REAL_BINDING_CASES = tuple(case for case in json.loads(CORPUS.read_bytes()) if case["case_id"] in REAL_BINDING_CASE_IDS)
ADAPTIVE_MATERIALIZATION_CASE_IDS = {
    "materialization/adaptive_single_exact",
    "materialization/adaptive_single_multiple",
    "materialization/adaptive_collection_1",
    "materialization/adaptive_collection_2",
    "materialization/adaptive_collection_3",
    "materialization/adaptive_collection_one_over",
    "materialization/adaptive_collection_0",
    "materialization/adaptive_canonical_reorder",
    "materialization/adaptive_duplicate",
    "materialization/adaptive_outer_count_precedence",
    "materialization/adaptive_single_zero_collection_limit",
    "materialization/adaptive_foreign_association",
    "materialization/adaptive_unmaterialized_result",
    "materialization/adaptive_missing_parent",
    "materialization/adaptive_foreign_parent",
    "materialization/adaptive_invented_parent",
    "materialization/adaptive_missing_source_fact",
    "materialization/adaptive_result_bridge",
    "binding/adaptive_omission_misuse",
}
ADAPTIVE_MATERIALIZATION_CASES = tuple(
    case for case in json.loads(CORPUS.read_bytes()) if case["case_id"] in ADAPTIVE_MATERIALIZATION_CASE_IDS
)
LATE_ADAPTIVE_CASES = tuple(
    case
    for case in json.loads(CORPUS.read_bytes())
    if case["case_id"] in {"materialization/adaptive_late_lost", "materialization/adaptive_late_cancelled"}
)
LATE_INITIAL_CASES = tuple(
    case
    for case in json.loads(CORPUS.read_bytes())
    if case["case_id"]
    in {
        "binding/cancelled_late_source_failure",
        "binding/cancelled_late_source_result",
        "binding/lost_late_source_failure",
        "binding/lost_late_source_result",
        "materialization/initial_late_cancelled",
        "materialization/initial_late_lost",
    }
)


def _selector_data():
    graph = DataGraph.new()
    graph, target = graph.add_text("selector")
    return graph.validate(
        targets=(target,),
        source_relations=(),
        contexts=(),
        dependencies=(),
        coherence=(),
        atomic=(),
        output_regions=(),
        limits=DataLimits(
            max_datums=1,
            max_targets=1,
            max_text_bytes=8,
            max_declarations=0,
            max_group_members=0,
        ),
    )


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


@dataclass
class _DecisionProvider:
    mode: str = "wait"
    started: asyncio.Event = field(default_factory=asyncio.Event)
    calls: int = 0

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalDecisionWait:
        self.calls += 1
        self.started.set()
        if self.mode == "cancel":
            await asyncio.Event().wait()
        if self.mode == "failure":
            raise RuntimeError("decision implementation failed")
        artifact = request[0].inputs[0].artifact
        assert artifact is not None
        assert isinstance(request[0].association, SemanticAssociation)
        return LocalDecisionWait(association=request[0].association, artifact=artifact)


@dataclass
class _DecisionClock:
    now: int = 0

    def now_ns(self) -> int:
        return self.now


async def _assert_decision_submission(case_id: str, expected: str, *, pending_limit: int | None = None) -> None:
    workflow, node, artifact_type = _workflow(with_input=True)
    data = _data(2 if pending_limit is not None else 1)
    capability = _capability(workflow)
    prepared = _prepare(
        data=data,
        workflow=workflow,
        capability=capability,
        bound_inputs=tuple(
            BoundInput(target=target, source=target, port="input", artifact_type=artifact_type)
            for target in data.targets
        ),
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
                ("deadline_exhausted", None, "failure"),
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
                outcomes=(
                    DecisionOutcome(decision="approve", outcome="ok"),
                    DecisionOutcome(decision="reject", outcome="ok"),
                ),
                max_lifetime_ns=10,
            ),
        ),
        assessment_productions=(),
        assessment_limits=_assessment_limits(),
    )
    callback = _DecisionProvider(
        mode={
            "decisions/cancel_condition": "cancel",
            "decisions/implementation_failure": "failure",
        }.get(case_id, "wait")
    )
    clock = _DecisionClock()
    running = await start_execution(
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
                max_local_in_flight=2,
                max_remote_outstanding=0,
                max_runtime_artifacts=2,
                max_runtime_artifact_bytes=100,
                max_collection_items=1,
            ),
            decision_limits=DecisionLimits(
                max_pending=pending_limit if pending_limit is not None else 1, max_lifetime_ns=10
            ),
            clock=clock,
        ),
    )
    if pending_limit is not None:
        await callback.started.wait()
        for _ in range(20):
            await asyncio.sleep(0)
        assert len(running.pending_decisions()) == pending_limit
        assert callback.calls == pending_limit
        first = running.pending_decisions()
        for wait in first:
            running.submit_decision(
                DecisionResponse(wait=wait.wait, workflow=wait.workflow, artifact=wait.artifact, decision="approve")
            )
        if pending_limit == 1:
            while not running.pending_decisions() or running.pending_decisions()[0].wait == first[0].wait:
                await asyncio.sleep(0)
            assert callback.calls == 2
            wait = running.pending_decisions()[0]
            assert wait.wait != first[0].wait and wait.artifact != first[0].artifact
            running.submit_decision(
                DecisionResponse(wait=wait.wait, workflow=wait.workflow, artifact=wait.artifact, decision="approve")
            )
        result = await running.wait()
        assert len(result.record.terminals) == 2
        assert all(item.category == "success" for item in result.record.terminals)
        assert not result.pending_decisions
        return
    if case_id in {"decisions/cancel_condition", "decisions/implementation_failure", "decisions/deadline"}:
        case = next(item for item in json.loads(CORPUS.read_bytes()) if item["case_id"] == case_id)
        if case_id == "decisions/cancel_condition":
            await callback.started.wait()
            running.request_cancel()
        elif case_id == "decisions/deadline":
            while not running.pending_decisions():
                await asyncio.sleep(0)
            wait = running.pending_decisions()[0]
            assert wait.allowed_decisions == frozenset(case["events"][1]["allowed"])
            clock.now = wait.deadline_ns
        result = await running.wait()
        assert callback.calls == 1
        assert [item.category for item in result.record.terminals] == list(case["expected"]["state"]["tasks"].values())
        assert not result.pending_decisions
        assert result.requests.dispatched_count == 0
        assert not result.final_outputs and not result.assessments
        assert len(result.artifacts) == len(result.ports) == len(result.provenance) == 1
        assert isinstance(result.provenance[0].key, RootInputKey)
        assert not result.provenance[0].parents and not result.provenance[0].decision
        assert result.provenance[0].artifact == result.artifacts[0][0]
        assert all(state.complete for state in result.states)
        return
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


@pytest.mark.parametrize(
    ("case_id", "items", "expected"),
    (
        (
            "map/duplicate_item",
            (
                TextCollectionItem(key=0, version=1, value=TextArtifactValue(text="a")),
                TextCollectionItem(key=0, version=1, value=TextArtifactValue(text="b")),
            ),
            "duplicate",
        ),
        (
            "map/noncanonical_items",
            (
                TextCollectionItem(key=1, version=1, value=TextArtifactValue(text="b")),
                TextCollectionItem(key=0, version=1, value=TextArtifactValue(text="a")),
            ),
            "invalid_value",
        ),
    ),
)
def test_map_collection_shape_rejects_at_typed_constructor(
    case_id: str,
    items: tuple[TextCollectionItem, ...],
    expected: str,
) -> None:
    case = next(item for item in json.loads(CORPUS.read_bytes()) if item["case_id"] == case_id)
    assert case["expected"] == {"status": "rejected", "code": expected}
    with pytest.raises(EffectRejected) as rejected:
        TextCollectionValue(items=items)
    assert rejected.value.code.value == case["expected"]["code"]
    canonical = (
        TextCollectionItem(key=0, version=1, value=TextArtifactValue(text="a")),
        TextCollectionItem(key=1, version=1, value=TextArtifactValue(text="b")),
    )
    assert TextCollectionValue(items=canonical).items == canonical


@pytest.mark.parametrize(
    ("case_id", "value"),
    (
        ("materialization/initial_wrong_item", 7),
        ("materialization/adaptive_wrong_item", 7),
        ("materialization/initial_nested_value", {"items": []}),
        ("materialization/adaptive_nested_value", {"items": []}),
    ),
)
def test_materialization_item_shape_rejects_at_typed_constructor(case_id: str, value: object) -> None:
    del case_id
    binding = BindingId.new()
    association = BindingAssociation(declaration=BindingDeclarationId.new(binding=binding, ordinal=0))
    with pytest.raises(EffectRejected) as rejected:
        SourceItem(association=association, key=0, version=1, text=cast(Any, value))
    assert rejected.value.code.value == "invalid_type"
    assert SourceItem(association=association, key=0, version=1, text="valid").text == "valid"


@pytest.mark.parametrize(
    ("case_id", "kind", "same_type", "max_items"),
    (
        ("materialization/single_output_mismatch", "single", False, 1),
        ("materialization/single_max_items", "single", True, 2),
        ("materialization/collection_scalar_output", "collection", True, 3),
        ("materialization/collection_zero_max", "collection", False, 0),
    ),
)
def test_initial_materialization_shape_rejects_at_typed_constructor(
    case_id: str,
    kind: str,
    same_type: bool,
    max_items: int,
) -> None:
    del case_id
    data = _data(1)
    target = next(iter(data.targets))
    workflow = WorkflowId.new()
    item_type = ArtifactType(name="text", revision=1)
    output_type = item_type if same_type else ArtifactType(name="text-collection", revision=1)
    common = {
        "target": target,
        "node": NodeId.new(workflow=workflow),
        "port": "context",
        "source": ContextSourceRef(name="source", revision=1),
        "selector": ContextSelector(fields=()),
        "requirement": "required",
    }
    with pytest.raises(EffectRejected) as rejected:
        InitialContextDecl(
            **cast(Any, common),
            artifact_type=output_type,
            bounds=RetrievalBounds(max_items=max_items, max_bytes=12, max_requests=1),
            materialization=ContextMaterialization(kind=cast(Any, kind), item_type=item_type),
        )
    assert rejected.value.code.value == "contradictory"
    valid_kind = cast(Any, kind)
    valid_output = item_type if kind == "single" else ArtifactType(name="valid-collection", revision=1)
    valid = InitialContextDecl(
        **cast(Any, common),
        artifact_type=valid_output,
        bounds=RetrievalBounds(max_items=1, max_bytes=12, max_requests=1),
        materialization=ContextMaterialization(kind=valid_kind, item_type=item_type),
    )
    assert valid.materialization.kind == kind


def test_map_collection_limit_rejects_boolean_at_typed_constructor() -> None:
    with pytest.raises(EffectRejected) as rejected:
        ExecutionLimits(
            max_local_in_flight=1,
            max_remote_outstanding=1,
            max_runtime_artifacts=8,
            max_runtime_artifact_bytes=32,
            max_collection_items=cast(Any, True),
        )
    assert rejected.value.code.value == "invalid_type"
    assert (
        ExecutionLimits(
            max_local_in_flight=1,
            max_remote_outstanding=1,
            max_runtime_artifacts=8,
            max_runtime_artifact_bytes=32,
            max_collection_items=4,
        ).max_collection_items
        == 4
    )


@pytest.mark.parametrize(
    "case_id",
    (
        "binding/source_failure_missing_failure",
        "binding/source_failure_missing_settlement",
        "binding/omission_failure_mismatch",
    ),
)
def test_binding_source_failure_constructor_cases(case_id: str) -> None:
    if case_id == "binding/source_failure_missing_failure":
        with pytest.raises(TypeError):
            cast(Any, SourceFailure)(source=ContextSourceRef(name="source", revision=1))
        return
    if case_id == "binding/source_failure_missing_settlement":
        with pytest.raises(TypeError):
            cast(Any, SourceFailure)(
                source=ContextSourceRef(name="source", revision=1),
                failure="permanent",
            )
        return
    with pytest.raises(EffectRejected) as rejected:
        SourceFailure(
            source=ContextSourceRef(name="source", revision=1),
            failure="retryable",
            settlement=None,
            disposition="omitted_optional",
        )
    assert rejected.value.code.value == "contradictory"


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
        fallback_associations: list[str],
        default_source: ContextSourceRef,
    ) -> None:
        self.events = events
        self.sources = sources
        self.association_names = association_names
        self.request_names = request_names
        self.fallback_associations = fallback_associations
        self.default_source = default_source

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
        label = self.association_names.get(association)
        if label is None:
            label = self.fallback_associations.pop(0)
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
        source = self.sources[cast(str, event["source"])] if "source" in event else self.default_source
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
                    association=(
                        association
                        if item.get("association", event.get("association")) == label
                        else BindingAssociation(
                            declaration=BindingDeclarationId.new(binding=BindingId.new(), ordinal=0)
                        )
                    ),
                    key=item["key"],
                    version=item["version"],
                    text=item.get("text", item.get("value")),
                )
                for item in cast(list[dict[str, Any]], event["items"])
            ),
            settlement=settlement,
        )

    async def cancel(self, request: PhysicalRequestId) -> StopConfirmed:
        del request
        return StopConfirmed(usage=ExactUsage(input_units=0, output_units=0))


@dataclass
class _BridgeLocalCallback:
    failure: str | None
    calls: int = 0
    block: bool = False
    started: asyncio.Event = field(default_factory=asyncio.Event)

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted | LocalFailure:
        self.calls += 1
        self.started.set()
        if self.block:
            await asyncio.Event().wait()
        if self.failure is not None:
            return LocalFailure(failure=cast(Any, self.failure))
        return LocalCompleted(
            results=tuple(
                AssociationResult(
                    association=item.association,
                    outcome="ok",
                    outputs=(),
                    consumed_context_ports=frozenset(),
                )
                for item in request
            )
        )


@pytest.mark.parametrize("case", LOCAL_BRIDGE_CASES, ids=lambda case: cast(str, case["case_id"]))
def test_local_bridge_case_through_real_execution(case: dict[str, Any]) -> None:
    asyncio.run(_assert_local_bridge_case(case))


async def _assert_local_bridge_case(case: dict[str, Any]) -> None:
    prepared = _prepare(data=_data(1))
    context = admit_context_plan(
        prepared=prepared,
        bound_context=None,
        adaptive_retrievals=(),
        context_capabilities=(),
    )
    selected = next(iter(prepared.implementations))
    implementation = ExecutionImplementation(
        implementation=selected.capability.implementation,
        configuration=selected.capability.configuration,
        capability=selected.capability,
        request=None,
    )
    policy = OperationExecutionPolicy(
        node=selected.node,
        kind="local",
        request=None,
        safe_detachment="forbidden",
        implementations=(implementation,),
        result_outcomes=frozenset({"ok"}),
        runtime_outcomes=_valid_runtime_rows("local", frozenset({"ok"})),
    )
    admitted = admit_execution_plan(
        context=context,
        capabilities=(selected.capability,),
        policies=(policy,),
        decisions=(),
        assessment_productions=(),
        assessment_limits=_assessment_limits(),
    )
    condition = (
        next(
            cast(str, event["failure"])
            for event in cast(list[dict[str, Any]], case["events"])
            if event["kind"] == "bridge_condition" and event.get("failure") is not None
        )
        if "/failure_" in cast(str, case["case_id"])
        else None
    )
    callback = _BridgeLocalCallback(
        failure=condition,
        block=case["case_id"] == "bridges/cancel_after_start",
    )
    running = await start_execution(
        admitted=admitted,
        capabilities=(selected.capability,),
        services=ExecutionServices(
            handles=(
                ImplementationHandle(
                    implementation=implementation.implementation,
                    operation=implementation.capability.operation,
                    configuration=implementation.configuration,
                    local=callback,
                    transport=None,
                    resource=None,
                ),
            ),
            context_resources=(),
            limits=ExecutionLimits(
                max_local_in_flight=1,
                max_remote_outstanding=0,
                max_runtime_artifacts=4,
                max_runtime_artifact_bytes=64,
                max_collection_items=1,
            ),
            decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
            clock=_ZeroClock(),
        ),
    )
    if case["case_id"] == "bridges/cancel_before_start":
        running.request_cancel()
    if callback.block:
        await callback.started.wait()
        running.request_cancel()
    result = await running.wait()
    expected = cast(dict[str, Any], case["expected"])["state"]
    if case["case_id"] == "bridges/cancel_before_start":
        assert callback.calls == 0
        assert len(result.record.terminals) == 1
        assert result.record.terminals[0].attempt is None
        assert result.record.terminals[0].category == expected["closed_unstarted"]["A0"]
        assert not result.requests.dispatches and not result.artifacts
        assert all(state.complete for state in result.states)
        return
    assert callback.calls == 1
    assert len(result.record.terminals) == 1
    terminal = result.record.terminals[0]
    assert terminal.category == expected["tasks"]["T0"]
    assert terminal.attempt is not None


@dataclass
class _LateAdaptiveProvider:
    source: ContextSourceRef
    association_names: dict[object, str]
    request_names: dict[PhysicalRequestId, str]
    acknowledge_stop: bool
    started: asyncio.Event = field(default_factory=asyncio.Event)
    returned_after_cancel: bool = False

    async def retrieve(
        self,
        *,
        request: PhysicalRequestId,
        association: SemanticAssociation,
        selector: ContextSelector,
        bounds: RetrievalBounds,
    ) -> SourceResponse:
        del selector, bounds
        self.association_names[association] = "A0"
        self.request_names[request] = "R0"
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.returned_after_cancel = True
        return SourceResponse(
            source=self.source,
            items=(SourceItem(association=association, key=0, version=1, text="late"),),
            settlement=ExternalSettlement(
                request=request,
                disposition="completed",
                usage=ExactUsage(input_units=0, output_units=0),
                remote_stopped=True,
            ),
        )

    async def cancel(self, request: PhysicalRequestId) -> StopConfirmed:
        del request
        if not self.acknowledge_stop:
            raise RuntimeError("transport did not acknowledge cancellation")
        return StopConfirmed(usage=UnknownUsage())


@dataclass
class _LateBindingCaseProvider:
    delegate: _CaseProvider
    acknowledge_stop: bool
    started: asyncio.Event = field(default_factory=asyncio.Event)

    async def retrieve(self, **kwargs: Any) -> SourceResponse | SourceFailure:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass
        return await self.delegate.retrieve(**kwargs)

    async def cancel(self, request: PhysicalRequestId) -> StopConfirmed:
        del request
        if not self.acknowledge_stop:
            raise RuntimeError("stop not acknowledged")
        return StopConfirmed(usage=UnknownUsage())


@dataclass
class _ContextConsumer:
    port: str

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        assert len(request) == 1
        return LocalCompleted(
            results=(
                AssociationResult(
                    association=request[0].association,
                    outcome="ok",
                    outputs=(),
                    consumed_context_ports=frozenset({self.port}),
                ),
            )
        )


@pytest.mark.parametrize("case", REAL_BINDING_CASES, ids=lambda case: cast(str, case["case_id"]))
def test_binding_corpus_case_through_real_provider(case: dict[str, Any]) -> None:
    asyncio.run(_assert_binding_corpus_case(case))


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
    providers = {
        name: _CaseProvider(
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
            resource_owner="caller",
            cancellation="cooperative_ack",
            settlement="explicit_ack",
            usage="exact",
            request=policies["P0"],
            safe_detachment="forbidden",
        )
        for name in source_refs
        if name in declared_source_names
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
    if materializations:
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
    assert sorted(actual_artifacts, key=lambda item: item["identity"]) == sorted(
        expected_artifacts, key=lambda item: item["identity"]
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
    if materializations and result.context is not None and expected.get("materialization") is not None:
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
    context_port = bound_context.receipt.sources[0].declaration.port
    execution = await (
        await start_execution(
            admitted=admitted,
            capabilities=(capability,),
            services=ExecutionServices(
                handles=(
                    ImplementationHandle(
                        implementation=capability.implementation,
                        operation=capability.operation,
                        configuration=capability.configuration,
                        local=_ContextConsumer(port=context_port),
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
    ).wait()
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


@pytest.mark.parametrize("case", LATE_INITIAL_CASES, ids=lambda case: cast(str, case["case_id"]))
def test_late_initial_provider_facts_through_running_binding(case: dict[str, Any]) -> None:
    asyncio.run(_assert_late_initial_provider_case(case))


async def _assert_late_initial_provider_case(case: dict[str, Any]) -> None:
    raw = cast(dict[str, Any], case["declaration"])
    workflow, node, artifact_type = _context_workflow()
    materializations = cast(list[dict[str, Any]], raw.get("materializations", []))
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
        resource_owner="caller",
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
        bounds=RetrievalBounds(max_items=1, max_bytes=12, max_requests=policy.max_attempts),
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


@pytest.mark.parametrize(
    "case",
    ADAPTIVE_MATERIALIZATION_CASES,
    ids=lambda case: cast(str, case["case_id"]),
)
def test_adaptive_materialization_corpus_through_real_execution(case: dict[str, Any]) -> None:
    asyncio.run(_assert_adaptive_materialization_case(case))


@pytest.mark.parametrize("case", LATE_ADAPTIVE_CASES, ids=lambda case: cast(str, case["case_id"]))
def test_late_adaptive_result_stays_request_terminal_at_authority_boundary(case: dict[str, Any]) -> None:
    raw = cast(dict[str, Any], case["declaration"])
    invocation = InvocationId.new(plan=PlanId.new())
    scope = InvocationRequestScope(invocation=invocation)
    association = SemanticAssociation(
        task=TaskAttemptId.new(
            activation=ActivationKey(invocation=invocation, occurrence=0, parent=None, iteration=None)
        )
    )
    policy = _policy(cast(dict[str, Any], raw["policies"])["P0"])
    state = initialize_requests(scope=scope, hard_limit=cast(int, raw["hard_limit"]), policies=frozenset({policy}))
    state = bind_request_policies(
        state=state,
        binding=RequestPolicyBinding.create(association=association, policies=frozenset({policy})),
    )
    request = PhysicalRequestId.new(scope=scope)
    state = advance_requests(
        state=state,
        event=Reserve(
            request=request,
            purpose="adaptive_retrieval",
            associations=frozenset({association}),
            policy=policy,
        ),
    )
    state = advance_requests(state=state, event=Dispatch(request=request))
    if case["case_id"] == "materialization/adaptive_late_lost":
        state = advance_requests(state=state, event=RequestCancel(request=request))
        state = advance_requests(state=state, event=MarkLost(request=request))
    else:
        state = advance_requests(state=state, event=RequestCancel(request=request))
        state = advance_requests(
            state=state,
            event=StopAcknowledged(request=request, usage=UnknownUsage()),
        )
    state = advance_requests(
        state=state,
        event=AcceptResult(
            request=request,
            results=(
                AssociationResult(
                    association=association,
                    outcome="ok",
                    outputs=(),
                    consumed_context_ports=frozenset(),
                ),
            ),
        ),
    )
    settlement_value = next(
        cast(dict[str, Any], event["settlement"])
        for event in cast(list[dict[str, Any]], case["events"])
        if event["kind"] == "materialize_result"
    )
    state = advance_requests(
        state=state,
        event=ObserveSettlement(
            settlement=ExternalSettlement(
                request=request,
                disposition=settlement_value["disposition"],
                usage=_usage(settlement_value["usage"]),
                remote_stopped=settlement_value["remote_stopped"],
            )
        ),
    )
    actual = _normalize(state, {"A0": association}, {"R0": request}, {"P0": policy})
    expected = cast(dict[str, Any], case["expected"])["state"]
    for key in (
        "bindings",
        "dispatched",
        "dispatched_count",
        "terminals",
        "settlements",
        "request_facts",
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
        "attempts",
    ):
        assert actual[key] == expected[key], (case["case_id"], key)
    assert not expected["materialization"]["ports"]
    assert all(not item["identity"].startswith("OperationOutputKey:") for item in expected["artifacts"])


@pytest.mark.parametrize("case", LATE_ADAPTIVE_CASES, ids=lambda case: cast(str, case["case_id"]))
def test_late_adaptive_provider_result_is_suppressed_by_real_executor(case: dict[str, Any]) -> None:
    asyncio.run(
        _assert_adaptive_materialization_case(
            case,
            late_mode="lost" if case["case_id"] == "materialization/adaptive_late_lost" else "cancelled",
        )
    )


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
    binding = BindingId.new()
    binding_labels = set(cast(dict[str, str], declaration.get("binding_declarations", {})))
    scope = BindingRequestScope(binding=binding) if binding_labels else InvocationRequestScope(invocation=invocation)
    policies = {
        name: _policy(value) for name, value in cast(dict[str, dict[str, Any]], declaration["policies"]).items()
    }
    association_names = (
        {
            item
            for event in cast(list[dict[str, Any]], case["events"])
            for field in ("associations", "returned")
            for item in cast(list[str], event.get(field, ()))
        }
        | {
            cast(str, event["association"])
            for event in cast(list[dict[str, Any]], case["events"])
            if "association" in event
        }
        | {"T0", "T1"}
    )
    foreign_invocation = InvocationId.new(plan=invocation.plan)
    tasks = {
        name: (
            BindingAssociation(declaration=BindingDeclarationId.new(binding=binding, ordinal=index))
            if name in binding_labels
            else SemanticAssociation(
                task=TaskAttemptId.new(
                    activation=ActivationKey(
                        invocation=foreign_invocation if name.startswith("X") else invocation,
                        occurrence=index,
                        parent=None,
                        iteration=None,
                    )
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
    tasks: dict[str, Any],
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
    elif kind == "source_failure":
        state = advance_requests(
            state=state,
            event=AcceptFailure(request=requests[event["request"]], failure=event["failure"]),
        )
        settlement = cast(dict[str, Any], event["settlement"])
        return advance_requests(
            state=state,
            event=ObserveSettlement(
                settlement=ExternalSettlement(
                    request=requests[event["request"]],
                    disposition=settlement["disposition"],
                    usage=_usage(settlement["usage"]),
                    remote_stopped=settlement["remote_stopped"],
                )
            ),
        )
    elif kind == "source_result":
        returned = cast(list[dict[str, Any]], event["items"])
        state = advance_requests(
            state=state,
            event=AcceptResult(
                request=requests[event["request"]],
                results=(
                    AssociationResult(
                        association=tasks[returned[0]["association"]],
                        outcome=event["outcome"],
                        outputs=(),
                        consumed_context_ports=frozenset(),
                    ),
                ),
            ),
        )
        settlement = cast(dict[str, Any], event["settlement"])
        return advance_requests(
            state=state,
            event=ObserveSettlement(
                settlement=ExternalSettlement(
                    request=requests[event["request"]],
                    disposition=settlement["disposition"],
                    usage=_usage(settlement["usage"]),
                    remote_stopped=settlement["remote_stopped"],
                )
            ),
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
    tasks: dict[str, Any],
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


@pytest.mark.parametrize(
    "case_id", ["decisions/cancel_condition", "decisions/implementation_failure", "decisions/deadline"]
)
def test_decision_runtime_terminal_cases(case_id: str) -> None:
    asyncio.run(_assert_decision_submission(case_id, ""))


@pytest.mark.parametrize("pending_limit", [1, 2])
def test_decision_capacity_includes_callbacks_before_wait_publication(pending_limit: int) -> None:
    asyncio.run(_assert_decision_submission("capacity", "", pending_limit=pending_limit))
