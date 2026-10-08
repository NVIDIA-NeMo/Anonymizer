# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Effects corpus selection and admission and decision execution fixtures."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import Any, cast

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectRejected
from anonymizer.engine.graph_sdk.capabilities import ImplementationRef
from anonymizer.engine.graph_sdk.context import (
    admit_context_plan,
)
from anonymizer.engine.graph_sdk.data import DataGraph, DataLimits
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
    RootInputKey,
    RuntimeOutcome,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import BoundInput
from anonymizer.engine.graph_sdk.requests import (
    AssociationInput,
    PhysicalRequestPolicy,
    SemanticAssociation,
)
from anonymizer.graph._values import ArtifactRef, InvocationId
from anonymizer.graph.workflow import (
    NodeId,
    WorkflowId,
)
from tests.graph_sdk.reference.corpora import load_cases
from tests.graph_sdk.test_preparation import _capability, _data, _limits, _prepare, _workflow

REQUEST_FAMILIES = {"budgets", "keyed", "retry", "races", "inflight"}

_BASE_CASES = tuple(
    case
    for case in load_cases("effects")
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
    for case in load_cases("effects")
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
    for case in load_cases("effects")
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
    for case in load_cases("effects")
    if case["case_id"]
    in {
        "binding/source_result_wrong_outcome",
        "binding/source_result_outputs_present",
        "binding/source_result_consumed_present",
    }
)

ADMISSION_CASES = tuple(
    case
    for case in load_cases("effects")
    if case["family"] == "admission" or case["case_id"] == "retry/implementation_owner_rejected"
)

REAL_BINDING_CASE_IDS = {
    "materialization/latest_optional_omission",
    "materialization/latest_provenance_edges_exact",
    "materialization/latest_provenance_edges_one_over",
    "materialization/latest_one_version",
    "materialization/latest_two_versions",
    "materialization/latest_reordered_versions",
    "materialization/latest_version_gap",
    "materialization/latest_items_exact",
    "materialization/latest_bytes_exact",
    "materialization/latest_retry",
    "materialization/latest_correction",
    "materialization/latest_caller_cleanup",
    "materialization/latest_sdk_cleanup_close_failed",
    "materialization/latest_sdk_cleanup_close_unknown",
    "materialization/latest_duplicate_pair",
    "materialization/latest_multiple_keys",
    "materialization/latest_items_one_over",
    "materialization/latest_bytes_one_over",
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
    "materialization/collection_ceiling",
    "materialization/initial_artifact_count_one_over",
    "materialization/initial_logical_bytes_one_over",
    "materialization/initial_provenance_one_over",
    "materialization/binding_finish_before_materialization",
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

REAL_BINDING_CASES = tuple(case for case in load_cases("effects") if case["case_id"] in REAL_BINDING_CASE_IDS)

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
    case for case in load_cases("effects") if case["case_id"] in ADAPTIVE_MATERIALIZATION_CASE_IDS
)

LATE_ADAPTIVE_CASES = tuple(
    case
    for case in load_cases("effects")
    if case["case_id"] in {"materialization/adaptive_late_lost", "materialization/adaptive_late_cancelled"}
)

LATE_INITIAL_CASES = tuple(
    case
    for case in load_cases("effects")
    if case["case_id"]
    in {
        "binding/cancelled_late_source_failure",
        "binding/cancelled_late_source_result",
        "binding/lost_late_source_failure",
        "binding/lost_late_source_result",
        "materialization/initial_late_cancelled",
        "materialization/initial_late_lost",
        "materialization/latest_cancelled_late_valid",
        "materialization/latest_cancelled_late_multikey",
        "materialization/latest_lost_late_valid",
        "materialization/latest_lost_late_multikey",
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
        case = next(item for item in load_cases("effects") if item["case_id"] == case_id)
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
