# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Effects corpus selection and admission fixtures."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, cast

from anonymizer.engine.graph_sdk.capabilities import ImplementationCapability, ImplementationRef
from anonymizer.engine.graph_sdk.context import (
    admit_context_plan,
)
from anonymizer.engine.graph_sdk.executor import (
    AdmittedExecutionPlan,
    AssessmentLimits,
    ExecutionImplementation,
    ExecutionLimits,
    ExecutionServices,
    OperationExecutionPolicy,
    RuntimeOutcome,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import PreparedPlan
from anonymizer.engine.graph_sdk.requests import (
    PhysicalRequestPolicy,
)
from anonymizer.graph import NodeId, WorkflowId
from tests.graph_sdk.implementation_fixtures import _execution_implementation
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

# Decision waits are outside the SDK; their single admission case is excluded.
ADMISSION_CASES = tuple(
    case
    for case in load_cases("effects")
    if (case["family"] == "admission" or case["case_id"] == "retry/implementation_owner_rejected")
    and case["case_id"] != "admission/valid_decision_runtime_product"
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


def _assessment_limits() -> AssessmentLimits:
    return AssessmentLimits(
        max_productions=0,
        max_findings_per_production=0,
        max_finding_code_bytes=0,
        max_assessment_facts=0,
        max_port_facts=2,
        max_provenance_edges=2,
    )


def _admit_unbound_execution(
    *,
    prepared: PreparedPlan,
    capabilities: tuple[ImplementationCapability, ...],
    policies: tuple[OperationExecutionPolicy, ...],
) -> AdmittedExecutionPlan:
    return admit_execution_plan(
        context=admit_context_plan(
            prepared=prepared,
            bound_context=None,
        ),
        capabilities=capabilities,
        policies=policies,
        assessment_productions=(),
        assessment_limits=_assessment_limits(),
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
        "rejected_before_acceptance",
        "retryable",
        "malformed_response",
        "permanent",
        "transport_unknown",
        "implementation_exception",
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
    workflow, node, _ = _workflow(requests=3 if external else 0)
    primary = _capability(workflow, external=external)
    if external:
        primary = replace(primary, max_physical_requests_per_activation=2)
    data = _data(2)
    capability_maximum = cast(dict[str, int], declaration.get("admission_limits", {})).get("max_capabilities", 2)
    prepared = _prepare(
        data=data,
        workflow=workflow,
        capability=primary,
        limits=_limits(capabilities=capability_maximum),
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
                implementations.append(_execution_implementation(capability, request=request))
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
    admitted = _admit_unbound_execution(
        prepared=prepared,
        capabilities=tuple(capabilities),
        policies=cast(Any, policies),
    )
    if case["boundary"] == "pre_execution":
        changed = replace(capabilities[-1], max_physical_requests_per_activation=3)
        asyncio.run(
            start_execution(
                admitted=admitted,
                capabilities=tuple([*capabilities[:-1], changed]),
                services=ExecutionServices(
                    handles=(),
                    limits=ExecutionLimits(
                        max_local_in_flight=1,
                        max_remote_outstanding=1,
                        max_runtime_artifacts=1,
                        max_runtime_artifact_bytes=1,
                        max_collection_items=1,
                    ),
                    clock=_ZeroClock(),
                ),
            )
        )


class _ZeroClock:
    def now_ns(self) -> int:
        return 0
