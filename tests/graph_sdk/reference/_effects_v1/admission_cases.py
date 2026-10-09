# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent effects_v1 reference: admission cases."""

from __future__ import annotations

from tests.graph_sdk.reference._effects_v1.builders import (
    _policy_runtime_decl,
    _runtime_decl,
)
from tests.graph_sdk.reference._effects_v1.case import (
    _case,
)
from tests.graph_sdk.reference._effects_v1.model import (
    Object,
    _array,
    _object,
)


def admission_cases() -> list[Object]:
    c: list[Object] = []
    opened: list[Object] = [
        {"kind": "bridge_start", "node": "N0", "task": "T0"},
        {
            "allowed": ["approve", "reject"],
            "artifact": "V0",
            "kind": "decision_open",
            "task": "T0",
            "wait": "W0",
            "workflow": "F0",
        },
    ]

    open1: Object = {
        "allowed": ["approve"],
        "artifact": "V1",
        "kind": "decision_open",
        "task": "T1",
        "wait": "W1",
        "workflow": "F0",
    }

    c += [
        _case(
            "decisions",
            "pending_exact",
            {"max_pending": 2},
            [*opened, {"kind": "bridge_start", "node": "N1", "task": "T1"}, open1],
        ),
        _case(
            "decisions",
            "pending_one_over",
            {"max_pending": 1},
            [*opened, {"kind": "bridge_start", "node": "N1", "task": "T1"}, open1],
        ),
    ]

    valid_policy: Object = {
        "implementations": [{"physical_policy": "P0"}],
        "kind": "external",
        "node": "N0",
        "physical_policy": "P0",
        "result_outcomes": ["ok"],
        "retry_owner": "executor",
    }

    result_mapping = _object(
        _array(_runtime_decl("result", "ok", "success", reported_outcome="ok")["runtime_mappings"])[0]
    )

    missing_runtime = _policy_runtime_decl("external")

    missing_runtime["execution_policies"] = [valid_policy]

    missing_runtime["runtime_mappings"] = [
        raw for raw in _array(missing_runtime["runtime_mappings"]) if _object(raw)["condition"] != "result"
    ]

    duplicate_condition = _policy_runtime_decl("external")

    duplicate_condition["execution_policies"] = [valid_policy]

    blocked_row = next(
        _object(raw)
        for raw in _array(duplicate_condition["runtime_mappings"])
        if _object(raw)["condition"] == "cancel_before_start"
    )

    _array(duplicate_condition["runtime_mappings"]).append(dict(blocked_row, category="cancelled"))

    primary_capability: Object = {"implementation": "C0", "physical_policy": "P0"}

    alternate_capability: Object = {"implementation": "C1", "physical_policy": "P0"}

    admission_negatives: tuple[tuple[str, Object], ...] = (
        ("outer_type", {"execution_policies": "invalid"}),
        (
            "aggregate_limit",
            {
                "admission_limits": {"max_capabilities": 1},
                "capability_catalog": [primary_capability, alternate_capability],
            },
        ),
        ("member_type", {"execution_policies": [0]}),
        ("invalid_value", {"execution_policies": [dict(valid_policy, kind="unknown")]}),
        ("foreign_owner", {"execution_policies": [dict(valid_policy, owner="I1")]}),
        ("duplicate", {"execution_policies": [valid_policy, valid_policy]}),
        ("missing", {"execution_policies": [dict(valid_policy, implementations=[])]}),
        ("unsupported", {"execution_policies": [dict(valid_policy, retry_owner="implementation")]}),
        (
            "contradictory",
            {
                "declared_outcomes": ["ok"],
                "outcome_categories": {"ok": "success"},
                "runtime_mappings": [dict(result_mapping, category="failure")],
            },
        ),
        (
            "runtime_mapping_missing",
            missing_runtime,
        ),
        (
            "wrong_category",
            {
                "declared_outcomes": ["ok"],
                "outcome_categories": {"ok": "success"},
                "runtime_mappings": [dict(result_mapping, category="blocked")],
            },
        ),
        (
            "unknown_outcome",
            {
                "declared_outcomes": ["ok"],
                "runtime_mappings": [dict(result_mapping, outcome="other", reported_outcome="other")],
            },
        ),
        (
            "duplicate_runtime_mapping",
            {
                "declared_outcomes": ["ok"],
                "outcome_categories": {"ok": "success"},
                "runtime_mappings": [result_mapping, result_mapping],
            },
        ),
        (
            "duplicate_required_condition",
            duplicate_condition,
        ),
    )

    for name, declaration in admission_negatives:
        c.append(_case("admission", name, declaration, [], "admission"))

    c += [
        _case(
            "admission",
            "valid_external_failover",
            {
                **_policy_runtime_decl("external"),
                "execution_policies": [
                    {
                        "implementations": [{"physical_policy": "P0"}, {"physical_policy": "P0"}],
                        "kind": "external",
                        "node": "N0",
                        "physical_policy": "P0",
                        "result_outcomes": ["ok"],
                        "retry_owner": "executor",
                    }
                ],
            },
            [],
            "admission",
        ),
        _case(
            "admission",
            "changed_failover_policy",
            {
                "admitted": {
                    **_policy_runtime_decl("external"),
                    "execution_policies": [
                        dict(valid_policy, implementations=[{"physical_policy": "P0"}, {"physical_policy": "P0"}])
                    ],
                    "capability_catalog": [primary_capability, alternate_capability],
                },
                "capability_catalog": [primary_capability, dict(alternate_capability, physical_policy="P1")],
            },
            [],
            "pre_execution",
        ),
        _case(
            "admission",
            "local_multiple_implementations",
            {
                "execution_policies": [
                    {
                        "implementations": [{"physical_policy": None}, {"physical_policy": None}],
                        "kind": "local",
                        "node": "N0",
                        "physical_policy": None,
                        "result_outcomes": ["ok"],
                        "retry_owner": "none",
                    }
                ]
            },
            [],
            "admission",
        ),
        _case(
            "admission",
            "capability_one_over",
            {
                "admission_limits": {"max_capabilities": 1},
                "capability_catalog": [primary_capability, primary_capability],
            },
            [],
            "admission",
        ),
        _case(
            "admission",
            "valid_local_runtime_product",
            {
                **_policy_runtime_decl("local"),
                "execution_policies": [
                    {
                        "implementations": [{"physical_policy": None}],
                        "kind": "local",
                        "node": "N0",
                        "physical_policy": None,
                        "result_outcomes": ["ok"],
                        "retry_owner": "none",
                    }
                ],
            },
            [],
            "admission",
        ),
        _case(
            "admission",
            "valid_decision_runtime_product",
            {
                **_policy_runtime_decl("decision"),
                "execution_policies": [
                    {
                        "implementations": [{"physical_policy": None}],
                        "kind": "decision",
                        "node": "N0",
                        "physical_policy": None,
                        "result_outcomes": [],
                        "retry_owner": "none",
                    }
                ],
            },
            [],
            "admission",
        ),
    ]

    return c
