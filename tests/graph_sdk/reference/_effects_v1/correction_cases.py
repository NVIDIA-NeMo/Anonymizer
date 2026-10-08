# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent effects_v1 reference: correction cases."""

from __future__ import annotations

from tests.graph_sdk.reference._effects_v1.builders import (
    _binding_decl,
    _decl,
    _dispatch,
    _policy,
    _policy_runtime_decl,
    _reserve,
    _result,
    _runtime_decl,
    _source_failure,
    _source_result,
    _trace,
)
from tests.graph_sdk.reference._effects_v1.case import (
    _case,
)
from tests.graph_sdk.reference._effects_v1.model import (
    Object,
    _array,
    _object,
)


def correction_cases() -> list[Object]:
    c: list[Object] = []
    result_mapping = _object(
        _array(_runtime_decl("result", "ok", "success", reported_outcome="ok")["runtime_mappings"])[0]
    )

    settlement: Object = {
        "disposition": "completed",
        "kind": "settlement",
        "remote_stopped": True,
        "request": "R0",
        "usage": {"input": 1, "output": 1},
    }

    external_policy: Object = {
        "implementations": [{"physical_policy": "P0"}],
        "kind": "external",
        "node": "N0",
        "physical_policy": "P0",
        "result_outcomes": ["ok"],
        "retry_owner": "executor",
    }

    missing_product = _policy_runtime_decl("external")

    missing_product["execution_policies"] = [external_policy]

    missing_product["runtime_mappings"] = [
        value for value in _array(missing_product["runtime_mappings"]) if _object(value).get("condition") != "result"
    ]

    extra_product = _policy_runtime_decl("local")

    extra_product["execution_policies"] = [
        {
            "implementations": [{"physical_policy": None}],
            "kind": "local",
            "node": "N0",
            "physical_policy": None,
            "result_outcomes": ["ok"],
            "retry_owner": "none",
        }
    ]

    extra_product["runtime_mappings"] = [
        *_array(extra_product["runtime_mappings"]),
        _object(_array(_runtime_decl("lost", None, "lost")["runtime_mappings"])[0]),
    ]

    duplicate_outcome = _policy_runtime_decl("external")

    duplicate_outcome["execution_policies"] = [external_policy]

    _array(duplicate_outcome["runtime_mappings"]).append(dict(result_mapping, category="failure"))

    c += [
        _case("admission", "runtime_product_missing_result", missing_product, [], "admission"),
        _case("admission", "runtime_product_extra_local_condition", extra_product, [], "admission"),
        _case("admission", "duplicate_result_outcome", duplicate_outcome, [], "admission"),
    ]

    for replay in ("never", "before_acceptance", "idempotent"):
        failover_decl = _decl()
        _object(failover_decl["policies"])["P0"] = _policy(replay=replay)
        c.append(
            _case(
                "retry",
                f"failover_permanent_replay_{replay}",
                failover_decl,
                [
                    *_trace(),
                    {"kind": "failure", "request": "R0", "failure": "permanent"},
                    _reserve("R1", ["T0"], purpose="failover"),
                ],
            )
        )

    c.append(
        _case(
            "races",
            "lost_late_failure_without_settlement",
            _decl(),
            [
                *_trace(),
                {"kind": "lost", "request": "R0"},
                {"kind": "failure", "request": "R0", "failure": "retryable"},
            ],
        )
    )

    c.append(
        _case(
            "races",
            "lost_conflicting_settlement_preserves_remote",
            _decl(),
            [
                *_trace(),
                {"kind": "lost", "request": "R0"},
                dict(settlement, disposition="unknown", remote_stopped=None, usage="unknown"),
                settlement,
            ],
        )
    )

    late_binding_responses: tuple[Object, ...] = (
        _source_result(
            "D0",
            "S0",
            [{"association": "D0", "key": 0, "version": 1, "text": "late"}],
            request="R0",
        ),
        _source_failure("D0", "S0", request="R0"),
    )

    for terminal in ("lost", "cancelled"):
        closure: list[Object] = (
            [{"kind": "lost", "request": "R0"}]
            if terminal == "lost"
            else [
                {"kind": "cancel", "request": "R0"},
                {"kind": "stop", "request": "R0", "usage": "unknown"},
            ]
        )
        if terminal == "lost":
            closure.insert(0, {"kind": "cancel", "request": "R0"})
        for response in late_binding_responses:
            late_events = [*_trace(("D0",)), *closure, response]
            c.append(
                _case(
                    "binding",
                    f"{terminal}_late_{response['kind']}",
                    _binding_decl({"D0": "S0"}),
                    late_events,
                    traces=[[*late_events, settlement]],
                )
            )

    retry_prefix: list[Object] = [
        *_trace(),
        {"kind": "failure", "request": "R0", "failure": "retryable"},
        _reserve("R1", ["T0"], purpose="retry"),
        _dispatch("R1"),
    ]

    latest_cases: tuple[tuple[str, str, list[Object]], ...] = (
        ("pending", "retry", []),
        ("success", "retry", [_result("R1", ("T0",), ("T0",))]),
        ("permanent", "retry", [{"kind": "failure", "request": "R1", "failure": "permanent"}]),
        ("malformed", "correction", [{"kind": "failure", "request": "R1", "failure": "malformed_response"}]),
    )

    for latest, continuation, terminal_events in latest_cases:
        c.append(
            _case(
                "retry",
                f"latest_request_{latest}",
                _decl(limit=3, attempts=3),
                [*retry_prefix, *terminal_events, _reserve("R2", ["T0"], purpose=continuation)],
            )
        )

    return c
