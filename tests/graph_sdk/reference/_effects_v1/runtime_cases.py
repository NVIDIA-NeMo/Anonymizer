# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent effects_v1 reference: runtime cases."""

from __future__ import annotations

from tests.graph_sdk.reference._effects_v1.builders import (
    _decl,
    _dispatch,
    _reserve,
    _result,
    _runtime_decl,
    _trace,
)
from tests.graph_sdk.reference._effects_v1.case import (
    _case,
)
from tests.graph_sdk.reference._effects_v1.model import (
    FAILURE_CLASSES,
    Object,
)


def runtime_cases() -> list[Object]:
    c: list[Object] = []
    for condition, outcome, category in (
        ("result", "ok", "success"),
        ("cancel_after_start", None, "cancelled"),
        ("cancel_after_dispatch", None, "cancelled"),
        ("lost", None, "lost"),
        ("request_inconsistent", None, "inconsistent"),
        ("budget_exhausted", None, "blocked"),
        ("request_limit_exhausted", None, "blocked"),
        ("artifact_limit_exhausted", None, "blocked"),
        ("deadline_exhausted", None, "failure"),
    ):
        c.append(
            _case(
                "bridges",
                condition,
                _runtime_decl(condition, outcome, category, reported_outcome="ok" if condition == "result" else None),
                [
                    {"kind": "bridge_start", "node": "N0", "task": "T0"},
                    {
                        "condition": condition,
                        "kind": "bridge_condition",
                        **({"reported_outcome": "ok"} if condition == "result" else {}),
                        "task": "T0",
                    },
                    {"category": category, "kind": "bridge_emit", "outcome": outcome, "task": "T0"},
                ],
            )
        )

    for failure in FAILURE_CLASSES:
        c.append(
            _case(
                "bridges",
                f"failure_{failure}",
                _runtime_decl("failure", None, "failure", failure=failure),
                [
                    {"kind": "bridge_start", "node": "N0", "task": "T0"},
                    {
                        "condition": "failure",
                        "failure": failure,
                        "kind": "bridge_condition",
                        "task": "T0",
                    },
                    {"category": "failure", "kind": "bridge_emit", "outcome": None, "task": "T0"},
                ],
            )
        )

    shared_bridge = _decl()

    shared_bridge.update(_runtime_decl("result", "ok", "success", reported_outcome="ok"))

    c.append(
        _case(
            "bridges",
            "shared_request_two_targets",
            shared_bridge,
            _trace(("T0", "T1"))
            + [
                {"kind": "bridge_start", "node": "N0", "task": "T0"},
                {"kind": "bridge_start", "node": "N1", "task": "T1"},
                _result("R0", ("T0", "T1"), ("T1", "T0")),
                {"condition": "result", "kind": "bridge_condition", "reported_outcome": "ok", "task": "T0"},
                {"condition": "result", "kind": "bridge_condition", "reported_outcome": "ok", "task": "T1"},
                {"category": "success", "kind": "bridge_emit", "outcome": "ok", "task": "T0"},
                {"category": "success", "kind": "bridge_emit", "outcome": "ok", "task": "T1"},
            ],
        )
    )

    c.append(
        _case(
            "bridges",
            "shared_request_start_before_dispatch",
            shared_bridge,
            [
                {"kind": "bridge_start", "node": "N0", "task": "T0"},
                {"kind": "bridge_start", "node": "N1", "task": "T1"},
                *_trace(("T0", "T1")),
                _result("R0", ("T0", "T1"), ("T0", "T1")),
                {"condition": "result", "kind": "bridge_condition", "reported_outcome": "ok", "task": "T0"},
                {"condition": "result", "kind": "bridge_condition", "reported_outcome": "ok", "task": "T1"},
                {"category": "success", "kind": "bridge_emit", "outcome": "ok", "task": "T0"},
                {"category": "success", "kind": "bridge_emit", "outcome": "ok", "task": "T1"},
            ],
        )
    )

    retry_bridge = _decl()

    retry_bridge.update(_runtime_decl("result", "ok", "success", reported_outcome="ok"))

    c.append(
        _case(
            "bridges",
            "retry_uses_latest_physical_request",
            retry_bridge,
            [
                {"kind": "bridge_start", "node": "N0", "task": "T0"},
                *_trace(),
                {"failure": "retryable", "kind": "failure", "request": "R0"},
                _reserve("R1", ["T0"], purpose="retry"),
                _dispatch("R1"),
                _result("R1", ("T0",), ("T0",)),
                {"condition": "result", "kind": "bridge_condition", "reported_outcome": "ok", "task": "T0"},
                {"category": "success", "kind": "bridge_emit", "outcome": "ok", "task": "T0"},
            ],
        )
    )

    for name, returned in (
        ("missing", ("T0",)),
        ("duplicate", ("T0", "T0", "T1")),
        ("extra", ("T0", "T1", "T2")),
        ("foreign", ("T0", "T1", "X0")),
    ):
        inconsistent = _decl()
        inconsistent.update(_runtime_decl("request_inconsistent", None, "inconsistent"))
        c.append(
            _case(
                "bridges",
                f"shared_request_{name}",
                inconsistent,
                _trace(("T0", "T1"))
                + [
                    {"kind": "bridge_start", "node": "N0", "task": "T0"},
                    {"kind": "bridge_start", "node": "N1", "task": "T1"},
                    _result("R0", ("T0", "T1"), returned),
                    {"condition": "request_inconsistent", "kind": "bridge_condition", "task": "T0"},
                    {"condition": "request_inconsistent", "kind": "bridge_condition", "task": "T1"},
                    {"category": "inconsistent", "kind": "bridge_emit", "outcome": None, "task": "T0"},
                    {"category": "inconsistent", "kind": "bridge_emit", "outcome": None, "task": "T1"},
                ],
            )
        )

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

    c.append(
        _case(
            "decisions",
            "matching_resume_unrelated_advances",
            _runtime_decl("result", "ok", "success", reported_outcome="ok"),
            [
                *opened,
                {"kind": "bridge_start", "node": "N1", "task": "T1"},
                {
                    "condition": "result",
                    "kind": "bridge_condition",
                    "reported_outcome": "ok",
                    "task": "T1",
                },
                {"category": "success", "kind": "bridge_emit", "outcome": "ok", "task": "T1"},
                {"artifact": "V0", "decision": "approve", "kind": "decision_submit", "wait": "W0", "workflow": "F0"},
            ],
        )
    )

    for name, change in (
        ("stale_artifact", {"artifact": "V1"}),
        ("foreign_workflow", {"workflow": "F1"}),
        ("foreign_wait", {"wait": "W1", "invocation": "I1"}),
        ("unknown_decision", {"decision": "maybe"}),
    ):
        submit: Object = {
            "artifact": "V0",
            "decision": "approve",
            "kind": "decision_submit",
            "wait": "W0",
            "workflow": "F0",
        }
        submit.update(change)
        c.append(_case("decisions", name, {}, [*opened, submit]))

    ok: Object = {"artifact": "V0", "decision": "approve", "kind": "decision_submit", "wait": "W0", "workflow": "F0"}

    c += [
        _case("decisions", "duplicate_response", {}, [*opened, ok, ok]),
        _case("decisions", "deadline", {}, [*opened, {"kind": "decision_deadline", "wait": "W0"}]),
        _case(
            "decisions",
            "cancel_condition",
            _runtime_decl("cancel_after_start", None, "cancelled"),
            [
                {"kind": "bridge_start", "node": "N0", "task": "T0"},
                {
                    "condition": "cancel_after_start",
                    "kind": "bridge_condition",
                    "task": "T0",
                },
                {"category": "cancelled", "kind": "bridge_emit", "outcome": None, "task": "T0"},
            ],
        ),
        _case(
            "decisions",
            "implementation_failure",
            _runtime_decl("failure", None, "failure", failure="implementation_exception"),
            [
                {"kind": "bridge_start", "node": "N0", "task": "T0"},
                {
                    "condition": "failure",
                    "failure": "implementation_exception",
                    "kind": "bridge_condition",
                    "task": "T0",
                },
                {"category": "failure", "kind": "bridge_emit", "outcome": None, "task": "T0"},
            ],
        ),
    ]

    return c
