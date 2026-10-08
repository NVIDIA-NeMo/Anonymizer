# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent effects_v1 reference: race cases."""

from __future__ import annotations

from tests.graph_sdk.reference._effects_v1.builders import (
    _bind,
    _decl,
    _reserve,
    _result,
    _trace,
)
from tests.graph_sdk.reference._effects_v1.case import (
    _case,
)
from tests.graph_sdk.reference._effects_v1.model import (
    Object,
)


def race_cases() -> list[Object]:
    c: list[Object] = []
    base = _trace()

    settlement: Object = {
        "disposition": "completed",
        "kind": "settlement",
        "remote_stopped": True,
        "request": "R0",
        "usage": {"input": 1, "output": 1},
    }

    c += [
        _case("races", "success", _decl(), base + [_result("R0", ("T0",), ("T0",))]),
        _case("races", "failure", _decl(), base + [{"failure": "permanent", "kind": "failure", "request": "R0"}]),
        _case(
            "races",
            "cancel_trusted_stop",
            _decl(),
            base + [{"kind": "cancel", "request": "R0"}, {"kind": "stop", "request": "R0", "usage": "unknown"}],
        ),
        _case(
            "races",
            "cancel_unknown_lost",
            _decl(),
            base + [{"kind": "cancel", "request": "R0"}, {"kind": "lost", "request": "R0"}],
        ),
        _case(
            "races",
            "cancel_then_result",
            _decl(),
            base + [{"kind": "cancel", "request": "R0"}, _result("R0", ("T0",), ("T0",))],
        ),
        _case(
            "races",
            "result_then_cancel",
            _decl(),
            base + [_result("R0", ("T0",), ("T0",)), {"kind": "cancel", "request": "R0"}],
        ),
        _case(
            "races",
            "lost_late_result_settlement",
            _decl(),
            base + [{"kind": "lost", "request": "R0"}, _result("R0", ("T0",), ("T0",)), settlement],
        ),
        _case(
            "races",
            "lost_late_result_without_settlement",
            _decl(),
            base + [{"kind": "lost", "request": "R0"}, _result("R0", ("T0",), ("T0",))],
        ),
        _case(
            "races",
            "settlement_before_result",
            _decl(),
            base + [settlement, _result("R0", ("T0",), ("T0",))],
            traces=[base + [_result("R0", ("T0",), ("T0",)), settlement]],
        ),
        _case(
            "races",
            "failure_then_settlement",
            _decl(),
            base + [{"failure": "permanent", "kind": "failure", "request": "R0"}, settlement],
        ),
        _case(
            "races",
            "settlement_then_failure",
            _decl(),
            base + [settlement, {"failure": "permanent", "kind": "failure", "request": "R0"}],
        ),
        _case(
            "races",
            "lost_unknown_settlement",
            _decl(),
            base
            + [
                {"kind": "lost", "request": "R0"},
                dict(settlement, disposition="unknown", remote_stopped=None, usage="unknown"),
            ],
        ),
        _case(
            "races",
            "cancel_stop_late_result",
            _decl(),
            base
            + [
                {"kind": "cancel", "request": "R0"},
                {"kind": "stop", "request": "R0", "usage": "unknown"},
                _result("R0", ("T0",), ("T0",)),
            ],
        ),
        _case(
            "races",
            "identical_terminal",
            _decl(),
            base + [_result("R0", ("T0",), ("T0",)), _result("R0", ("T0",), ("T0",))],
        ),
        _case("races", "identical_settlement", _decl(), base + [settlement, settlement]),
        _case(
            "races",
            "conflicting_terminal",
            _decl(),
            base + [_result("R0", ("T0",), ("T0",)), {"failure": "permanent", "kind": "failure", "request": "R0"}],
        ),
        _case(
            "races", "conflicting_settlement", _decl(), base + [settlement, dict(settlement, disposition="rejected")]
        ),
        _case("races", "scope_cancel", _decl(), [_bind("T0", "P0"), _reserve("R0", ["T0"]), {"kind": "scope_cancel"}]),
        _case(
            "races",
            "stop_missing_usage",
            _decl(),
            base + [{"kind": "cancel", "request": "R0"}, {"kind": "stop", "request": "R0"}],
        ),
        _case(
            "races", "settlement_completed_without_remote_stop", _decl(), base + [dict(settlement, remote_stopped=None)]
        ),
        _case(
            "races", "settlement_unknown_with_remote_stop", _decl(), base + [dict(settlement, disposition="unknown")]
        ),
        _case(
            "races",
            "settlement_invalid_remote_stopped_type",
            _decl(),
            base + [dict(settlement, disposition="unknown", remote_stopped="invalid", usage="unknown")],
        ),
    ]

    return c
