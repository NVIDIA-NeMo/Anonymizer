# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent effects_v1 reference: binding cases."""

from __future__ import annotations

from tests.graph_sdk.reference._effects_v1.builders import (
    _bind,
    _binding_decl,
    _decl,
    _dispatch,
    _reserve,
    _result,
    _source_failure,
    _source_result,
    _trace,
    binding,
)
from tests.graph_sdk.reference._effects_v1.case import (
    _case,
)
from tests.graph_sdk.reference._effects_v1.model import (
    Object,
)


def binding_cases() -> list[Object]:
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
        _case("inflight", "dispatch_sets_both", _decl(), base),
        _case("inflight", "lost_keeps_remote", _decl(), base + [{"kind": "lost", "request": "R0"}]),
        _case(
            "inflight",
            "trusted_settlement_clears_remote",
            _decl(),
            base + [{"kind": "lost", "request": "R0"}, settlement],
        ),
        _case(
            "inflight",
            "unknown_settlement_keeps_remote",
            _decl(),
            base
            + [
                {"kind": "lost", "request": "R0"},
                dict(settlement, disposition="unknown", remote_stopped=None, usage="unknown"),
            ],
        ),
        _case("inflight", "result_clears_both", _decl(), base + [_result("R0", ("T0",), ("T0",))]),
    ]

    c += [
        _case(
            "binding",
            "two_sources_same_key",
            _binding_decl({"D0": "S0", "D1": "S1"}),
            binding("D0", "S0", "R0", "a") + binding("D1", "S1", "R1", "b") + [{"kind": "binding_finish"}],
        ),
        _case(
            "binding",
            "one_source_two_declarations",
            _binding_decl({"D0": "S0", "D1": "S0"}),
            [
                {"kind": "resource", "owner": "sdk", "resource": "Q0", "safe_detachment": "forbidden"},
                *binding("D0", "S0", "R0", "a"),
                *binding("D1", "S0", "R1", "b"),
                {"kind": "binding_finish"},
            ],
        ),
        _case(
            "binding",
            "required_failure_preserves_prior",
            _binding_decl({"D0": "S0", "D1": "S1"}),
            binding("D0", "S0", "R0", "a")
            + [
                _bind("D1", "P0"),
                _reserve("R1", ["D1"], purpose="initial_binding"),
                _dispatch("R1"),
                _source_failure("D1", "S1", request="R1"),
            ],
        ),
        _case(
            "binding",
            "optional_failure_partial",
            _binding_decl({"D0": "S0", "D1": "S1"}, optional=("D1",)),
            binding("D0", "S0", "R0", "a")
            + [
                _bind("D1", "P0"),
                _reserve("R1", ["D1"], purpose="initial_binding"),
                _dispatch("R1"),
                _source_failure("D1", "S1", request="R1"),
            ],
        ),
        _case(
            "binding",
            "omitted_optional",
            _binding_decl({"D0": "S0"}, optional=("D0",)),
            [
                _bind("D0", "P0"),
                _reserve("R0", ["D0"], purpose="initial_binding"),
                _dispatch("R0"),
                _source_failure("D0", "S0", request="R0", disposition="omitted_optional"),
                {"kind": "binding_finish"},
            ],
        ),
        _case(
            "binding",
            "response_reordered",
            _binding_decl({"D0": "S0", "D1": "S1"}),
            binding("D1", "S1", "R1", "b") + binding("D0", "S0", "R0", "a") + [{"kind": "binding_finish"}],
        ),
        _case(
            "binding",
            "oversize_no_truncation",
            _binding_decl({"D0": "S0"}, max_bytes=1),
            binding("D0", "S0", "R0", "aa"),
        ),
        _case(
            "binding",
            "cancel_before_dispatch",
            _binding_decl({"D0": "S0"}),
            [_bind("D0", "P0"), _reserve("R0", ["D0"], purpose="initial_binding"), {"kind": "cancel", "request": "R0"}],
        ),
        _case(
            "binding",
            "cancel_after_dispatch_lost",
            _binding_decl({"D0": "S0"}),
            binding("D0", "S0", "R0", "a")[:3]
            + [{"kind": "cancel", "request": "R0"}, {"kind": "lost", "request": "R0"}],
        ),
        _case(
            "binding",
            "invalid_local_zero_effects",
            {"binding_targets": ["B0"], "targets": ["A0"]},
            [],
            "admission",
        ),
    ]

    exact_bounds = _binding_decl({"D0": "S0"}, max_bytes=2, max_items=2)

    one_byte = _binding_decl({"D0": "S0"}, max_bytes=1, max_items=2)

    one_item = _binding_decl({"D0": "S0"}, max_bytes=2, max_items=1)

    two_items = _source_result(
        "D0",
        "S0",
        [
            {"association": "D0", "key": 0, "text": "a", "version": 1},
            {"association": "D0", "key": 1, "text": "b", "version": 1},
        ],
        request="R0",
    )

    c += [
        _case(
            "binding",
            "exact_item_byte_bounds",
            exact_bounds,
            [_bind("D0", "P0"), _reserve("R0", ["D0"]), _dispatch("R0"), two_items, {"kind": "binding_finish"}],
        ),
        _case(
            "binding",
            "one_over_byte_bound",
            one_byte,
            [_bind("D0", "P0"), _reserve("R0", ["D0"]), _dispatch("R0"), two_items],
        ),
        _case(
            "binding",
            "one_over_item_bound",
            one_item,
            [_bind("D0", "P0"), _reserve("R0", ["D0"]), _dispatch("R0"), two_items],
        ),
        _case(
            "binding",
            "unsolicited_source_result",
            _binding_decl({"D0": "S0"}),
            [dict(_result("R0", ("D0",), ("D0",)), outcomes={"D0": "retrieved"})],
        ),
        _case(
            "binding",
            "wrong_source",
            _binding_decl({"D0": "S0"}, max_requests=1),
            [
                _bind("D0", "P0"),
                _reserve("R0", ["D0"]),
                _dispatch("R0"),
                _source_result("D0", "S1", [{"association": "D0", "key": 0, "version": 1, "text": "a"}], request="R0"),
                {"kind": "binding_finish"},
            ],
        ),
        _case(
            "binding",
            "missing_result",
            _binding_decl({"D0": "S0"}, max_requests=1),
            [
                _bind("D0", "P0"),
                _reserve("R0", ["D0"]),
                _dispatch("R0"),
                _source_result("D0", "S0", [], request="R0"),
            ],
        ),
        _case(
            "binding",
            "duplicate_result",
            _binding_decl({"D0": "S0"}, max_requests=1),
            [
                _bind("D0", "P0"),
                _reserve("R0", ["D0"]),
                _dispatch("R0"),
                _source_result(
                    "D0",
                    "S0",
                    [
                        {"association": "D0", "key": 0, "text": "a", "version": 1},
                        {"association": "D0", "key": 0, "text": "a", "version": 1},
                    ],
                    request="R0",
                ),
            ],
        ),
        _case(
            "binding",
            "foreign_result_association",
            _binding_decl({"D0": "S0"}, max_requests=1),
            [
                _bind("D0", "P0"),
                _reserve("R0", ["D0"]),
                _dispatch("R0"),
                _source_result(
                    "D0",
                    "S0",
                    [{"association": "D1", "key": 0, "text": "a", "version": 1}],
                    request="R0",
                ),
            ],
        ),
    ]

    c += [
        _case(
            "resources",
            "caller_left_open",
            {},
            [
                {"kind": "resource", "owner": "caller", "resource": "Q0", "safe_detachment": "forbidden"},
                {"kind": "close_resource", "resource": "Q0"},
            ],
        ),
        _case(
            "resources",
            "sdk_closed",
            {},
            [
                {"kind": "resource", "owner": "sdk", "resource": "Q0", "safe_detachment": "forbidden"},
                {"kind": "close_resource", "resource": "Q0"},
            ],
        ),
        _case(
            "resources",
            "sdk_close_failed",
            {},
            [
                {"kind": "resource", "owner": "sdk", "resource": "Q0", "safe_detachment": "forbidden"},
                {"disposition": "close_failed", "kind": "close_resource", "resource": "Q0"},
            ],
        ),
        _case(
            "resources",
            "sdk_close_unknown",
            {},
            [
                {"kind": "resource", "owner": "sdk", "resource": "Q0", "safe_detachment": "forbidden"},
                {"disposition": "close_unknown", "kind": "close_resource", "resource": "Q0"},
            ],
        ),
    ]

    for name, detach, terminal in (
        ("sdk_remote_waits", "forbidden", "lost"),
        ("sdk_safe_detach", "independent_after_dispatch", "lost"),
        ("local_inflight_waits", "independent_after_dispatch", None),
        ("trusted_stop_then_close", "forbidden", "stop"),
    ):
        events: list[Object] = [
            {"kind": "resource", "owner": "sdk", "resource": "Q0", "safe_detachment": detach},
            *base,
        ]
        if terminal == "lost":
            events.append({"kind": "lost", "request": "R0"})
        elif terminal == "stop":
            events += [
                {"kind": "cancel", "request": "R0"},
                {"kind": "stop", "request": "R0", "usage": "unknown"},
            ]
        events.append({"kind": "close_resource", "resource": "Q0"})
        c.append(_case("resources", name, _decl(), events))

    c.append(
        _case(
            "bridges",
            "cancel_before_start",
            {},
            [{"activation": "A0", "category": "blocked", "kind": "bridge_close_unstarted"}],
        )
    )

    return c
