# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent effects_v1 reference: budget cases."""

from __future__ import annotations

from tests.graph_sdk.reference._effects_v1.builders import (
    _bind,
    _decl,
    _dispatch,
    _policy,
    _reserve,
    _result,
    _runtime_decl,
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


def budget_cases() -> list[Object]:
    c: list[Object] = []
    for limit in (0, 1, 2):
        events = [_bind("T0", "P0"), _reserve("R0", ["T0"])] + ([_dispatch("R0")] if limit else [])
        c.append(_case("budgets", f"limit_{limit}", _decl(limit), events))

    c += [
        _case("budgets", "partial_two", _decl(1), _trace() + [_bind("T1", "P0"), _reserve("R1", ["T1"])]),
        _case(
            "budgets", "exact_two", _decl(2), _trace() + [_bind("T1", "P0"), _reserve("R1", ["T1"]), _dispatch("R1")]
        ),
        _case(
            "budgets",
            "cancel_reserved_zero_charge",
            _decl(),
            [_bind("T0", "P0"), _reserve("R0", ["T0"]), {"kind": "cancel", "request": "R0"}],
        ),
    ]

    shared = _trace(("T0", "T1"))

    c.append(_case("keyed", "valid_shared_reordered", _decl(), shared + [_result("R0", ("T0", "T1"), ("T1", "T0"))]))

    for name, returned in (
        ("missing", ("T0",)),
        ("duplicate", ("T0", "T0", "T1")),
        ("extra", ("T0", "T1", "T2")),
        ("foreign", ("T0", "T1", "X0")),
    ):
        c.append(_case("keyed", name, _decl(), shared + [_result("R0", ("T0", "T1"), returned)]))

    c += [
        _case("keyed", "single_t0", _decl(), _trace() + [_result("R0", ("T0",), ("T0",))]),
        _case("keyed", "single_t1", _decl(), _trace(("T1",)) + [_result("R0", ("T1",), ("T1",))]),
        _case("keyed", "parent_summary_no_charge", _decl(), shared + [_result("R0", ("T0", "T1"), ("T0", "T1"))]),
    ]

    c += [
        _case("retry", "cross_policy_semantic", _decl(), [_bind("T0", "P0"), _reserve("R0", ["T0"], "P1")]),
        _case("retry", "cross_policy_binding", _decl(), [_bind("D0", "P0"), _reserve("R0", ["D0"], "P1")]),
    ]

    variants = (
        ("retry_success", "retryable", "retry"),
        ("correction", "malformed_response", "correction"),
        ("failover", "permanent", "failover"),
    )

    for name, failure, purpose in variants:
        c.append(
            _case(
                "retry",
                name,
                _decl(),
                _trace()
                + [
                    {"failure": failure, "kind": "failure", "request": "R0"},
                    _reserve("R1", ["T0"], purpose=purpose),
                    _dispatch("R1"),
                ],
            )
        )

    both_mappings = _decl()

    result_mapping = _object(
        _array(_runtime_decl("result", "ok", "success", reported_outcome="ok")["runtime_mappings"])[0]
    )

    inconsistent_mapping = _object(
        _array(_runtime_decl("request_inconsistent", None, "inconsistent")["runtime_mappings"])[0]
    )

    both_mappings.update(
        {
            "declared_outcomes": ["ok"],
            "outcome_categories": {"ok": "success"},
            "runtime_mappings": [result_mapping, inconsistent_mapping],
        }
    )

    c += [
        _case(
            "bridges",
            "inconsistent_request_cannot_emit_success",
            both_mappings,
            _trace(("T0", "T1"))
            + [
                {"kind": "bridge_start", "node": "N0", "task": "T0"},
                _result("R0", ("T0", "T1"), ("T0",)),
                {"condition": "result", "kind": "bridge_condition", "reported_outcome": "ok", "task": "T0"},
            ],
        ),
        _case(
            "bridges",
            "valid_request_cannot_emit_inconsistent",
            both_mappings,
            _trace(("T0", "T1"))
            + [
                {"kind": "bridge_start", "node": "N0", "task": "T0"},
                _result("R0", ("T0", "T1"), ("T0", "T1")),
                {"condition": "request_inconsistent", "kind": "bridge_condition", "task": "T0"},
            ],
        ),
    ]

    c.append(
        _case(
            "retry",
            "repair_new_task",
            _decl(),
            _trace()
            + [
                {"failure": "permanent", "kind": "failure", "request": "R0"},
                _bind("T1", "P0"),
                _reserve("R1", ["T1"], purpose="repair"),
                _dispatch("R1"),
            ],
        )
    )

    c.append(
        _case(
            "retry",
            "mixed_exhausted_shared",
            _decl(attempts=1),
            _trace()
            + [
                {"failure": "retryable", "kind": "failure", "request": "R0"},
                _bind("T1", "P0"),
                _reserve("R1", ["T0", "T1"]),
                _dispatch("R1"),
            ],
        )
    )

    for replay, failure in (
        ("never", "retryable"),
        ("before_acceptance", "rejected_before_acceptance"),
        ("idempotent", "transport_unknown"),
    ):
        declaration = _decl()
        _object(declaration["policies"])["P0"] = _policy(2, replay)
        c.append(
            _case(
                "retry",
                f"replay_{replay}_{failure}",
                declaration,
                _trace()
                + [
                    {"failure": failure, "kind": "failure", "request": "R0"},
                    _reserve("R1", ["T0"], purpose="retry"),
                    _dispatch("R1"),
                ],
            )
        )

    c += [
        _case(
            "retry",
            "budget_denies_retry",
            _decl(1),
            _trace()
            + [{"failure": "retryable", "kind": "failure", "request": "R0"}, _reserve("R1", ["T0"], purpose="retry")],
        ),
        _case(
            "retry",
            "cross_association_predecessor",
            _decl(),
            _trace()
            + [
                {"failure": "retryable", "kind": "failure", "request": "R0"},
                _bind("T1", "P0"),
                _reserve("R1", ["T1"], purpose="retry"),
            ],
        ),
        _case(
            "retry",
            "implementation_owner_rejected",
            {
                "execution_policies": [
                    {
                        "implementations": [{"physical_policy": "P0"}],
                        "kind": "external",
                        "node": "N0",
                        "physical_policy": "P0",
                        "retry_owner": "implementation",
                    }
                ]
            },
            [],
            "admission",
        ),
        _case(
            "retry",
            "retry_after_permanent",
            _decl(),
            _trace()
            + [
                {"failure": "permanent", "kind": "failure", "request": "R0"},
                _reserve("R1", ["T0"], purpose="retry"),
            ],
        ),
        _case(
            "retry",
            "correction_after_retryable",
            _decl(),
            _trace()
            + [
                {"failure": "retryable", "kind": "failure", "request": "R0"},
                _reserve("R1", ["T0"], purpose="correction"),
            ],
        ),
        _case(
            "retry",
            "failover_after_malformed",
            _decl(),
            _trace()
            + [
                {"failure": "malformed_response", "kind": "failure", "request": "R0"},
                _reserve("R1", ["T0"], purpose="failover"),
            ],
        ),
        _case(
            "retry",
            "late_failure_does_not_change_authority",
            _decl(),
            _trace()
            + [
                {"failure": "permanent", "kind": "failure", "request": "R0"},
                {"failure": "retryable", "kind": "failure", "request": "R0"},
                _reserve("R1", ["T0"], purpose="retry"),
            ],
        ),
    ]

    return c
