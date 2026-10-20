# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from tests.graph_sdk.reference.observations.reference import (
    OUTCOMES,
    PURPOSES,
    REASONS,
    Dispatch,
    Limits,
    Pair,
    Phase,
    Scope,
    account,
    observe,
    requirements,
)

LIMITS = Limits(100, 200, 10000, 20, 100000, 100, 100)


@dataclass(frozen=True)
class ObservationCase:
    name: str
    phases: tuple[Phase, ...]
    limits: Limits = LIMITS
    sink_failure_at: int | None = None
    sink_configured: bool = False


def pair(
    ordinal: int = 0,
    *,
    parent: int | None = None,
    boundary: str = "execution",
    outcome: str = "success",
    reasons: tuple[str, ...] = (),
) -> Pair:
    return Pair(ordinal, parent, boundary, ordinal, parent, boundary, outcome, reasons, 10, 12)


def phase(*pairs: Pair, forecast_pairs: int | None = None) -> Phase:
    return Phase(
        requirements(len(pairs) if forecast_pairs is None else forecast_pairs, LIMITS),
        pairs,
    )


def observations() -> tuple[ObservationCase, ...]:
    rows = [ObservationCase("outcome-" + outcome, (phase(pair(outcome=outcome)),)) for outcome in OUTCOMES]
    rows += [
        ObservationCase("reason-" + reason, (phase(pair(outcome="failure", reasons=(reason,))),)) for reason in REASONS
    ]
    rows += [
        ObservationCase("boundary-" + boundary, (phase(pair(boundary=boundary)),))
        for boundary in (
            "execution",
            "operation",
            "physical_request",
            "cleanup",
            "qualification",
            "release",
        )
    ]
    nested = (phase(pair(), pair(1, parent=0, boundary="operation")),)
    rows += [
        ObservationCase("nested", nested),
        ObservationCase("sink-zero", nested, sink_failure_at=0, sink_configured=True),
        ObservationCase("sink-one", nested, sink_failure_at=1, sink_configured=True),
        ObservationCase("sink-complete", nested, sink_configured=True),
    ]
    need = requirements(2, LIMITS)
    exact = Limits(need.pairs, need.events, need.nodes, need.depth, need.encoded_bytes)
    for field in ("pairs", "events", "nodes", "depth", "encoded_bytes"):
        rows.append(ObservationCase("phase-exact-" + field, (phase(pair(), forecast_pairs=2),), exact))
        rows.append(
            ObservationCase(
                "phase-short-" + field,
                (phase(pair(), forecast_pairs=2),),
                replace(exact, **{field: getattr(exact, field) - 1}),
            )
        )
    for label, fact in (
        ("no-start", replace(pair(), start_present=False)),
        ("no-terminal", replace(pair(), terminal_present=False)),
        ("terminal-identity", replace(pair(), terminal_ordinal=1)),
        ("terminal-boundary", replace(pair(), terminal_boundary="cleanup")),
        ("foreign-parent", replace(pair(), parent=2, terminal_parent=2)),
        ("clock-reverse", replace(pair(), end_ns=9)),
        ("duration-over", replace(pair(), end_ns=111)),
        ("reason-canary", replace(pair(), reasons=("秘密-canary",))),
    ):
        rows.append(ObservationCase(label, (phase(fact),)))
    rows += [
        ObservationCase("duplicate-correlation", (phase(pair(), pair()),)),
        ObservationCase("foreign-owner", (replace(phase(pair()), owner=1),)),
        ObservationCase("phase-underforecast", (phase(pair(), pair(1), forecast_pairs=1),)),
        ObservationCase(
            "second-phase-underfunded",
            (phase(pair()), phase(pair(1), forecast_pairs=2)),
            exact,
        ),
        ObservationCase(
            "empty-phase-underfunded",
            (phase(forecast_pairs=2),),
            replace(exact, pairs=1),
        ),
        ObservationCase("begin-zero-pairs", (), replace(LIMITS, pairs=0)),
    ]
    deep = tuple(pair(i, parent=None if i == 0 else i - 1) for i in range(12))
    deep_need = requirements(12, LIMITS, logical_depth=12)
    deep_limits = Limits(
        deep_need.pairs,
        deep_need.events,
        deep_need.nodes,
        deep_need.depth,
        deep_need.encoded_bytes,
    )
    rows += [
        ObservationCase("logical-depth-exact", (Phase(deep_need, deep),), deep_limits),
        ObservationCase(
            "logical-depth-short",
            (Phase(deep_need, deep),),
            replace(deep_limits, depth=11),
        ),
    ]
    count_two = replace(LIMITS, count=2)
    count_one = replace(LIMITS, count=1)
    rows += [
        ObservationCase(
            "count-exact-two",
            (phase(pair(), pair(1)),),
            count_two,
            sink_configured=True,
        ),
        ObservationCase("count-one-less", (phase(pair(), pair(1)),), count_one, sink_configured=True),
        ObservationCase("count-exact-one", (phase(pair()),), count_one, sink_configured=True),
        ObservationCase("count-zero", (), replace(LIMITS, count=0)),
        ObservationCase(
            "count-cumulative",
            (phase(pair()), phase(pair(1))),
            count_one,
            sink_configured=True,
        ),
        ObservationCase(
            "terminal-ordinal-bool",
            (phase(pair(), replace(pair(1), terminal_ordinal=True)),),
        ),
        ObservationCase(
            "terminal-parent-bool",
            (phase(pair(), pair(1), replace(pair(2, parent=1), terminal_parent=True)),),
        ),
    ]
    return tuple(
        replace(
            case,
            phases=tuple(
                replace(
                    p,
                    need=requirements(p.need.pairs, case.limits, logical_depth=p.need.depth),
                )
                for p in case.phases
            ),
        )
        for case in rows
    )


def accounts() -> tuple[tuple[str, tuple[Scope, ...]], ...]:
    rows: list[tuple[str, tuple[Scope, ...]]] = [
        ("purpose-" + purpose, (Scope((Dispatch(0, purpose, (2, 3)),)),)) for purpose in PURPOSES
    ]
    rows += [
        ("zero-proven", (Scope(()),)),
        ("missing-owner", (Scope(None),)),
        ("unknown-usage", (Scope((Dispatch(0, "initial", None),)),)),
        (
            "partial-usage",
            (Scope((Dispatch(0, "initial", (2, 3)), Dispatch(1, "repair", None))),),
        ),
        (
            "seven-purposes",
            (Scope(tuple(Dispatch(i, purpose, (i, i + 1)) for i, purpose in enumerate(PURPOSES))),),
        ),
        (
            "two-owners",
            (
                Scope((Dispatch(0, "initial_binding", (1, 2)),)),
                Scope((Dispatch(1, "adaptive_retrieval", (3, 4)),)),
            ),
        ),
        (
            "missing-binding-only",
            (
                Scope(None, ("initial_binding",)),
                Scope((Dispatch(0, "initial", (2, 3)),)),
            ),
        ),
    ]
    return tuple(rows)


def generate() -> bytes:
    observation_rows = []
    for case in observations():
        close, delivered = observe(
            case.phases,
            case.limits,
            case.sink_failure_at,
            sink_configured=case.sink_configured,
        )
        observation_rows.append(
            {
                "id": case.name,
                "facts": asdict(case),
                "expected": asdict(close),
                "expected_type": type(close).__name__,
                "delivered_pairs": len(delivered),
            }
        )
    account_rows = [
        {
            "id": name,
            "facts": [asdict(scope) for scope in scopes],
            "expected": asdict(account(scopes)),
        }
        for name, scopes in accounts()
    ]
    return (
        json.dumps(
            {"observations": observation_rows, "accounting": account_rows},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode()


if __name__ == "__main__":
    import sys

    Path(sys.argv[1]).write_bytes(generate())
