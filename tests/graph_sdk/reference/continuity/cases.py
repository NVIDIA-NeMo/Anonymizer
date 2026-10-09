# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path

from tests.graph_sdk.reference.continuity.reference import (
    Bundle,
    Decision,
    Evidence,
    Support,
    Target,
    _fixture_bundle,
    greatest,
)


@dataclass(frozen=True)
class Case:
    name: str
    bundle: Bundle
    conflicts: frozenset[str]


def support(key: str, *, node: str = "N") -> Support:
    return Support("stable-" + key, node, "ok", "P" + key, frozenset({"full"}))


def evidence(key: str, target: str, *, consumes: frozenset[str] | None = None) -> Evidence:
    return Evidence(
        key + target,
        "N",
        "ok",
        "P" + key,
        frozenset({"full"}),
        frozenset({key}) if consumes is None else consumes,
    )


def build(
    x_on: str = "AB",
    y_on: bool = True,
    dependency: str = "none",
    atomic: bool = False,
    *,
    consumes_y: bool = False,
) -> Bundle:
    rows = tuple(evidence("x", t, consumes=frozenset({"x", "y"}) if consumes_y else None) for t in x_on) + (
        (evidence("y", "B"),) if y_on else ()
    )
    targets = (
        Target(
            "A",
            frozenset({"x", "y"}) if consumes_y else frozenset({"x"}),
            frozenset(e.reference for e in rows if e.reference.endswith("A")),
            frozenset({"B"}) if dependency == "A-B" else frozenset(),
            frozenset({"C"}) if atomic else frozenset(),
        ),
        Target(
            "B",
            frozenset({"x", "y"}),
            frozenset(e.reference for e in rows if e.reference.endswith("B")),
        ),
        Target(
            "C",
            frozenset(),
            frozenset(),
            frozenset({dependency[-1]}) if dependency.startswith("C-") else frozenset(),
            frozenset({"A"}) if atomic else frozenset(),
        ),
    )
    return _fixture_bundle(targets, (Decision("x", (support("x"),)), Decision("y", (support("y"),))), rows)


def generated() -> tuple[Case, ...]:
    return tuple(
        Case(
            f"closure-{index:03}",
            build(x_on, y_on, dependency, atomic),
            frozenset(conflict),
        )
        for index, (x_on, y_on, dependency, atomic, conflict) in enumerate(
            product(
                ("", "A", "B", "AB"),
                (False, True),
                ("none", "C-A", "C-B", "A-B"),
                (False, True),
                ("", "x", "y", "xy"),
            )
        )
    )


def known() -> tuple[tuple[Case, frozenset[str]], ...]:
    base = build()
    rows: list[tuple[Case, frozenset[str]]] = [
        (Case("independent-A", base, frozenset({"y"})), frozenset({"A", "C"})),
        (Case("sole-B", build("B"), frozenset({"y"})), frozenset({"C"})),
        (
            Case("consumes-y", build(consumes_y=True), frozenset({"y"})),
            frozenset({"C"}),
        ),
        (
            Case("zero-decision-dependent", build(dependency="C-B"), frozenset({"y"})),
            frozenset({"A"}),
        ),
        (
            Case(
                "alternating-datum-atomic-support",
                build("B", dependency="C-B", atomic=True),
                frozenset({"y"}),
            ),
            frozenset(),
        ),
        (Case("all-supported", base, frozenset()), frozenset({"A", "B", "C"})),
    ]
    atomic_targets = tuple(
        replace(t, atomic=frozenset({"B" if t.name == "A" else "A"})) if t.name in {"A", "B"} else t
        for t in base.targets
    )
    rows.append(
        (
            Case(
                "independent-atomic-AB",
                _fixture_bundle(atomic_targets, base.decisions, base.verified),
                frozenset({"y"}),
            ),
            frozenset({"C"}),
        )
    )
    for field, changed in (
        ("node", "foreign"),
        ("outcome", "other"),
        ("promise", "other"),
        ("observed_revision", 0),
        ("coverage", frozenset()),
        ("consumed", frozenset()),
    ):
        facts = tuple(replace(e, **{field: changed}) if e.reference == "xA" else e for e in base.verified)
        rows.append(
            (
                Case(
                    "support-" + field,
                    _fixture_bundle(base.targets, base.decisions, facts),
                    frozenset({"y"}),
                ),
                frozenset({"C"}),
            )
        )
    extra = Decision("x", (support("x"), support("x", node="other")))
    rows.append(
        (
            Case(
                "every-declared-support",
                _fixture_bundle(base.targets, (extra, base.decisions[1]), base.verified),
                frozenset({"y"}),
            ),
            frozenset({"C"}),
        )
    )
    # A3 named extension: B conflict -> D datum -> C support -> E atomic -> F datum.
    targets = (
        Target("A", frozenset({"x"}), frozenset({"xA"})),
        Target("B", frozenset({"y"}), frozenset({"yB"})),
        Target("D", frozenset({"z"}), frozenset({"zD"}), frozenset({"B"})),
        Target("C", frozenset({"z"}), frozenset(), atomic=frozenset({"E"})),
        Target("E", frozenset(), frozenset(), atomic=frozenset({"C"})),
        Target("F", frozenset(), frozenset(), frozenset({"E"})),
        Target("U", frozenset(), frozenset()),
    )
    extended = _fixture_bundle(
        targets,
        tuple(Decision(k, (support(k),)) for k in "xyz"),
        (evidence("x", "A"), evidence("y", "B"), evidence("z", "D")),
    )
    rows.append(
        (
            Case("alternating-seven-target", extended, frozenset({"y"})),
            frozenset({"A", "U"}),
        )
    )
    rows.append((Case("empty-qualified", _fixture_bundle((), (), ()), frozenset()), frozenset()))
    return tuple(rows)


def json_default(value: object) -> object:
    if isinstance(value, frozenset):
        return sorted(value)
    raise TypeError(type(value).__name__)


def corpus_bytes() -> bytes:
    rows = [
        {
            "id": case.name,
            "facts": asdict(case),
            "expected_survivors": sorted(greatest(case.bundle, case.conflicts)),
        }
        for case in generated() + tuple(case for case, _ in known())
    ]
    return (json.dumps(rows, default=json_default, sort_keys=True, separators=(",", ":")) + "\n").encode()


if __name__ == "__main__":
    import sys

    Path(sys.argv[1]).write_bytes(corpus_bytes())
