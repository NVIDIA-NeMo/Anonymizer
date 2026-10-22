# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent finite compiler observations; imports no product modules."""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Outcome:
    """One finite declared alternative; name is not intrinsically observable."""

    name: str
    ports: tuple[str, ...] = ("out",)
    labels: tuple[str, ...] = ("public",)
    context: tuple[str, ...] = ()
    state: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    cost: int = 0
    dynamic: tuple[str, ...] = ()
    subgraph: tuple[str, ...] = ()
    selected: tuple[int, ...] = (1, 2)
    interface: str = "done"


@dataclass(frozen=True)
class Case:
    """Three-node chain with an optional selector bypass of its middle node."""

    name: str
    nodes: tuple[tuple[Outcome, ...], ...]


@dataclass(frozen=True, order=True)
class Observation:
    """Observable selected topology and exact declared effect inventory."""

    selected: tuple[int, ...]
    ports: tuple[tuple[int, tuple[str, ...]], ...]
    labels: tuple[tuple[int, tuple[str, ...]], ...]
    context: tuple[tuple[int, tuple[str, ...]], ...]
    state: tuple[tuple[int, tuple[str, ...]], ...]
    evidence: tuple[tuple[int, tuple[str, ...]], ...]
    cost: int
    dynamic: tuple[tuple[int, tuple[str, ...]], ...]
    subgraph: tuple[tuple[int, tuple[str, ...]], ...]
    interface: str


def observe_assignment(assignment: tuple[Outcome, ...]) -> Observation:
    """Project a complete selected history, retaining declaration identities."""
    selected = (0, *assignment[0].selected)
    return Observation(
        selected=selected,
        ports=tuple((node, assignment[node].ports) for node in selected),
        labels=tuple((node, assignment[node].labels) for node in selected),
        context=tuple((node, assignment[node].context) for node in selected),
        state=tuple((node, assignment[node].state) for node in selected),
        evidence=tuple((node, assignment[node].evidence) for node in selected),
        cost=sum(assignment[node].cost for node in selected),
        dynamic=tuple((node, assignment[node].dynamic) for node in selected),
        subgraph=tuple((node, assignment[node].subgraph) for node in selected),
        interface=assignment[2].interface,
    )


def enumerate_observations(case: Case) -> tuple[Observation, ...]:
    """Exhaustively evaluate all alternatives without any equivalence quotient."""
    return tuple(sorted({observe_assignment(items) for items in itertools.product(*case.nodes)}))


def quotient_case(case: Case, omitted: str | None = None) -> Case:
    """Candidate quotient used only to calibrate the oracle and its mutants."""
    nodes: list[tuple[Outcome, ...]] = []
    for index, alternatives in enumerate(case.nodes):
        representatives: dict[str, Outcome] = {}
        for outcome in alternatives:
            signature = asdict(outcome)
            del signature["name"]
            if index != 0:
                del signature["selected"]
            if index != 2:
                del signature["interface"]
            if omitted is not None:
                signature.pop(omitted, None)
            representatives.setdefault(json.dumps(signature, sort_keys=True), outcome)
        nodes.append(tuple(representatives.values()))
    return Case(name=case.name, nodes=tuple(nodes))


def generate_cases() -> tuple[Case, ...]:
    """Generate a fixed alphabet, complete binary products and near neighbors."""
    cases: list[Case] = []
    for duplicate_mask in range(8):
        nodes = tuple(
            (Outcome(name="a"), Outcome(name="b")) if duplicate_mask & (1 << node) else (Outcome(name="a"),)
            for node in range(3)
        )
        cases.append(Case(name=f"unobserved-{duplicate_mask}", nodes=nodes))
    changes: tuple[tuple[str, Outcome], ...] = (
        ("ports", Outcome(name="b", ports=("alternate",))),
        ("labels", Outcome(name="b", labels=("restricted",))),
        ("context", Outcome(name="b", context=("context-v1",))),
        ("state", Outcome(name="b", state=("state-v2",))),
        ("evidence", Outcome(name="b", evidence=("assessment-v1",))),
        ("cost", Outcome(name="b", cost=1)),
        ("dynamic", Outcome(name="b", dynamic=("map-key-version",))),
        ("subgraph", Outcome(name="b", subgraph=("child-effect",))),
    )
    for dimension, changed in changes:
        for node in range(3):
            nodes = [(Outcome(name="a"), Outcome(name="b")) for _ in range(3)]
            nodes[node] = (Outcome(name="a"), changed)
            cases.append(Case(name=f"neighbor-{dimension}-{node}", nodes=tuple(nodes)))
    for selector, interface in itertools.product((False, True), repeat=2):
        nodes = (
            (Outcome(name="a"), Outcome(name="b", selected=(2,) if selector else (1, 2))),
            (Outcome(name="a"), Outcome(name="b", state=("middle",))),
            (Outcome(name="a"), Outcome(name="b", interface="alternate" if interface else "done")),
        )
        cases.append(Case(name=f"observed-{int(selector)}-{int(interface)}", nodes=nodes))
    return tuple(cases)


def encode_corpus() -> bytes:
    """Emit deterministic complete expected observations and bad summaries."""
    rows: list[dict[str, object]] = []
    for case in generate_cases():
        observations = enumerate_observations(case)
        rows.append(
            {
                "case": asdict(case),
                "raw_assignment_count": len(tuple(itertools.product(*case.nodes))),
                "expected": [asdict(item) for item in observations],
                "rejected_summary": [asdict(replace(item, cost=item.cost + 1)) for item in observations],
            }
        )
    return (json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n").encode()


if __name__ == "__main__":
    Path(__file__).with_name("cases.json").write_bytes(encode_corpus())
