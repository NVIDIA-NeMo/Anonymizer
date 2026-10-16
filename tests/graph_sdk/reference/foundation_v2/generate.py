# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate bounded alphabets without production-normalized expected values."""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict
from pathlib import Path

from tests.graph_sdk.reference.foundation_v2.predicates import (
    OccurrenceEvent,
    RequestEvent,
    final_loop,
    has_disjoint_outputs,
    has_total_order,
    is_legal_activation,
    is_legal_requests,
    is_partition,
    map_join,
    matches_window,
    may_close,
    may_reissue,
    terminal_projection,
)


def generate_cases() -> list[dict[str, object]]:
    """Enumerate independent axes and selected causal interactions."""
    rows: list[dict[str, object]] = []
    edges = tuple(itertools.permutations(range(3), 2))
    for bits in itertools.product((False, True), repeat=6):
        chosen = tuple(edge for edge, selected in zip(edges, bits, strict=True) if selected)
        rows.append({"family": "dag", "edges": chosen, "expected": has_total_order((0, 1, 2), chosen)})
    for start, end, derived in itertools.product(range(-1, 4), range(-1, 4), ("", "a", "é", "aé")):
        rows.append(
            {
                "family": "window",
                "start": start,
                "end": end,
                "derived": derived,
                "expected": matches_window("aé", derived, start, end),
            }
        )
    windows = tuple((root, start, end) for root in range(2) for start in range(3) for end in range(start, 3))
    for pair in itertools.combinations_with_replacement(windows, 2):
        rows.append({"family": "ownership", "windows": pair, "expected": has_disjoint_outputs(pair)})
    groups = (frozenset(), frozenset({0}), frozenset({1}), frozenset({0, 1}), frozenset({2}))
    for pair in itertools.product(groups, repeat=2):
        rows.append(
            {"family": "groups", "groups": [sorted(x) for x in pair], "expected": is_partition(frozenset({0, 1}), pair)}
        )
    atoms = (
        OccurrenceEvent("A", "start"),
        OccurrenceEvent("A", "terminal", "success"),
        OccurrenceEvent("B", "start"),
        OccurrenceEvent("B", "terminal", "success"),
    )
    for linked, permutation in itertools.product((False, True), itertools.permutations(atoms)):
        dependency = (("A", "B"),) if linked else ()
        rows.append(
            {
                "family": "activation",
                "linked": linked,
                "events": [asdict(x) for x in permutation],
                "expected": is_legal_activation(frozenset({"A", "B"}), dependency, permutation, True),
            }
        )
    for expansion, closed, count, status in itertools.product(
        ("success", "failed", "overflow"), (False, True), range(3), ("success", "failure", "missing")
    ):
        members = frozenset(str(x) for x in range(count))
        terminals = tuple((member, status) for member in sorted(members)) if status != "missing" else ()
        rows.append(
            {
                "family": "map",
                "expansion": expansion,
                "closed": closed,
                "members": sorted(members),
                "terminals": terminals,
                "expected": map_join(expansion, closed, members, terminals),
            }
        )
    for bound, entered, length in itertools.product(range(3), (False, True), range(4)):
        for outcomes in itertools.product(("continue", "exit", "failed"), repeat=length):
            rows.append(
                {
                    "family": "loop",
                    "bound": bound,
                    "entered": entered,
                    "outcomes": outcomes,
                    "expected": final_loop(outcomes, bound, entered),
                }
            )
    requests = (
        RequestEvent(0, "reserve", (0,)),
        RequestEvent(0, "dispatch"),
        RequestEvent(0, "terminal", value="lost"),
        RequestEvent(0, "settlement", usage=3, stopped=True),
    )
    for budget, permutation in itertools.product(range(3), itertools.permutations(requests)):
        rows.append(
            {
                "family": "requests",
                "budget": budget,
                "events": [asdict(x) for x in permutation],
                "expected": is_legal_requests(permutation, budget),
                "projection": terminal_projection(permutation, 0),
            }
        )
    for returned in ((), (0,), (1,), (0, 0), (0, 1), (1, 0), (0, 1, 2)):
        events = (RequestEvent(0, "reserve", (0, 1)), RequestEvent(0, "dispatch"), RequestEvent(0, "result", returned))
        rows.append({"family": "associations", "returned": returned, "expected": terminal_projection(events, 0)})
    for owner, replay, failure, purpose, attempts, alternate in itertools.product(
        ("none", "executor"),
        ("never", "before_acceptance", "idempotent"),
        (
            "retryable",
            "transport_unknown",
            "rejected_before_acceptance",
            "malformed_response",
            "permanent",
            "implementation_exception",
        ),
        ("retry", "correction", "failover"),
        (0, 2),
        (False, True),
    ):
        args = (owner, replay, failure, purpose, attempts, 2, alternate)
        rows.append({"family": "reissue", "args": args, "expected": may_reissue(*args)})
    for args in itertools.product((False, True), repeat=4):
        rows.append({"family": "resource", "args": args, "expected": may_close(*args)})
    for index, row in enumerate(rows):
        row["case_id"] = f"foundation-{index:04d}"
    return rows


def encode_corpus() -> bytes:
    """Encode repeatably; list ordering is part of this reference version."""
    return (json.dumps(generate_cases(), sort_keys=True, separators=(",", ":")) + "\n").encode()


if __name__ == "__main__":
    Path(__file__).with_name("cases.json").write_bytes(encode_corpus())
