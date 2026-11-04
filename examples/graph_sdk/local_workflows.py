# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider-free example of one local graph workflow: declare, prepare, protect, read the outcomes."""

from __future__ import annotations

from anonymizer import graph as g

NAMES = ("Alice", "Bob")


def redact_alice(text: str) -> str:
    """A transform that knows only one of the two names."""
    return text.replace("Alice", "[NAME]")


def check_names(text: str, target: g.Datum) -> g.Assessment:
    """A verifier that is separate from the transform and checks the candidate it produced."""
    return g.Assessment(not any(name in text for name in NAMES), target)


def main() -> None:
    # Declare: two target texts and the relationships between them (none here).
    scope = g.GraphScope()
    alice = scope.datum("Alice uses alice@example.test")
    bob = scope.datum("Bob uses bob@example.test")
    graph = g.ProtectionGraph(scope, (alice, bob))

    # Compose: one transform, then one verifier over the transform's output answering the required check.
    inputs = {"text": "text", "target": "datum"}
    verifier = g.Operation.verifier("check-names", check_names, inputs, checks=("names_removed",))
    steps = (g.Operation("redact-alice", redact_alice), verifier)
    workflow = g.Workflow("redact-names", steps, requires=("names_removed",))

    # Prepare admits the workflow before any callback runs; protect runs it on this thread.
    prepared = g.prepare(graph, workflow, limits=g.Limits(max_targets=2))
    print(prepared.summary().render())
    result = g.protect(prepared)

    # Read the outcomes: execution terminal, qualification and release are separate facts.
    for name, datum in (("alice", alice), ("bob", bob)):
        outcome = result.outcome(datum)
        print(
            f"{name}: terminal={outcome.terminal} qualification={outcome.qualification} protected={outcome.protected}"
        )
    assert result.outcome(alice).protected and result.protected[alice] == "[NAME] uses alice@example.test"
    assert result.outcome(bob).qualification == "not_met" and bob not in result.protected
    assert result.outcome(bob).reasons == (g.Reason("failed", "names_removed", "check-names"),)

    summary = result.summary()
    print(summary.render())
    assert summary.protected == 1 and summary.consistent


if __name__ == "__main__":
    main()
