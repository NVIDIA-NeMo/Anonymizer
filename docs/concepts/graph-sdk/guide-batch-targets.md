<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Process all targets in one call

After reading this page you can make one model call serve every target instead of one call per target, declare the
requests that call makes, and handle per-target failures inside a batch.

!!! warning "Provisional"

    The graph SDK is provisional. Names, signatures and default values may change in a later release.

## Why batch

A per-target operation calls your function once for each target. With a model behind it, 1,000 targets mean 1,000
requests. A **phase-scoped** operation calls your function once with the values of every target that reached it,
so you can send them in one request, or split them however your client prefers.

## A phase-scoped operation

Build it with `Operation.phase_scoped`. Your function receives each input port as a tuple, one entry per target in
target order, and returns a sequence with one entry per target in the same order. The example uses a stand-in
client in place of a model, so it runs without a network:

```python
from anonymizer import graph as g


class StandInClient:
    """Stands in for a model client: one call takes many texts."""

    def __init__(self) -> None:
        self.calls = 0

    def redact_many(self, texts: list[str]) -> list[str]:
        self.calls += 1
        return [t.replace("Alice", "[NAME]").replace("Bob", "[NAME]") for t in texts]


client = StandInClient()


def redact_all(text: tuple[str, ...]) -> list[str]:
    return client.redact_many(list(text))


def check(text: str, target: g.Datum) -> g.Assessment:
    return g.Assessment("Alice" not in text and "Bob" not in text, target)


scope = g.GraphScope()
targets = tuple(scope.datum(t) for t in ("Alice called.", "Bob replied.", "Nobody else wrote."))
graph = g.ProtectionGraph(scope, targets)

workflow = g.Workflow(
    "batch",
    (
        g.Operation.phase_scoped("redact-all", redact_all, requests=g.Requests.exact(1)),
        g.Operation.verifier("check", check, {"text": "text", "target": "datum"}, checks=("clean",), subject=("text",)),
    ),
    requires=("clean",),
)
prepared = g.prepare(graph, workflow, request_budget=5)
print(prepared.summary().planned_requests)
result = g.protect(prepared)
print(client.calls, [result.protected[t] for t in targets])
```

```text
exact 1
1 ['[NAME] called.', '[NAME] replied.', 'Nobody else wrote.']
```

The client was called once for three targets. The verifier stays per target, so each target gets its own verdict.
A verifier can also be phase-scoped; build it with the `Operation` constructor
(`outputs={"assessment": "assessment"}, scope="phase_scoped"`) and return one `Assessment` per target.

## Declare the requests

`requests=g.Requests.exact(1)` tells `prepare` that the step makes one external request per call. For a
phase-scoped step that is one per run; for a per-target step it is one per target. `prepare` adds them up into
the **planned requests**, and with `request_budget=` refuses a workflow that could exceed the budget:

```python
from anonymizer import graph as g

scope = g.GraphScope()
graph = g.ProtectionGraph(scope, tuple(scope.datum(t) for t in ("a", "b", "c")))
per_target = g.Workflow("per-target", (g.Operation("call", lambda text: text, requests=g.Requests.exact(1)),))
print(g.prepare(graph, per_target).summary().planned_requests)
try:
    g.prepare(graph, per_target, request_budget=2)
except g.GraphRejection as exc:
    print(exc.code)
```

```text
exact 3
over_request_budget
```

The budget is checked once, at `prepare`. It is not enforced while the workflow runs. A workflow with a step that
declares `Requests.unknown()` or `Requests.opaque()` cannot be checked against a budget, so `prepare` refuses it
with `hard_budget_without_visibility` when you pass `request_budget`. See
[Set limits, budget requests and cancel](guide-limits-budgets-cancel.md) for the other declarations.

## Per-target results inside a batch

Each entry your function returns decides one target:

| Entry | Effect on that target |
|---|---|
| A value of the output's kind | Accepted as the target's output |
| `None` | Fails with `missing_output` |
| An `Exception` instance (returned, not raised) | Fails with `raised` |

```python
from anonymizer import graph as g


def partial(text: tuple[str, ...]) -> list:
    return [text[0].upper(), None, ValueError("model refused")]


scope = g.GraphScope()
graph = g.ProtectionGraph(scope, tuple(scope.datum(t) for t in ("one", "two", "three")))
result = g.protect(g.prepare(graph, g.Workflow("p", (g.Operation.phase_scoped("partial", partial),))))
print([(o.terminal, o.reason) for o in result.outcomes])
```

```text
[('success', ''), ('failed', 'missing_output'), ('failed', 'raised')]
```

If the function raises, every target in the batch fails with `raised`. If it returns something that is not a
sequence of exactly one entry per target, every target fails with `invalid_output`.

## Which targets a batch receives

A phase-scoped step receives only the targets still going: a target that failed an earlier step is not passed in,
and the ledger shows the batch step as `blocked` for it.

```python
from anonymizer import graph as g

sizes = []


def fail_bob(text: str) -> str:
    if "Bob" in text:
        raise RuntimeError("no")
    return text


def batch(text: tuple[str, ...]) -> list[str]:
    sizes.append(len(text))
    return list(text)


scope = g.GraphScope()
graph = g.ProtectionGraph(scope, tuple(scope.datum(t) for t in ("Alice", "Bob", "Carol")))
workflow = g.Workflow("p", (g.Operation("fail-bob", fail_bob), g.Operation.phase_scoped("batch", batch)))
result = g.protect(g.prepare(graph, workflow))
print(sizes, [o.terminal for o in result.outcomes])
```

```text
[2] ['success', 'failed', 'success']
```

To keep the members of one record together inside a batch (for example, the fields of one row in one prompt), read
the `record` seed: it arrives as a tuple of `Record` values parallel to the texts, equal for targets of the same
record. See [Release related texts together](guide-release-together.md).
