<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Graph SDK

This page is about building a custom protection workflow, for the cases where the normal Anonymizer flow does not
fit. It is not a tutorial on running Anonymizer. After reading it you can run a small custom workflow, read its
result, and know which page to open next.

!!! warning "For advanced use"

    The graph SDK is for advanced users who need to change how protection works in a way the `Anonymizer` class does
    not allow. If you are new to Anonymizer, start with [Get Started](../index.md#get-started) and
    [Choosing a strategy](choosing-a-strategy.md) instead.

!!! warning "Provisional"

    The graph SDK is provisional. Its names, signatures and default values may change in a later release. The
    `Anonymizer` class does not depend on them staying the same.

## What the graph SDK is for

The `Anonymizer` class runs fixed strategies: Substitute, Redact, Annotate, Hash, Rewrite. The graph SDK, in the module
`anonymizer.graph`, lets you build your own strategy from Python functions and run it locally:

- Write each step as an ordinary function: a regular expression, a call to your own model, a lookup.
- Add a separate check that decides whether the output is good enough to release.
- Get back, for each text, whether the steps ran, whether the check passed, and whether the output may be
  released, as three separate answers.

Use it when the built-in strategies do not fit: you want your own detector, a different rewrite loop, a field-by-field
policy for table rows, or an audit of text that was anonymized elsewhere. If a built-in strategy does what you need,
use the `Anonymizer` class instead.

The SDK runs your functions synchronously, on the thread that calls it. It calls no model and no service by itself.
Every external call happens inside a function you wrote.

## A first program

This program protects two short texts. The first, "Alice uses alice@example.test", contains the name Alice. The
second, "Bob uses bob@example.test", contains the name Bob. There are two names to protect, but the only step
replaces Alice and knows nothing about Bob. A separate check then looks for both names in each text. The first text
no longer contains either name, so it passes and is released. The second still contains Bob, so it fails the check
and is withheld.

```python
from anonymizer import graph as g

NAMES = ("Alice", "Bob")


def redact_alice(text: str) -> str:
    """A step that knows only one of the two names."""
    return text.replace("Alice", "[NAME]")


def check_names(text: str, target: g.Datum) -> g.Assessment:
    """A check that is separate from the step and looks at the step's output."""
    return g.Assessment(not any(name in text for name in NAMES), target)


# 1. Declare the texts to protect.
scope = g.GraphScope()
alice = scope.datum("Alice uses alice@example.test")
bob = scope.datum("Bob uses bob@example.test")
graph = g.ProtectionGraph(scope, (alice, bob))

# 2. Compose the steps into a workflow.
inputs = {"text": "text", "target": "datum"}
verifier = g.Operation.verifier("check-names", check_names, inputs, checks=("names_removed",), subject=("text",))
steps = (g.Operation("redact-alice", redact_alice), verifier)
workflow = g.Workflow("redact-names", steps, requires=("names_removed",))

# 3. Prepare (validate) the workflow, then run it.
prepared = g.prepare(graph, workflow, limits=g.Limits(max_targets=2))
print(prepared.summary().render())
result = g.protect(prepared)

# 4. Read the outcomes.
for name, datum in (("alice", alice), ("bob", bob)):
    outcome = result.outcome(datum)
    print(f"{name}: terminal={outcome.terminal} qualification={outcome.qualification} protected={outcome.protected}")
print(result.protected[alice])
print(result.outcome(bob).reasons)
print(result.summary().render())
```

It prints:

```text
workflow: 2 operations [redact-alice, check-names]; targets: 2
planned external requests: exact 0
terminals: none (not executed)
alice: terminal=success qualification=met protected=True
bob: terminal=success qualification=not_met protected=False
[NAME] uses alice@example.test
(Reason(code='failed', check='names_removed', verifier='check-names', cause=''),)
workflow: 2 operations [redact-alice, check-names]; targets: 2
planned external requests: exact 0
terminals: success=2; qualification: met=1, not_met=1
reasons: failed=1; protected: 1; consistent: True
observed: binding=exact 0, correction=exact 0, retry=exact 0, total=exact 0, requests=exact 0; budget: within
```

The repository ships the same program, with `assert` statements on its result, as
[`examples/graph_sdk/local_workflows.py`](https://github.com/NVIDIA-NeMo/Anonymizer/blob/main/examples/graph_sdk/local_workflows.py).
It needs no model and no network.

## What each part does

**Declare.** A `GraphScope` makes `Datum` values. A datum is one text plus an identity. Each datum you pass to
`ProtectionGraph` is a **target**: a text the workflow protects and may release. Here there are two targets and no
relations between them.

**Compose.** An `Operation` wraps one of your functions. `Operation("redact-alice", redact_alice)` reads the current
text and returns new text. `Operation.verifier(...)` wraps a function that returns an `Assessment`, a verdict about
one target. The verifier says which **check** its verdict answers (`checks=("names_removed",)`) and which input it
judged (`subject=("text",)`). The `Workflow` lists the steps in order and names the checks a target must pass
(`requires=("names_removed",)`).

Each function receives its inputs as keyword arguments named after its input ports. `check_names` asks for `text`
(the current text) and `target` (the datum being processed), so the verifier declares
`{"text": "text", "target": "datum"}`: port name on the left, kind of value on the right.

**Prepare.** `prepare` checks the whole workflow before any of your functions runs: that every input can be
supplied, that every required check is answered by a verifier of the final text, and that the workflow fits the
limits you set. It returns a `PreparedProtection` or raises `GraphRejection`. The first summary shows the plan: two
operations, two targets, no external requests declared.

**Protect.** `protect` runs the prepared workflow: it calls `redact_alice` for each target, then `check_names` for
each target.

**Read.** Each target has a `TargetOutcome` with three separate facts:

| Fact | Question it answers | Alice | Bob |
|---|---|---|---|
| `terminal` | Did every step run? | `success` | `success` |
| `qualification` | Did every required check pass on the final text? | `met` | `not_met` |
| `protected` | Was the output released? | `True` | `False` |

Bob's text ran without error but still contains "Bob", so the check failed and the output was withheld. A step that
succeeds is not enough for release. Alice's text was released because Alice passed the declared check: a
case-sensitive search for the names "Alice" and "Bob". `result.protected` maps released targets to their text, so
it holds Alice's text only. `reasons` says why Bob was not released.

The released text shows the scope of that check: it still contains the email address `alice@example.test`. `met`
means the declared checks passed, nothing more. The checks you write define what "protected" means for your
workflow.

## Where to go next

- [Concepts](graph-sdk/concepts.md): the model behind the four stages, and why the check is separate from the steps.
- [Glossary](graph-sdk/glossary.md): every term used in these pages, defined once.
- Task guides:
  - [Write an operation](graph-sdk/guide-write-an-operation.md)
  - [Add a verifier and require a check](graph-sdk/guide-verify-and-require-checks.md)
  - [Process all targets in one call](graph-sdk/guide-batch-targets.md)
  - [Give an operation context](graph-sdk/guide-give-an-operation-context.md)
  - [Write your own detector](graph-sdk/guide-custom-detector.md)
  - [Release related texts together](graph-sdk/guide-release-together.md)
  - [Set limits, budget requests and cancel](graph-sdk/guide-limits-budgets-cancel.md)
  - [Rebuild a document from protected parts](graph-sdk/guide-rebuild-documents.md)
  - [Find out why a target was withheld](graph-sdk/guide-read-outcomes.md)
- [Ideas](graph-sdk/ideas.md): workflows the SDK makes possible beyond the built-in strategies.
- [Reference](graph-sdk/reference.md): every public name, field, limit and error code.

## If you only use the `Anonymizer` class

You do not need this SDK to read a verdict. `run()` and `preview()` results carry one `TargetOutcome` per output
row, the same type described here: `result.outcome(row)`. A row stays in the output frame even when it is not
released, so read its outcome before you use the row. What `not_met` means depends on the mode:

- [Replace](replace.md#reading-each-rows-outcome): the row's replacements did not all land.
- [Rewrite](rewrite.md#reading-each-rows-outcome): a final check on the rewritten text failed or could not be assessed.
- [Evaluation](evaluation.md#outcomes-after-evaluate): what the judges of `evaluate()` add to each row's outcome.
