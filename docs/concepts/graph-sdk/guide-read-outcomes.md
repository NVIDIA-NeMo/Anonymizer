<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Find out why a target was withheld

After reading this page you can read a result for one target or for a whole run, tell a failed step from a failed
check from a withholding, and see which step ran for which target.

!!! warning "Provisional"

    The graph SDK is provisional. Names, signatures and default values may change in a later release.

## One program, three kinds of failure

Alice's text passes. Bob's text runs but fails the check. Carol's text makes the step raise.

```python
from anonymizer import graph as g

NAMES = ("Alice", "Bob")


def redact_alice(text: str) -> str:
    if "Carol" in text:
        raise ValueError("cannot parse")
    return text.replace("Alice", "[NAME]")


def check_names(text: str, target: g.Datum) -> g.Assessment:
    return g.Assessment(not any(n in text for n in NAMES), target)


scope = g.GraphScope()
people = {name: scope.datum(f"{name} here.") for name in ("Alice", "Bob", "Carol")}
graph = g.ProtectionGraph(scope, tuple(people.values()))
workflow = g.Workflow(
    "names",
    (
        g.Operation("redact", redact_alice),
        g.Operation.verifier(
            "check", check_names, {"text": "text", "target": "datum"}, checks=("names_removed",), subject=("text",)
        ),
    ),
    requires=("names_removed",),
)
result = g.protect(g.prepare(graph, workflow))
for name, datum in people.items():
    o = result.outcome(datum)
    print(name, o.terminal, o.qualification, o.available, o.protected, [r.code for r in o.reasons])
```

```text
Alice success met True True []
Bob success not_met True False ['failed']
Carol failed not_met False False ['raised']
```

## The fields of a `TargetOutcome`

| Field | Meaning |
|---|---|
| `terminal` | `success`, `failed`, `cancelled` or `blocked`: did every step run. |
| `qualification` | `met`, `not_met`, or `not_assessed` when the workflow requires no checks. A target that did not succeed is `not_met`. |
| `available` | Whether the target has a final candidate, that is, the last step that outputs `text` produced one. |
| `protected` | Whether the final candidate was released. |
| `reasons` | One `Reason` per cause of not qualifying or not being released. Empty for a released target. |
| `reason` | The code of the last entry in `reasons`, or `""`. |
| `withheld_by` | The targets whose withholding withheld this one, through a dependency or an output group. |
| `advisories` | One `Advisory` per check a final-text verifier answered that the workflow does not require. |
| `error` | The exception that failed the target, if any. Never printed in summaries. |

A `Reason` has `code`, `check`, `verifier` and `cause`. The codes:

| Code | Meaning |
|---|---|
| `failed` | A required check returned `False`. `check` and `verifier` name it; `cause` holds the verifier's reason. |
| `could_not_assess` | A required check returned `None`. `cause` holds the verifier's reason. |
| `missing` | A required check has no assessment from a verifier of the final candidate for this target. `prepare` and the verifier construction rules are designed to prevent this, so it is not expected in normal use. |
| `raised` | A step raised, or a phase-scoped step returned an exception for this target. |
| `missing_output` | A phase-scoped step returned `None` for this target. |
| `invalid_output` | A step returned a value of the wrong kind or the wrong port names. |
| `mention_not_on_target`, `mention_out_of_range`, `mention_slice_mismatch`, `mention_unlabelled`, `overlapping_output_ownership` | A detector's mentions broke a rule ([guide](guide-custom-detector.md)). |
| `over_limit` | Mentions returned in one call passed `Limits.max_mentions`. |
| `binding_failed` | A context source this target reads did not return usable items ([guide](guide-give-an-operation-context.md)). |
| `cancelled` | The run was cancelled before this target finished. |
| `inconsistent_invocation` | Some verifier in the run returned an assessment naming another target, or no target, so no target in the run qualifies. |
| `withheld` | The target is `met` but a dependency or group member was withheld. See `withheld_by`. |
| `annotation` | The target is `met` but the workflow is an annotation workflow, which never releases. |

## What ran for each target

`result.ledger()` gives one `(target, operation, status)` entry for every target and step, whether the step ran or
not. A step that never ran is `blocked` if its target had already failed, and `cancelled` if the run was cancelled
first. `result.attempts` lists only the calls that happened, one entry per target a step was called for, in the
order they ran, each with `target`, `operation`, `status` and `reason`. Steps run in workflow order, and each step
runs for every target before the next step starts.

```python
from anonymizer import graph as g


def redact(text: str) -> str:
    if "Carol" in text:
        raise ValueError("cannot parse")
    return text


check = g.Operation.verifier(
    "check",
    lambda text, target: g.Assessment(True, target),
    {"text": "text", "target": "datum"},
    checks=("ok",),
    subject=("text",),
)
scope = g.GraphScope()
alice, carol = scope.datum("Alice here."), scope.datum("Carol here.")
result = g.protect(
    g.prepare(
        g.ProtectionGraph(scope, (alice, carol)),
        g.Workflow("w", (g.Operation("redact", redact), check), requires=("ok",)),
    )
)
names = {alice: "alice", carol: "carol"}
for target, operation, status in result.ledger():
    print(names[target], operation, status)
print([(names[a.target], a.operation, a.status, a.reason) for a in result.attempts])
```

```text
alice redact succeeded
alice check succeeded
carol redact failed
carol check blocked
[('alice', 'redact', 'succeeded', ''), ('carol', 'redact', 'failed', 'raised'), ('alice', 'check', 'succeeded', '')]
```

The ledger uses step statuses (`succeeded`, `failed`, `blocked`, `cancelled`), which differ from the target
terminal values (`success`, ...).

## A summary of the whole run

`result.summary()` returns an `InspectionSummary`: counts and reason codes, never text, so it is safe to log.
`render()` formats it:

```text
workflow: 2 operations [redact, check]; targets: 3
planned external requests: exact 0
terminals: failed=1, success=2; qualification: met=1, not_met=2
reasons: failed=1, raised=1; protected: 1; consistent: True
observed: binding=exact 0, correction=exact 0, retry=exact 0, total=exact 0, requests=exact 0; budget: within
```

That is the summary of the first program on this page. Its fields are `operations`, `targets`, `planned_requests`,
`terminals`, `qualifications`, `reasons` (pairs of code and count, counting each target's `reason`, the code of its
last `Reason`), `observed`, `protected` (a count), `consistent` and `budget`. `PreparedProtection.summary()` gives
the same type before the run, with only the first three filled. The `observed` row and `budget` are described in
[Set limits, budget requests and cancel](guide-limits-budgets-cancel.md#cap-the-plan-with-a-budget).

## Reading text

| Read | Gives | Protection claim |
|---|---|---|
| `result.protected` | A dict of released targets to their final candidate | Yes |
| `result.content()` | Each target with its final candidate, or `None` when it has none | No |
| `result.value(target, port)` | The latest value of any port for a target | No |
| `result.value(target, port, producer=name)` | That port's value from the named step only | No |

Release only what `result.protected` holds. The other reads are for inspection and debugging.
