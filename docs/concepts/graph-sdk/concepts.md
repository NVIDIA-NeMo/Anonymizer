<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Graph SDK concepts

After reading this page you can explain what each of the four stages (declare, compose, prepare, protect) does,
why a workflow's checks are kept apart from its steps, and how the SDK decides whether to release a text. Terms in
bold are defined where they first appear and again in the [glossary](glossary.md).

!!! warning "Provisional"

    The graph SDK is provisional. Names, signatures and default values may change in a later release.

## The picture

```text
 Declare                Compose                 Prepare              Protect               Read
 -------                -------                 -------              -------               ----
 GraphScope             Operation (step)        prepare(graph,       protect(prepared)     ProtectionResult
   .datum(text)  ---->  Operation.verifier ---> workflow, limits) -->  runs each step  -->   .outcome(target)
 ProtectionGraph        Workflow(requires=)     checks everything    per target, or        .protected
   targets, relations                           before effects       once per batch        .summary()
```

You describe the data, then the work, then ask the SDK to check the description, then run it. Nothing runs until
`protect`. Everything you build before that is an immutable value: constructing it calls none of your functions.

## Declare: targets and their relations

A **datum** (`Datum`) is one text with an identity. You make datums from a **scope** (`GraphScope`), which is the
namespace the identities belong to. Two datums with the same text are still two different datums: equality uses the
identity, never the text.

A **target** is a datum the workflow processes and may release. The **protection graph** (`ProtectionGraph`) holds
the targets and the relations among them:

| Relation | What it says | Guide |
|---|---|---|
| `Mention` | This span of a target holds a value of this label, such as a name. | [Write your own detector](guide-custom-detector.md) |
| `ContextScope` | While processing this target, a step may also read this other text. Context is read, never released. | [Give an operation context](guide-give-an-operation-context.md) |
| `Dependency` | Withhold this target if that one is withheld. | [Release related texts together](guide-release-together.md) |
| `OutputGroup` | Release these targets all together or not at all. | [Release related texts together](guide-release-together.md) |
| `Record` | These targets belong together, such as the fields of one table row. | [Release related texts together](guide-release-together.md) |

The graph constructor validates the relations at once: a mention outside its target, a dependency cycle, or a
datum from another scope raises `GraphRejection` immediately.

## Compose: operations, ports and workflows

An **operation** (`Operation`) is one step: a name, your function, and its **ports**. An input port is a parameter
of your function; an output port is a value it returns. Each port has a **kind**, a registered type name such as
`"text"` or `"datum"`, which the SDK uses to check that each input receives the kind of value its step expects.

```python
from anonymizer import graph as g

step = g.Operation("lowercase", lambda text: text.lower(), inputs={"text": "text"}, outputs={"text": "text"})
print(step.inputs, step.outputs, step.scope)
```

An input port is filled from the most recent earlier step that has an output port of the same name. If no earlier
step has one, it is filled from a **seed**: a value the SDK supplies for every target.

| Seed | Kind | Value for each target |
|---|---|---|
| `text` | `text` | The current candidate: the target's text, or the latest text a step returned |
| `source` | `text` | The target's original text |
| `mentions` | `mentions` | The mentions declared on the graph for this target |
| `context` | `context` | The context items this target may read |
| `target` | `datum` | The target itself |
| `session` | `session-view` | A read-only view of the run's Session (see [Session](reference.md#session-provisional)) |
| `identities` | `identities` | The identity declarations on this target |
| `record` | `record` | The record this target belongs to |

The **candidate** is the text a step produces for a target. Each step with an output port named `text` makes a new
candidate. The **final candidate** is the text returned by the last step that outputs `text`; it is what the SDK
releases if the target qualifies.

A **workflow** (`Workflow`) is an ordered tuple of operations, and may contain nested workflows. By default an
operation runs once per target. A **phase-scoped** operation runs once for all targets that reached it, which lets
one model call serve a whole batch ([guide](guide-batch-targets.md)).

## Verifiers and checks: why they are separate

A **verifier** is an operation whose function returns an **assessment** (`Assessment`): a verdict about one target,
`satisfied=True`, `False`, or `None` when it could not decide. A **check** is a name, such as `"names_removed"`,
that a verifier says its verdict answers. A workflow **requires** checks; a target **qualifies** only when every
required check was answered on its final candidate and passed.

The SDK keeps the step that changes the text apart from the verifier that judges it. A step that ran without error
has only shown that it ran. Whether its output is safe is a separate question, asked by code that can be written,
reviewed and replaced on its own. This is the core rule of the SDK: **release depends on a separate verifier's
assessment of the final candidate**, never on a step's success.

Two details follow from it:

- A verifier names its **subject**, the input ports its verdict is about, such as `subject=("text",)`. `prepare`
  uses the subject to make sure a required check is made on the text that would be released. A check of an earlier
  draft does not count, because a later step may have changed the text since. `prepare` refuses such a workflow
  with `required_check_not_final`.
- `prepare` trusts the subject you declare. It does not inspect your function. A verifier that declares `text` as
  its subject but bases its verdict on something else is still accepted.

A verifier of the final candidate may also answer checks the workflow does not require. Those verdicts become
**advisories**: reported on the outcome, never changing qualification or release. See
[Add a verifier and require a check](guide-verify-and-require-checks.md).

## Prepare: validation before effects

`prepare(graph, workflow, limits=..., request_budget=...)` validates everything that can be known before running:

- every input port can be filled, with the kind its step declares;
- every required check is answered by a verifier of the final candidate;
- operation names are unique (a nested step is named `inner/step`);
- the graph fits the **limits** (`Limits`): number of targets, operations, mentions, context items and optional byte
  ceilings;
- the **planned requests**, the sum of what each step declares it will send to external services, fit the request
  budget if you gave one.

If anything fails, `prepare` raises `GraphRejection` with a `code` and a `detail`, and none of your functions has
run. Otherwise it returns a `PreparedProtection`, the only value `protect` accepts.

## Protect: running the workflow

`protect(prepared)` runs the steps in order on the calling thread. A per-target step runs once for each target that
is still going; a phase-scoped step runs once for all of them. When a function raises, its target fails with the
reason `raised` and later steps skip that target; `protect` itself does not raise. A `CancelToken` stops it between
calls.

## Outcomes: three separate facts

For each target, `result.outcome(target)` returns a `TargetOutcome`. It keeps three facts apart:

1. **Terminal** (`terminal`): did the steps run? `success`, `failed`, `cancelled` or `blocked`.
2. **Qualification** (`qualification`): did the required checks pass on the final candidate? `met`, `not_met`, or
   `not_assessed` when the workflow requires no checks.
3. **Release** (`protected`): was the output released? This needs `met`, a workflow that releases output, and no
   withheld dependency or output group member.

A target that is not released is **withheld**. Its `reasons` say why, one `Reason` per cause. The result also
offers the final candidate of every target without any release claim (`result.content()`), for inspection and
debugging.

The SDK decides each target in this order:

```text
 terminal = success? ------------------------------------ no --> not_met, withheld (reason: raised, ...)
      | yes
 workflow requires at least one check? ------------------ no --> not_assessed, withheld (no reason)
      | yes
 every required check passed on the final candidate? ---- no --> not_met, withheld (reason: failed, ...)
      | yes
 every assessment in the run names the target it judged?  no --> every target in the run:
      | yes                                                       not_met, withheld (inconsistent_invocation)
 met
      |
 workflow has release="annotation"? --------------------- yes -> met, withheld (reason: annotation)
      | no
 a dependency or group member withheld? ----------------- yes -> met, withheld (reason: withheld)
      | no
 protected = True
```

The same cases as a table:

| Case | `terminal` | `qualification` | `protected` | `reasons` |
|---|---|---|---|---|
| Steps ran and every required check passed | `success` | `met` | `True` | none |
| A required check failed | `success` | `not_met` | `False` | `failed` |
| A step raised | `failed` | `not_met` | `False` | `raised` |
| The workflow has an empty `requires` | `success` | `not_assessed` | `False` | none |
| An annotation workflow, check passed | `success` | `met` | `False` | `annotation` |
| A verifier returned an assessment naming another target | `success` | `not_met`, for every target in the run | `False` | `inconsistent_invocation` |
| This target passed; another member of its output group failed | `success` | `met` | `False` | `withheld` |

Two rows need care. A workflow with no required checks makes no protection claim, so nothing it produces is
released, even when every step succeeds. And attribution covers the whole run: if any verifier, for any target,
returns an assessment that names a different target or no target, the SDK can no longer tell which verdict belongs
to which text. No target in the run then qualifies, and `result.consistent` is `False`.
This answer overrides every other one in the diagram: every target is `not_met` with `inconsistent_invocation` as
its last reason, even a target whose step raised and even when the workflow requires no checks.

A workflow with `release="annotation"` never releases anything. It is for measuring or labelling text without a
protection claim.

## What the SDK does not do

- It does not decide what "protected" means. Your required checks do.
- It does not call models or services. Your functions do, and they declare how many requests they make.
- It does not run steps in parallel or asynchronously.
- It does not keep state between calls to `protect`.
