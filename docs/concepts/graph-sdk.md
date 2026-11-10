<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Graph SDK

`anonymizer.graph` runs a local protection workflow over text targets. You declare the targets and their relationships, compose operations into a workflow, admit it with `prepare`, and execute it with `protect`. A successful operation does not by itself make an output protected: release depends on a separate verifier's assessment of the candidate text.

The surface is provisional. Names, signatures and defaults may change before it is published, and the `Anonymizer` facade does not depend on that stability.

The [provider-free example](../examples/graph_sdk/local_workflows.py) uses only names exported by `anonymizer.graph`. Save the file and run it in an environment that has `nemo-anonymizer`:

```bash
python local_workflows.py
```

## Declare

A `GraphScope` creates `Datum` values, each an immutable text with an identity that belongs to that scope. A `ProtectionGraph` binds the target datums and the relations among them:

- `Mention` marks an exact half-open character interval of a target with a label.
- `ContextScope` lists the context a target may read: other datums, a `ContextView` over part of a datum, or a `ContextSource`, a caller-owned provider retrieved once per run with a declared item bound. A source's `kind` names a registered artifact kind (default `"text"`); items reach the operation with that type and an item that fails (or raises in) the kind's check or measure closes the retrieval as `invalid_item`. An operation reading structured context declares that kind on its `context` port, so a text consumer is rejected as `port_kind_mismatch`. A graph's context has one kind: mixing kinds anywhere in the graph is `unsupported_context_kind`. `prepare` rejects a kind mismatch, mixed kinds, an unknown kind, and a kind without a byte measure when a byte ceiling applies, before any provider runs.
- `Dependency` withholds one target's output when another target's output is withheld.
- `OutputGroup` releases its member targets all together or not at all.

Constructing these values has no effect. The graph constructor rejects foreign datums, out-of-range or overlapping mentions, duplicate context sources and dependency cycles with a `GraphRejection`.

## Compose

An `Operation` is a name, a callback, typed input and output ports, and a declared `Requests` bound (none, exact, upper bound, unknown or opaque) on the external requests it makes per target. Input ports bind by name to the latest earlier output, or to a per-target seed: `text` (the current candidate), `source` (the original text), `mentions`, `context` and `target`. `Operation.verifier` builds an operation whose callback returns an `Assessment` and names the `checks` that assessment answers; a `Workflow` names the checks it `requires` and is an ordered tuple of operations and nested workflows; a nested workflow's `requires` are required by the whole invocation. Callers that need an artifact kind beyond the built-in ones declare it with `register_kind`.

A phase-scoped operation, built with `Operation.phase_scoped` (scope `phase_scoped`), runs once over every target that is ready for it instead of once per target. It declares per-target port kinds as usual, but receives each input port as a tuple of per-target values in target order, and returns a sequence with one entry per target in that order. An entry is the target's output; `None` fails that target with `missing_output`; an `Exception` instance fails it with `raised`. If the callback raises, every target it was given fails with `raised`; if it returns something that is not a sequence of one entry per target, they all fail with `invalid_output` and the error is recorded. Targets that failed an earlier operation are not passed to it and stay blocked. Its declared `Requests` count once per invocation, not once per target. Mentions an operation returns on a `mentions` port are checked when accepted, by the same rules as declared mentions: each must be a `Mention` on its own target with valid offsets, matching claimed text and no overlap, or that target fails with the rule's reason (`invalid_output`, `mention_not_on_target`, `mention_out_of_range`, `mention_slice_mismatch`, `overlapping_output_ownership`) and other targets continue. Declared and accepted mentions count together against `Limits.max_mentions`; when one call's mentions would exceed it, every target of that call fails with `over_limit`. In Replace and Rewrite modes, detection runs as the first phase-scoped step of the same call that replaces or rewrites.

`Workflow.release` is `"protected"` by default. An `"annotation"` workflow returns content that keeps the source text and never qualifies as protected.

## Prepare and protect

`prepare(graph, workflow, limits=...)` checks the workflow against `Limits` and the port wiring before any callback runs, and returns a `PreparedProtection`. It raises `GraphRejection`, which carries a closed `code` and a `detail` naming the limit or rule, when the workflow is not admissible. A `request_budget` argument turns the declared request bounds into a hard ceiling; a workflow whose request count is unknown or opaque is rejected under a budget rather than run unbudgeted. `PreparedProtection.summary()` returns the planned operations and request count without running anything.

Only a verifier of the final candidate text answers a required check. A verifier of earlier text is a diagnostic: its assessment is readable with `value(..., producer=...)` but answers no requirement. To stop early, raise from an operation. A workflow that requires a check declared only by a verifier of earlier text, or by no verifier, is rejected at `prepare` with `GraphRejection("required_check_not_final")`, whose `detail` names the unanswered checks.

`protect(prepared, cancel=...)` runs the prepared workflow synchronously on the calling thread. A callback that raises closes its target as failed and does not raise out of `protect`. A `CancelToken` stops further dispatches and retrievals.

`protect` accepts only a `PreparedProtection` returned by `prepare` and not altered since; otherwise it raises `GraphRejection("not_prepared")`. It rechecks the limits and admitted capabilities before any callback runs. Every `Assessment` a verifier returns must name the target it was produced for; one without a target, or naming another target, makes the whole invocation inconsistent, so no target qualifies, while the assessments stay readable on the result.

## Read the results

`ProtectionResult` keeps three facts apart for each target in its `TargetOutcome`:

- `terminal`: `success`, `failed`, `cancelled` or `blocked`.
- `qualification`: `met` when every required check (including those of nested workflows) is answered by a verifier of the final candidate and every such verifier was satisfied, `not_met` otherwise, or `not_assessed` when the workflow requires no checks.
- `protected`: whether the output was released, which requires `met` plus every dependency and group the target belongs to.

`reasons` says why a target is not met or not released, one `Reason` per cause: a required check `failed`, `could_not_assess` (with the verifier's stated `cause`) or `missing` (a final-text verifier declares its assessment on a port not named `assessment`, so its verdict is not read), an execution code such as `raised`, `inconsistent_invocation`, or `withheld` (with `withheld_by` naming the targets that withheld it). `reason` is the last reason's code. `error` holds the exception that failed the target and is never rendered.

`result.protected` maps each released target to its text. `result.content()` gives each target's final candidate without a protection claim: the output of the last operation that produces text. If that operation did not succeed, the target has no final candidate and its outcome gives the reason; earlier text is never substituted. `result.value(target, port)` reads the latest value of a named port from any operation, and `result.value(target, port, producer=name)` reads only the output of the operation called `name`. `result.compose(parts)` joins the candidate texts of the listed targets in the caller's order; `result.reconstruct(parts, source)` joins them in the order of the positions `source` maps them to, and rejects a part outside `source`, two parts at one position or an untaken position. Either rejects a listed target that did not succeed with an available candidate as `unavailable_input`. Projections read the saved result and never rerun a callback. `result.summary()` returns an `InspectionSummary` of counts and reason codes with no text content; `render()` prints it. `result.ledger()` gives one status per target and operation; an operation that never ran is `blocked` when its target had already failed, and `cancelled` when the invocation was cancelled first.

The example transforms two targets, one containing `Alice` and one containing `Bob`, with a transform that knows only `Alice`. The verifier answers the required `names_removed` check for both names, so the first target is `met` and released and the second is `not_met` and withheld. The email address in the released text is untouched: `met` applies to the declared name check, not to general anonymization quality.

The `Anonymizer` facade keeps the same outcomes. `result.outcome(row)` on the `AnonymizerResult` or `PreviewResult` of `run()` or `preview()` returns the `TargetOutcome` of that output row. In Replace mode `not_met` means the row's replacements did not all land (the `replacements_applied` check failed, for example because the replacement map omitted a detected entity); in Rewrite mode it means a final privacy check on the rewritten text failed. Either way the row is still in the output frame, so read its outcome before releasing it. A row index outside the result raises `IndexError`, as does a result from a custom runner, which carries no outcomes. On an `evaluate()` result, `outcome(row)` reports the judges and is never protected: its `reasons` hold one `passed`, `scored`, `could_not_assess` or `failed` reason per judge check, in that order (worst last), then any execution error such as `raised`, so `reason` names an execution error, else a failing or unassessed check, when there is one (`check` names it; `cause` gives the failure reason, or `judge_unavailable` when the judge gave no score). `scored` means the rewrite judge, which has no pass threshold, returned scores; read them in the judge evaluation column. The evaluated result keeps its own outcomes and `needs_human_review`.
