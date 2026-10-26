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
- `ContextScope` lists the context a target may read: other datums, a `ContextView` over part of a datum, or a `ContextSource`, a caller-owned provider retrieved once per run with a declared item bound.
- `Dependency` withholds one target's output when another target's output is withheld.
- `OutputGroup` releases its member targets all together or not at all.

Constructing these values has no effect. The graph constructor rejects foreign datums, out-of-range or overlapping mentions, duplicate context sources and dependency cycles with a `GraphRejection`.

## Compose

An `Operation` is a name, a callback, typed input and output ports, and a declared `Requests` bound (none, exact, upper bound, unknown or opaque) on the external requests it makes per target. Input ports bind by name to the latest earlier output, or to a per-target seed: `text` (the current candidate), `source` (the original text), `mentions`, `context` and `target`. `Operation.verifier` builds an operation whose callback returns an `Assessment`; a `Workflow` is an ordered tuple of operations and nested workflows. Callers that need an artifact kind beyond the built-in ones declare it with `register_kind`.

An invocation-scoped operation, built with `Operation.per_invocation`, runs once over every target that is ready for it instead of once per target. It declares per-target port kinds as usual, but receives each input port as a tuple of per-target values in target order, and returns a sequence with one entry per target in that order. An entry is the target's output; `None` fails that target with `missing_output`; an `Exception` instance fails it with `raised`. If the callback raises, every target it was given fails with `raised`; if it returns something that is not a sequence of one entry per target, they all fail with `invalid_output` and the error is recorded. Targets that failed an earlier operation are not passed to it and stay blocked. Its declared `Requests` count once per invocation, not once per target. In Replace mode, detection runs as the first invocation-scoped step of the same call that replaces.

`Workflow.release` is `"protected"` by default. An `"annotation"` workflow returns content that keeps the source text and never qualifies as protected.

## Prepare and protect

`prepare(graph, workflow, limits=...)` checks the workflow against `Limits` and the port wiring before any callback runs, and returns a `PreparedProtection`. It raises `GraphRejection`, which carries a closed `code`, when the workflow is not admissible. A `request_budget` argument turns the declared request bounds into a hard ceiling; a workflow whose request count is unknown or opaque is rejected under a budget rather than run unbudgeted. `PreparedProtection.summary()` returns the planned operations and request count without running anything.

`protect(prepared, cancel=...)` runs the prepared workflow synchronously on the calling thread. A callback that raises closes its target as failed and does not raise out of `protect`. A `CancelToken` stops further dispatches and retrievals.

## Read the results

`ProtectionResult` keeps three facts apart for each target in its `TargetOutcome`:

- `terminal`: `success`, `failed`, `cancelled` or `blocked`.
- `qualification`: `met` when every verifier covering the final candidate was satisfied, `not_met`, or `not_assessed` when no verifier covers it.
- `protected`: whether the output was released, which requires `met` plus every dependency and group the target belongs to.

`result.protected` maps each released target to its text. `result.content()` gives each target's final candidate text without a protection claim, and `result.value(target, port)` reads a named port. Projections read the saved result and never rerun a callback. `result.summary()` returns an `InspectionSummary` of counts and reason codes with no text content; `render()` prints it.

The example transforms two targets, one containing `Alice` and one containing `Bob`, with a transform that knows only `Alice`. The verifier checks both names, so the first target is `met` and released and the second is `not_met` and withheld. The email address in the released text is untouched: `met` applies to the declared name check, not to general anonymization quality.
