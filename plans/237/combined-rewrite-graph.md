<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Combined Rewrite Integration

Tracks the proof of concept for [GitHub issue #237](https://github.com/NVIDIA-NeMo/Anonymizer/issues/237).
It extends the rewrite portion of [Anonymizer Workflow Columns](../custom-column-plugins/anonymizer-workflow-columns.md).

## Goal

Replace the Python-controlled post-detection rewrite loop with one DataDesigner
execution:

```text
replacement map
  -> domain + disposition + QA + initial rewrite
  -> evaluate 0
  -> repair 0 when evaluate 0 fails
  -> evaluate 1 when repair 0 ran
  -> ...
  -> coalesce the last executed evaluation state
```

The repair count is static at graph-build time. Each configured round gets unique
columns and uses `SkipConfig` to bypass repair and downstream re-evaluation for
rows that already pass. This PR keeps the legacy workflow as the default. After
[Data Designer #861](https://github.com/NVIDIA-NeMo/DataDesigner/pull/861) is
merged and released, a small follow-up can consume its terminal-failure API,
make the combined graph the default, and retain legacy as an explicit fallback.

## Current Status

`CombinedRewriteWorkflow` remains an opt-in `RewriteWorkflow` subclass while the
legacy path serves as the default and parity oracle. This avoids losing precise
failure-stage attribution before Data Designer #861 is available in a supported
release and Anonymizer maps its failed column and seed-row position back to the
existing `FailedRecord` contract.

The graph currently:

- generates and filters the replacement map in the same execution;
- reuses the existing domain, disposition, QA, rewrite, evaluator, and repair helpers;
- supports `max_repair_iterations >= 0` by statically unrolling rounds;
- preserves no-entity passthrough outside DataDesigner;
- restores the existing final rewrite, metric, repair-count, and review columns;
- leaves the separate `evaluate()` judge path unchanged.

With two repair rounds, the graph has 36 columns and DataDesigner 0.8 validates it
without duplicate producers, missing dependencies, or cycles.

Tests execute real Data Designer conditional scheduling for mixed rows requiring
zero, one, two, and more-than-allowed repairs. They also cover no-entity
passthrough, mixed-row ordering, final state selection, repair counts, exhausted
review flags, malformed initial rewrites, coarse combined-boundary failures, and
graph validation with up to ten repair rounds. Local 64-row batches cover both
mostly-skipped and mostly-repaired scheduling.

## Expected Execution Change

For a rewrite run with entity rows:

| Path | Base DD runs | DD runs per repair round |
|---|---:|---:|
| Current full pipeline | 5 | 2 |
| Proof of concept | 3 | 0 |

The totals include the two existing detection runs. This proof combines only the
post-detection rewrite work, so a full detection-plus-rewrite graph remains a later step.

## Benchmark Result

The authoritative controlled run completed 12 pairs without workload failures.
Both paths received the same prepared and initially evaluated state. Ten rows
required one repair and two skipped repair in both paths; none exhausted the
three configured rounds. Both paths made 80 LLM requests, while the measured
rewrite/evaluate portion used four Data Designer workflows for legacy and two
for combined across two counterbalanced groups.

Combined took 10.56 seconds versus 10.80 seconds for legacy, a 2.2% reduction.
Leakage remained zero for every output, repair counts agreed for all rows, 10/12
outputs were byte-identical, and 11/12 review decisions agreed. The paired mean
utility delta was -0.0867 with median zero and an approximate 95% interval of
[-0.2496, 0.0763]. Separate real-model generation and judge calls remain
nondeterministic, so this establishes latency and behavioral parity rather than
a speedup.

Performance is therefore a regression guardrail, not the integration rationale.

## Completion and Follow-up

1. [x] **Conditional behavior**: verify row-local skipping, multiple repairs,
   exhausted repairs, passthrough defaults, row order, and graph validation.
2. [x] **Failure-attribution scope**: keep this implementation opt-in until the
   terminal-failure API in
   [Data Designer #861](https://github.com/NVIDIA-NeMo/DataDesigner/pull/861)
   is merged, released, and integrated. Data Designer 0.8 task traces expose
   column and row position only with full tracing, which is not an appropriate
   production result contract.
3. [x] **Measurements**: record one physical `rewrite-combined` Data Designer
   workflow while preserving aggregate model usage, repair counts, review flags,
   and runner-level row counts. Precise failure-stage measurements remain part of
   the failure-attribution gate.
4. [x] **Behavioral equivalence**: compare legacy and combined public outputs with
   deterministic repaired results, and cover partial row loss and malformed
   initial rewrites.
5. [x] **Scale guardrails**: local 64-row mostly-skipped and mostly-repaired
   batches pass, an earlier Slurm suite completed 60 executions without workload
   failures, and retained GB300 telemetry showed flat HBM use. Combined artifacts
   were larger but remained below 230 KB in the controlled runs. The retained
   telemetry cadence does not support tail-latency conclusions, which are not
   required for this opt-in proof of concept.
6. [x] **Default rollout deferred**: a follow-up after Data Designer #861 will
   consume terminal failure provenance, make the combined graph the default, and
   retain the legacy path as an explicit fallback. Legacy removal can follow
   production rollout evidence.
7. [x] **Performance guardrail**: rerun the corrected paired benchmark with
   balanced ordering and equivalent repair decisions.

## Portability

The current closure-based custom columns are sufficient for the local in-process
path. Serializable plugin configs remain part of the broader workflow-column plan
and become a prerequisite when distributed rewrite graph export is supported.
