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
rows that already pass. The legacy workflow remains the default until Data
Designer exposes terminal failure provenance. After that compatibility gate,
the combined graph can become the only production rewrite path.

## Current Status

`CombinedRewriteWorkflow` remains an opt-in `RewriteWorkflow` subclass while the
legacy path serves as the default and parity oracle. This avoids losing precise
failure-stage attribution before Data Designer exposes the failed column and
seed-row identity through its result API.

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

The authoritative paired run completed 30 pairs without failures. The combined
path reduced median Data Designer workflows from five to three, but did not show
a latency improvement: paired wall time was 1.81% slower and rewrite time was
1.58% slower. Quality and leakage were comparable. The run was confounded by
more repairs, more output tokens, and fixed variant ordering, so it establishes
neither an intrinsic slowdown nor a speedup.

Performance is therefore a regression guardrail, not the integration rationale.

## Production Gates

1. [x] **Conditional behavior**: verify row-local skipping, multiple repairs,
   exhausted repairs, passthrough defaults, row order, and graph validation.
2. [ ] **Failure attribution and default rollout**: add a Data Designer result API
   exposing the failed column and seed-row identity. Until then, keep the legacy
   workflow as the default. Data Designer 0.8 task traces expose the column and
   row position only when full tracing is enabled, which is not a scalable
   production mechanism and does not include Anonymizer's record id.
3. [x] **Measurements**: record one physical `rewrite-combined` Data Designer
   workflow while preserving aggregate model usage, repair counts, review flags,
   and runner-level row counts. Precise failure-stage measurements remain part of
   the failure-attribution gate.
4. [x] **Behavioral equivalence**: compare legacy and combined public outputs with
   deterministic repaired results, and cover partial row loss and malformed
   initial rewrites.
5. [ ] **Scale validation**: local mostly-skipped and mostly-repaired mixed batches
   pass. Peak memory, artifact size, and tail latency still need remote comparison.
6. [ ] **Consolidation**: after failure attribution is available, fold the graph
   into `RewriteWorkflow`, remove `use_combined_graph`, remove the duplicate
   runner, and delete the legacy loop.
7. [ ] **Performance guardrail**: rerun the corrected paired benchmark with
   balanced ordering and equivalent repair decisions.

## Portability

The current closure-based custom columns are sufficient for the local in-process
path. Serializable plugin configs remain part of the broader workflow-column plan
and become a prerequisite when distributed rewrite graph export is supported.
