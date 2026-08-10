<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Combined Rewrite Graph

Tracks the proof of concept for [GitHub issue #237](https://github.com/NVIDIA-NeMo/Anonymizer/issues/237).
It extends the rewrite portion of [Anonymizer Workflow Columns](../custom-column-plugins/anonymizer-workflow-columns.md).

## Scope

Collapse the post-detection rewrite path into one DataDesigner execution:

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
rows that already pass.

## Current Proof

`CombinedRewriteWorkflow` is an opt-in `RewriteWorkflow` subclass. The production
`RewriteWorkflow.run()` path remains unchanged.

The graph currently:

- generates and filters the replacement map in the same execution;
- reuses the existing domain, disposition, QA, rewrite, evaluator, and repair helpers;
- supports `max_repair_iterations >= 0` by statically unrolling rounds;
- preserves no-entity passthrough outside DataDesigner;
- restores the existing final rewrite, metric, repair-count, and review columns;
- leaves the separate `evaluate()` judge path unchanged.

With two repair rounds, the graph has 36 columns and DataDesigner 0.8 validates it
without duplicate producers, missing dependencies, or cycles.

## Expected Execution Change

For a rewrite run with entity rows:

| Path | Base DD runs | DD runs per repair round |
|---|---:|---:|
| Current full pipeline | 5 | 2 |
| Proof of concept | 3 | 0 |

The totals include the two existing detection runs. This proof combines only the
post-detection rewrite work, so a full detection-plus-rewrite graph remains a later step.

## Gaps Before Production Use

1. **Failure attribution**: a dropped row is currently reported as `rewrite-combined`
   instead of the precise replacement, evaluation, or repair stage.
2. **Measurements**: the existing per-workflow measurement boundaries become one
   workflow. Column/task traces need to provide equivalent stage reporting.
3. **Portable configs**: the proof remaps existing custom generators with closures.
   Stage 2 of the workflow-column plan should replace these with serializable
   Anonymizer plugin configs before exporting the graph to distributed runtimes.
4. **Behavioral equivalence**: run the legacy and combined paths against the same
   deterministic provider responses, including partial failures and malformed outputs.
5. **Performance evidence**: benchmark both paths with identical inputs, providers,
   concurrency, repair decisions, and tracing settings.

## Next Experiment

Implement namespaced rewrite evaluator and repair plugin configs, then add a private
opt-in in the Anonymizer facade so the legacy and combined paths can be benchmarked
without constructing internal adapters manually.
