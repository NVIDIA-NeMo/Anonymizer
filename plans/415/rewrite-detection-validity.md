<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Restore rewrite detection-validity evaluation

## Goal and scope

Fix [#415](https://github.com/NVIDIA-NeMo/Anonymizer/issues/415) on `main` so
rewrite detection-validity evaluation works when every replacement telemetry
entry contains an empty `skipped_span_label_counts` mapping. Preserve the
telemetry in the evaluated trace.

The implementation branch starts from `origin/main` at `f23a2317`.
Aaron and Lipika agreed on option A for `main`: exclude replacement telemetry
from the standalone judge seed and restore it in the output. Aaron's selected
seed-column approach on the graph-SDK branch is separate. Changing the telemetry
storage contract is tracked in [#417](https://github.com/NVIDIA-NeMo/Anonymizer/issues/417).

## Affected code

- `src/anonymizer/engine/evaluation/judge_base.py`: shared standalone judge
  execution and restoration of excluded telemetry.
- `tests/engine/evaluation/test_detection_judge.py`: regression coverage through
  the real shared judge implementation.
- `tests/engine/test_rewrite_workflow.py`: rewrite evaluation regression using
  the real detection judge.

## Implementation steps

1. Add a failing standalone detection-judge regression. Supply synthetic rows
   whose telemetry counters differ but whose skipped-label mappings are all
   empty. The adapter fake must attempt `to_parquet(io.BytesIO())` on its seed
   before returning valid judge payloads.
2. In `_BaseJudgeWorkflow.evaluate()`, use the existing
   `COL_REPLACEMENT_APPLICATION` constant to exclude that column, when present,
   from the frame passed to `NddAdapter.run_workflow()`.
3. Reattach excluded telemetry to returned judge rows using `ROW_ORDER_COL`
   before flattening judgments and recombining passthrough rows. Use a left join
   with one-to-one validation so dropped or reordered adapter rows retain their
   own telemetry. Preserve the adapter's failed-record reporting.
4. Cover normal results, dropped and reordered results, interleaved passthrough
   rows, and inputs without telemetry. Assert unchanged telemetry values and
   restored survivor order. Keep the existing exclusions in replacement and
   entity coverage, whose orchestration uses separate paths.
5. Add a rewrite evaluation regression using the real
   `DetectionJudgeWorkflow`, with an adapter fake that serializes judge seeds
   to Parquet. Assert numeric detection-validity scores and preserved telemetry.

## Trade-offs

Excluding one known telemetry column is the small independently reviewable fix
agreed for `main`. Selecting all seed columns from prompt dependencies requires
a broader change. Changing the telemetry representation affects trace readers
and persistence and belongs with #417. This plan keeps the current trace shape
and public API.

The public `Anonymizer.evaluate()` path already calls `_log_unavailable_scores`
in `src/anonymizer/interface/anonymizer.py` to warn about unavailable scores.
That existing warning handles failure visibility, so this fix does not add a
warning inside `RewriteWorkflow.evaluate()`.

The direct `EntityCoverageWorkflow.evaluate()` override bypasses the shared
base method. Its separate telemetry exposure is acknowledged in the issue
discussion; this patch targets the shared standalone path used by rewrite
detection validity. Existing coverage orchestration exclusions remain necessary.

## Validation and rollout

First verify that the new serialization regressions fail on the current code.
After implementation, run the affected evaluation and rewrite tests, followed
by `mise run check ::: test` before requesting review. All regression data is
synthetic; the adapter fakes require no model-provider credentials.

Open the implementation PR against `main`, link this plan, and reference #415.
Seek maintainer review through the PR. Coordinate the resulting change with
the graph-SDK branch's independently implemented fix; its row-position handling
differs from `main`. No migration or configuration change is required.

## Implementation and validation results

Implemented the shared seed exclusion and row-order-based telemetry
restoration. Added standalone serialization regressions covering telemetry
presence, reordered and dropped rows, and passthrough rows, plus a real-judge
rewrite regression.

Before the fix, the standalone telemetry cases reproduced the reported
`ArrowNotImplementedError`, and rewrite evaluation returned unavailable scores.
After removing the extra warning and its test, all six static checks passed via
`mise run check`, and 284 evaluation, rewrite, replacement, entity-coverage, and
public-interface logging tests passed.

Validation used a temporary Mise 2026.10.15 installation and the repository's
pinned tools. On macOS, the full suite also required the already installed GNU
coreutils directory on `PATH` and `TMPDIR=/private/tmp` to keep Unix socket paths
short while using a consistent temporary-file root. These were invocation-only
settings; repository tooling and test configuration were not changed.
