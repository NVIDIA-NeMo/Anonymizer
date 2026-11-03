<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Plain-text augmentation on the detection lifecycle

## Scope and implementation

PR #267 is stacked on PR #305 (`lipikaramaswamy/feature/detection-semantics`),
ported against `b6432169`. The original #267 commit is `a6f683af`.
Augmentation reads canonical `COL_TEXT` without seed tags or an existing-entity
list. Remove already-tagged, overlap-group, and tag-stripping instructions;
show plain-text examples in strict and permissive label modes.
Keep `value`, `label`, `context_before`, `context_after`, and `reason`, exact
source grounding, bounded grounding retries, label scope, existing date policy,
independent admission, and shared final reconciliation unchanged.

Do not port the old `resolve_overlaps()` patch: #305 replaces that path.
Exact rediscoveries coalesce with accepted provenance; conflicting labels and
bounds follow #305's existing policy, including coverage before source.
No every-occurrence instruction or native schema enforcement is added.
Only augmentation input dependencies change; detector/validator contracts do not.

## Reported local experiment

These are handoff results, not newly reproduced measurements. The experiment
adapted `a6f683af` onto #305 at `97c0ae80`, replaying augmentation with each
original run's exact validated seeds and no new detector or validator calls.
Sampling and the 32,768-token output limit were unchanged. Luna and Nemotron
3.5 Super VL used NVIDIA Inference; Gemma 4 26B A4B used OpenRouter.

| Model | Selected known-miss checks: current → plain text | Previously correct checks |
| --- | --- | --- |
| Luna | 6/9 → 9/9 | 15/15 → 15/15 |
| Super | 7/16 → 15/16 | 9/9 → 9/9 |
| Gemma | 16/45 → 45/45 | 6/6 → 3/6 |

Repeated targets contribute repeated checks, not unique records. All 34
Luna/Super replay attempts returned. Gemma returned 48/51 after retrying only
failed attempts on another provider; three still failed output-format validation.
No invalid final source spans were found among returned replays.
Gemma omitted one organization mention in all three control repeats; the
finding was absent from model output rather than discarded by reconciliation.
Average augmentation time including retries rose approximately 19 → 28 seconds
for Luna and 19 → 32 seconds for Super. Gemma timing is confounded by concurrent
traffic and cross-provider recovery; no causal speed claim is supported.

This is a small selected failure-enriched cohort using saved baseline outputs,
not a fresh counterbalanced comparison. Provider recovery further confounds
Gemma. These results do not establish population precision/recall, correctness
of all additional predictions, or an across-the-board improvement.

## Validation and outstanding evaluation

Offline tests render strict/permissive plain-text examples, preserve literal
source markup and Unicode, and exercise grounding, duplicate rediscovery,
conflicting labels/bounds, accepted provenance, and organization omission versus
reconciliation. The organization control is synthetic: it verifies pipeline
handling and cannot reproduce or fix an LLM omission. The exact reported control
fixture was not supplied locally.

Before broad quality or performance claims, run fresh paired seed-aware versus
plain-text augmentation on identical records/seeds with fixed providers, models,
sampling and output limits, counterbalanced order, and repeats. Include the exact
Gemma organization control once a sanitized fixture is available. Report omissions,
format/transport failures, grounding failures, final spans, added-prediction
precision, recall, tokens, and latency including retries; do not mix providers
without separating those results. This provider-backed comparison remains pending.

Run focused detection tests, the full unit suite, and repository checks.
Keep this prompt change separate from #305 and the regex work in #265.

## Port validation results

- Focused detection/lifecycle selection: 174 passed.
- Full suite: 1,825 passed (`TMPDIR=/tmp uv run --group dev pytest --basetemp=/tmp/anon267-final -q`).
  Aligning temporary roots avoids unrelated macOS Unix-socket path limits and
  a measurement test's temporary-root assertion.
- `make check` passed with `UV_PROJECT_ENVIRONMENT=/tmp/anonymizer-267-check`
  after `uv sync --group dev --group docs` in that isolated environment.
  The existing local environment's optional notebook packages expose two
  unused-ignore warnings in #305's backend; no unrelated source fix is included.
- `make docs-build` passed in strict mode.
