<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Substitution validation for issue #388

For Substitute, compare schema-valid generated mappings with the exact unique
`(entity value, label)` pairs supplied before generation. Retry missing pairs once,
retain already accepted mappings, and fail the affected record if coverage is
still incomplete. Complete records continue and records without entities need no
model call. The user selected this behavior during the investigation of #388.

## Scope and approach

Add a validated-map entry point in `engine/replace/llm_replace_workflow.py`, wire
Substitute to it in `replace_runner.py`, and document the successful-output and
failed-record behavior. Keep `generate_map_only()` and rewrite policy unchanged.
Reuse schema validation, filtering, collision handling, and NddAdapter execution.
Snapshot expected pairs before generation; do not infer coverage from entry counts
or model-returned request columns. Track positions so duplicate IDs and reordered
model output retain their original record association.

Schema validation remains first. This adds one retry for a returned, schema-valid
but incomplete map. DataDesigner's schema-validation retries still apply to each
call. Initial workflow errors retain existing behavior; a wholly failed
completeness retry becomes failures for the affected records while complete
records survive. This work does not introduce permissive transport schemas or
recover invalid entries from a schema-rejected response.

Reject whitespace-only synthetic values in the shared replacement schema using
`pattern="\\S"` alongside `min_length=1`. This requires at least one non-whitespace
character in both Pydantic validation and the JSON Schema supplied to DataDesigner.
Keep valid one-character values and preserve surrounding whitespace on valid
strings. The shared rule also applies to rewrite replacement maps; their existing
schema-retry and failure policy stays unchanged. Test the exported schema through
DataDesigner's actual response parser as well as direct validation, mixed batches,
and recovery through configured schema retries.

## Trade-offs and validation

A targeted retry avoids regenerating accepted values or making another request
for already complete records. Supply the accepted map as retry context so the
missing replacements can remain consistent. Compare exact pairs after filtering;
an unrelated mapping or duplicate does not fill a gap. Repeated occurrences of
the same pair require a single mapping.

Before discarding a schema-valid Substitute entry, prefer its exact requested
pair. If it has no exact match, reconcile only U+00A0 (no-break space), U+2009
(thin space), and U+202F (narrow no-break space) with ASCII space in a comparison
key. The label must match exactly and the normalized key must identify exactly
one pair in the full pre-call inventory. Restore that pair's trusted original
without changing source text, offsets, or the synthetic value. An exact entry
takes precedence over an approximate entry regardless of response order.
Ambiguous matches and other text differences use the existing retry/failure
policy. Do not strip or collapse whitespace, normalize case, or apply broader
Unicode normalization. Keep rewrite's existing exact matching unchanged.

Test recovery without another model call, exact-entry precedence, ambiguity
across the full inventory on both attempts, exact label matching, unsupported
characters, preservation of offsets/text, and the real preview/batch paths.

Exercise recovery, terminal incomplete retries, retry workflow drops, mixed
batches, no entities, repeated occurrences, mismatched labels/originals, preview
limits, duplicate IDs and output reordering. Verify preview and batch paths
through the real adapter with a local scripted provider. Failure summaries and
new logs contain counts and labels rather than entity values. Run relevant unit
tests and repository checks. No public API or default configuration changes are
planned; the bundled public skill template is unaffected.

## Rollout

Create three local commits on `codex/bugfix/388-origin-investigation`, authored
and signed off by memadi: missing-map completeness and one retry, narrow Unicode
whitespace reconciliation, and rejection of whitespace-only synthetic values.
Do not push. Completeness and reconciliation apply only to Substitute; the
nonblank schema rule also applies to rewrite replacement maps. Tests and
documentation make each behavior explicit.
