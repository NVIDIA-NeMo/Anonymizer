<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Ignored entity values

Add `Detect.ignored_entity_values` for additional exact, case-sensitive exclusions across labels.
Always combine additions with the immutable built-in ignored-value set. Reject blank entries and
remove exact duplicates while preserving casing and meaningful whitespace. No normalization,
label-specific rules, default opt-out, or prompt changes are included.

Wire config through the interface into a serializable detection finalization transform, after
explicit detection stages and overlap resolution, before grouping and replacement/rewrite consumers.
Export both pre-filter `_detected_entities` and filtered `final_entities`. Serialize the complete
effective ignored-value list so external workers use the saved defaults and additions. Preserve
intermediate detections and `_detected_tagged_text` for debugging. Render public span tags from
filtered `final_entities`. Adapt older injected workflow signatures by finalizing returned rows or
appending a serializable finalization transform to exported builders. Latent entities and independent
evaluation judges retain their existing behavior; this option ignores explicit detections and does
not promise unchanged rewrite output.

Case-sensitive whole-value matching limits accidental exclusion of identifiers containing payment
terms. Validate defaults, additions, casing, substring boundaries, all-filtered records, grouping,
trace preservation, and config serialization with unit tests; run formatting and type checks.
Ship as an additive config field with documentation and bundled agent skill guidance.
No issue number has been supplied; this plan is stored under the feature name pending issue linkage.
