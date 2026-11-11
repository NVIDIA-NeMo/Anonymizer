<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Ignored entity values

Add `Detect.ignored_entity_values` for additional exact, case-sensitive exclusions across labels.
Always combine additions with 15 immutable built-in generic payment and email-reference values. No normalization,
label-specific rules, default opt-out, or prompt changes are included.

Wire config through the interface into detection finalization, after all detection stages and label
scope filtering, before grouping and replacement/rewrite consumers. Preserve intermediate detections
for debugging. Latent entities and independent evaluation judges retain their existing behavior;
this option ignores explicit detections and does not promise unchanged rewrite output.

Case-sensitive whole-value matching limits accidental exclusion of identifiers containing payment
terms. Validate defaults, additions, casing, substring boundaries, all-filtered records, grouping,
trace preservation, and config serialization with unit tests; run formatting and type checks.
Ship as an additive config field with documentation and bundled agent skill guidance.
No issue number has been supplied; this plan is stored under the feature name pending issue linkage.
