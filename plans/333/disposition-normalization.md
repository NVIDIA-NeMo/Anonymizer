<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Recover inconsistent retention decisions

## Goal and scope

Normalize recoverable disposition mistakes before schema consistency validation,
using the engine-owned numbered inventory and label classifications. No public
API or configuration changes are required.

## Policy and tradeoffs

For an explicit direct identifier assigned `leave_as_is`, set sensitivity to
`high`, method to `replace`, and clear any low-sensitivity reason. Supply the
classified direct-identifier IDs as validation context when reconstructing raw
responses; do not trust model-generated identity or categories.

For any `medium` decision assigned `leave_as_is`, use `generalize` before field
validation, including strict schemas. Coverage normalization continues to turn
protected latent methods into `suppress_inference`. Preserve existing protective
methods for direct identifiers. High-sensitivity retention without a direct
classification still fails closed. Low decisions still require justification
unless direct-identifier policy already mandates replacement. Incomplete coverage
and unknown categories continue to fail the record.

This favors deterministic protection over losing a record to a recoverable model
mistake; it does not retry generation or relax inventory coverage.

## Validation and rollout

Exercise raw reconstruction for low/medium/high retained direct identifiers,
medium quasi-identifier retention, strict medium normalization, and unchanged
high-retention failure. Run schema, disposition, workflow, and skill integration
tests plus repository checks. Ship as an internal policy correction on this branch.
