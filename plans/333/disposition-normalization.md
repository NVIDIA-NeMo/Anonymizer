<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Recover inconsistent retention decisions

## Goal and scope

Normalize recoverable disposition mistakes before schema consistency validation,
using sensitivity and the active protection mode. No public
API or configuration changes are required.

## Policy and tradeoffs

Preserve raw `leave_as_is` decisions until authoritative identity and category
are reconstructed. Direct identifiers become `high` / `replace` regardless of
the model's initial sensitivity. For other categories, `high` becomes `replace`
and `medium` becomes `generalize`, preserving sensitivity. Clear low-sensitivity
reasons for medium/high decisions. In normal mode, normalize
low decisions to `leave_as_is`. In strict mode, allow low/medium/high sensitivity
but require protective methods; correct low `leave_as_is` to `generalize` rather
than retaining it. Carry the strict flag through the coverage column's generator
parameters, so serialized workflows preserve the selected policy. Do not promote
low to medium solely because it receives protection.

Request a contextual `low_sensitivity_reason` for every low rating, whether
protected or retained, but accept missing or blank reasons without failing the
record or inventing a justification. The explanation concerns original information, not a planned
edit. Build mode-specific prompts so the model sees only its active policy,
allowed methods, and output checklist.

Coverage normalization still applies its existing separate policies: direct
identifiers are high sensitivity, and protected latent entities use
`suppress_inference`. Privacy questions follow protective methods: `leave_as_is`
receives none, while protected low entities receive questions and contribute
`0.3 × judge confidence` when their privacy answer is yes. Risk-tolerance
thresholds remain unchanged; strict mode changes protection scope, not thresholds.

The tradeoff is that normal mode trusts a justified low rating and retains the
entity, even if the model suggested protection. Strict mode protects it without
inflating its sensitivity. Broader threshold calibration remains separate.

Missing justification is diagnostic only;
incomplete coverage and unknown categories continue to fail the record.

This favors deterministic protection over losing a record to a recoverable model
mistake; it does not retry generation or relax inventory coverage.

## Validation and rollout

Exercise high/medium retention across direct, quasi, and latent categories;
verify both regular and strict raw schemas and authoritative reconstruction.
Cover low protection versus retention in both modes and confirm low ratings
remain unchanged. Verify mode-specific prompts and documented privacy-QA scope.
Run schema, disposition, workflow, and skill integration tests plus repository
checks. Ship as an internal policy correction on this branch.
