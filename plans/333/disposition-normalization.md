<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Recover inconsistent retention decisions

## Goal and scope

Normalize recoverable disposition mistakes before schema consistency validation,
using the engine-owned numbered inventory and label classifications. No public
API or configuration changes are required.

## Policy and tradeoffs

Normalize `leave_as_is` according to sensitivity alone, before field and
consistency validation: `high` becomes `replace`, and `medium` becomes
`generalize`. Clear low-sensitivity reasons on both paths. Apply the same rule
in regular and strict schemas, without consulting direct/quasi classification.
Preserve valid protective decisions and sensitivity levels during this correction.

Coverage normalization still applies its existing separate policies: direct
identifiers are high sensitivity, and protected latent entities use
`suppress_inference`. Low retention still requires contextual justification;
incomplete coverage and unknown categories continue to fail the record.

This favors deterministic protection over losing a record to a recoverable model
mistake; it does not retry generation or relax inventory coverage.

## Validation and rollout

Exercise high/medium retention across direct, quasi, and latent categories;
verify both regular and strict raw schemas and authoritative reconstruction.
Run schema, disposition, workflow, and skill integration tests plus repository
checks. Ship as an internal policy correction on this branch.
