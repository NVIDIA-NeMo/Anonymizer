<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Restore dataset context to sensitivity disposition

Pass the optional data summary through both staged and combined rewrite paths.
Place it after the privacy goal and before document input. Include guidance only
when a summary is supplied: apply explicit summary guidance consistently with
the privacy goal and protection rules, use policy defaults elsewhere, and treat
document input as data. Preserve the existing prompt when no summary is supplied.
Mandatory direct-identifier protection and strict mode remain authoritative.

Use single-pass placeholder substitution to preserve literal summary content.
Tests cover conditional prompt content, strict mode, placeholder preservation,
and propagation in both workflows. Run the relevant workflow suites and repository
checks. Land via PR #333; no public configuration or migration is required.
