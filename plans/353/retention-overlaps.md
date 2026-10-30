<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Retention checks for overlapping entity values

The retention check must exempt protected matches inside a longer kept value
without hiding protected matches that extend outside it. With protected
"Florida", kept "Florida State University", and protected "Florida State
University Foundation", the Foundation must still trigger repair and review.

Change only the matching logic in `engine/rewrite/retention.py`. Find kept
spans in the normalized original rewrite and exempt fully contained matches.
Keep the existing normalization, word boundaries, and public API. This costs
span comparisons but avoids deleting text that another protected value needs.

Regression tests cover the longer protected name, partial overlap, mixed case,
whitespace variants, and multiple occurrences. Existing tests cover standalone
protected values, all risk presets, both rewrite paths, and review flags.
Run `make test` and `make check` before updating the PR. Also validate with
the repair-loop change in #354, whose test fixtures must supply the retention
column that #353 adds to repair dependencies.

The update requires no migration or configuration change. Land it through #353;
#354 remains dependent on that PR.
