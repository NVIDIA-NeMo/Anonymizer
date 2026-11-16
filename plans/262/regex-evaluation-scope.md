<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Regex labels in coverage evaluation

Persist resolved detection labels on run/preview results and forward them through
replace/rewrite evaluation into the coverage prompt's existing taxonomy list.
Keep the original `entity_labels` for scope filtering: `None` remains permissive.
Reuse detection's label resolver rather than rebuilding regex scope in evaluation.
Old/manual results default to the existing taxonomy. No configuration or scoring
change is needed. Validate prompt content, permissive filtering, and metadata
forwarding with no-model tests; ship as a small correction within PR #265.
