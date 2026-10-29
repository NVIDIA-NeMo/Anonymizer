<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# GLiNER candidate recall

Keep the broader Anonymizer entity scope and the 0.3 threshold. Use native word
windows (128 words, 32 overlap), preserve overlapping alternatives until
validation, and translate five equivalent labels only at the pinned GLiNER2
server boundary. Restore requested labels before returning detector results.
Do not change augmentation grounding or introduce global occurrence copying.

The workflow must preserve explicitly configured window settings and otherwise
let the server choose its defaults. Reuse the native-word chunking implementation
already on main, including source-bound offsets and its synthetic-period guard.
Deployments using an older separately managed server need to update that server;
client changes alone cannot restore candidates suppressed during inference.

Tests cover chunk boundaries, Unicode/source offsets, duplicate removal without
overlap suppression, native inference arguments, reverse label mapping (including
alias collisions), server defaults, and workflow configuration pass-through.
Run the repository checks and unit suite before committing.

In a selected 32-record diagnostic cohort, paired annotate runs recovered 24 of
25 reviewed targets versus 22 with the earlier detector configuration, with no
tagged negative controls among eight controls. This is targeted evidence, not a
full repeated benchmark. Candidate counts rose from 2,016 to 2,925; cached-detector
pipeline time increased from about 11 to 15 minutes. Retaining alternatives trades
more validation work for cleaner candidates. Keep these costs visible in review.
