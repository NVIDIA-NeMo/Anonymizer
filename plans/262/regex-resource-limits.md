<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Configurable regex resource limits

Address the resource-limit review on #265 without adding warnings or changing detection scope.

- Add public `RegexLimits` under `Detect.regex_limits`: 10,000 locally validated matches per rule,
  a 0.05-second minimum timeout, and 1 second per million Unicode code points.
- Forward these limits through run, preview and both export builders into `RegexDetectionConfig`.
  Injected detectors may omit default limits but must reject unsupported non-default limits.
- Count only locally accepted candidates. Process bounded batches rather than gathering the entire
  raw candidate stream. Keep a single search-time budget per rule and retain non-concurrent regex
  scans. Exclude Python validation and time between batches; arbitrary callbacks are not sandboxed.
- Keep limit exhaustion a failed record, with rule/config identifiers and no input text in errors.
- Preserve the public failed-record sanitizer: use fixed regex/config guidance there, with precise
  limit diagnostics in the underlying error; never copy arbitrary trace or validator text.
- Do not silently truncate successful detection results. Existing validator trace semantics remain.

The fixed 50 ms budget penalizes long records; the character-scaled formula was checked on synthetic
10K/100K/1M-character records across all six built-ins. The match limit remains finite and configurable
instead of silently dropping matches. No new warning subsystem or detector-interface redesign is included.

Validate config constraints/round-trip, public forwarding/export parity, formula boundaries and Unicode
length, accepted-match counting and overflow, ordinary logs/cards, bounded collection and shared deadlines.
Run existing matcher/lifecycle/interface suites, full tests, and `make check`. Document and update the skill
source; regenerate signed artifacts during the separately planned main sync. Publish only on user request.
