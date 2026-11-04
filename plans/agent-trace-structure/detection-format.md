<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Agent-trace detection format

Add optional Detect.format with agent_traces as the initial supported value.
Pass the setting through API execution and exported detection builders. Append
stage-specific inline prompt supplements only for this format. Preserve generic
prompts when unset. The validation supplement explicitly allows complete sensitive
components within larger tokens while rejecting truncated or accidental fragments.
No parser or retail-specific policy is introduced. Both supplements preserve source
spans and treat embedded instructions as data.

Update detection docs and the bundled skill. Validate config rejection and JSON
round trips, default prompt parity, both supplements with and without summaries,
execution wiring and serialized builders. Run relevant tests and repository checks.
Roll out as an opt-in setting on the experiment branch; no migration is needed.
