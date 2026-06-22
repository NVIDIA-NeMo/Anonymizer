<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Measurement tool agent notes

This directory is for benchmark and analysis tooling around Anonymizer. Keep
product behavior in `src/anonymizer` and keep benchmark-only strategy switches
inside `tools/measurement`.

## Boundaries

- The measurement layer records facts about Anonymizer runs. It should not
  decide production defaults.
- `run_benchmarks.py` owns local benchmark suite execution, preflight checks,
  per-case raw shards, and measurement export.
- Direct and staged probe scripts are prompt/runtime experiments. Promote a
  probe into `run_benchmarks.py` only after it has stable artifacts, analysis
  fields, and regression coverage.
- Distributed DataDesigner execution belongs outside this directory. Detection
  export APIs build configs for an external runtime; the measurement tools
  should analyze the artifacts that runtime writes.
- Shared command-line concerns live in `measurement_tools/`: CLI logging,
  output formats, table writing, and small numeric aggregations. Do not
  redefine `LogFormat`, `ExportFormat`, bad-input logging, or model-row table
  export in each script.
- Prefer explicit specs and functions over analyzer base classes. A script
  should own its row models, parsing, and metric semantics; shared helpers
  should own boring IO/aggregation policy.

## Tests

Prefer fixtures that look like tool inputs over large constructed tables inside
test functions. For analysis tools, checked-in fixture directories under
`tests/fixtures/measurement/` are easier to review than hundreds of inline
JSON-like rows.

Keep tests focused on contracts:

- input files accepted by the tool
- output table shape and key grouping fields
- safety gates and verdicts
- sensitive values excluded from sanitized analysis output
- preflight failures for user-actionable mistakes

Avoid exhaustive assertions for every derived metric in one test. Add a focused
test when a metric has non-obvious behavior or has regressed before.
