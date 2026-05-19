<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Benchmark CI handoff

This PR adds the first benchmark CI scaffold for Anonymizer. It is intentionally small: the goal is to prove the Brev self-hosted runner, real provider calls, artifact output, and a config-driven benchmark shape that other people can extend.

## What exists now

The new workflow is `.github/workflows/benchmark-ci.yml`. It is manual-only through `workflow_dispatch` and runs on the self-hosted runner labels:

```yaml
runs-on: [self-hosted, anonymizer-evals]
```

The workflow checks out two copies of the repo:

- the benchmark harness at the workflow commit
- the target ref under `.benchmark-sut`

It installs the target package into `.venv`, runs `scripts/benchmark_ci.py`, adds a GitHub step summary, and uploads `benchmark-results/` as an artifact.

The benchmark runner now reads JSON configs. The first one is `benchmarks/configs/smoke.json`. It defines one public-safe dataset, `synthetic_bios`, backed by `docs/data/NVIDIA_synthetic_biographies.csv`, and two preview experiments:

- `rewrite_preview`
- `redact_preview`

Datasets can be local paths or URLs. Every dataset can include a `sha256`; URL datasets must include one. The runner verifies the hash before running and records dataset metadata in `results.json`.

## What the current output tells us

The CI artifact contains:

- `results.json`
- `summary.md`

The summary includes the target ref, target commit, Anonymizer version, harness commit, config name, config hash, dataset metadata, experiment metrics, and per-model latency.

The last config-driven self-hosted smoke run passed on `redact_preview`:

- duration: about 50s for 1 row
- failed records: 0
- model calls: 5
- GLiNER: about 0.3s mean latency
- `gpt-oss-120b`: about 16s mean latency, about 25s max
- endpoint: `https://integrate.api.nvidia.com/v1`

So the smoke benchmark is useful as a real-provider health check and bottleneck detector, but it is not yet a quality benchmark.

## Things we learned

The runner label `anonymizer-evals` is working.

The CI secret `NVIDIA_API_KEY` is available to the benchmark workflow.

Manual dispatch could not be tested from the PR branch because GitHub only exposes new `workflow_dispatch` workflows after the workflow exists on the default branch. We temporarily added a branch-guarded `pull_request` trigger, tested the workflow, then removed that trigger before publishing the PR.

`SyncClientUnavailableError: Sync methods are not available on an async-mode HttpModelClient.` is expected DataDesigner bridge behavior in this path. DD raises it when a sync facade call reaches an async-mode client, catches it in the async custom-column bridge, and falls through to `agenerate()`. The benchmark latency probe now ignores that internal sentinel so it does not look like a model failure.

## How to add another benchmark

Add a dataset and experiment to a config file:

```json
{
  "datasets": {
    "my_dataset": {
      "path": "docs/data/example.csv",
      "sha256": "<sha256>",
      "text_column": "text",
      "data_summary": "Short dataset description"
    }
  },
  "experiments": [
    {
      "name": "redact_my_dataset",
      "pipeline": "redact",
      "dataset": "my_dataset",
      "num_records": 10
    }
  ]
}
```

For remote data, use `url` instead of `path`. The runner downloads it into `.benchmark-data-cache` and verifies `sha256` before use.

Supported pipeline values today:

- `redact`
- `rewrite`

The workflow input `experiments` can run `all`, `redact`, or `rewrite`. The `num_records` workflow input overrides each configured experiment's `num_records`, which is useful for quick smoke runs.

## What is not built yet

This PR does not add baselines or regression gates. It only emits current-run metrics.

The next useful pieces are:

1. Add `baseline_ref` and run the same config against target and baseline.
2. Render a delta report from two `results.json` files.
3. Add a nightly schedule once baseline comparison exists.
4. Add annotated fixtures and detection metrics: recall, precision, F1, span accuracy, label accuracy.
5. Add more datasets through the config path, preferably pinned public-safe data first.
6. Decide where long-lived benchmark artifacts should live: GitHub artifacts are fine for smoke runs, but S3 or another bucket is better for history.
7. Eventually split this into a fuller `evals/` harness if RAT-Bench, private datasets, or heavier dependencies become part of the benchmark suite.

## Recommended next commit

Add baseline comparison. Keep it simple:

- add `--baseline-results` or `--baseline-ref`
- compare aggregate experiment metrics and model latency rows
- emit `delta.md`
- upload both raw results and delta

That turns the scaffold from "did the benchmark run" into "what changed." It is the point where nightly runs become genuinely useful.
