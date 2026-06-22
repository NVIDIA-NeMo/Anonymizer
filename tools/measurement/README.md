<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Measurement tools

This directory contains developer tools for measuring Anonymizer runs and
exporting measurement JSONL to tables. Run the tools inside the project
environment, either with an activated venv or through `uv run`.

Use these tools when you need evidence about cost, latency, reliability, or
anonymization quality. They are not product entry points.

## Quick export to DataFrames or CSV

Start here when you have a `measurements.jsonl` file and want to analyze it in
pandas, Polars, a spreadsheet, or another local tool.

```bash
uv run python tools/measurement/export_measurements.py \
  benchmark-runs/suite/measurements.jsonl \
  --output benchmark-runs/suite/tables \
  --overwrite
```

By default, the exporter writes one Parquet table per measurement record type
plus `manifest.json`:

- `run.parquet`
- `stage.parquet`
- `record.parquet`
- `evaluation_record.parquet` when replace judge evaluation is enabled
- `ndd_workflow.parquet` when DataDesigner adapter records are present
- `model_workflow.parquet` when direct model workflow records are present

Use CSV or JSONL when those are easier to inspect:

```bash
uv run python tools/measurement/export_measurements.py \
  benchmark-runs/suite/measurements.jsonl \
  --output benchmark-runs/suite/tables-csv \
  --format csv \
  --overwrite
```

Then load the tables directly:

```python
import pandas as pd

records = pd.read_parquet("benchmark-runs/suite/tables/record.parquet")
stages = pd.read_parquet("benchmark-runs/suite/tables/stage.parquet")
ndd = pd.read_parquet("benchmark-runs/suite/tables/ndd_workflow.parquet")
```

You can also read the raw log, but the exporter is the better default because
it splits records by `record_type` and normalizes nested fields into columns.

```python
import pandas as pd

raw = pd.read_json("benchmark-runs/suite/measurements.jsonl", lines=True)
```

## System overview

The measurement system has three layers:

- Instrumentation in Anonymizer emits JSONL records for runs, stages,
  DataDesigner workflows, direct model workflows, per-record safety metrics,
  and optional sanitized replace-judge evaluation metrics.
- Benchmark runners create repeatable workloads and write those JSONL records
  plus optional sidecars such as detection artifacts and DataDesigner traces.
- Analysis tools convert raw run artifacts into case, group, and model tables.

External/distributed execution is a separate boundary. Detection export APIs are
responsible for building DataDesigner configs that an external runtime can
execute. The tools here should consume the resulting measurement JSONL,
detection artifacts, and trace sidecars; they should not own SLURM
orchestration or distributed DataDesigner execution.

## Tool map

| Task | Tool |
| --- | --- |
| Export raw measurement JSONL to tables | `export_measurements.py` |
| Run repeatable Anonymizer suites | `run_benchmarks.py` |
| Inspect detection artifact sidecars | `analyze_detection_artifacts.py` |
| Analyze benchmark output directories | `analyze_benchmark_output.py` |

Most workflows start with `run_benchmarks.py`, then either export the raw
measurement log with `export_measurements.py` or summarize the benchmark output
directory with `analyze_benchmark_output.py`.

## Implementation shape

The scripts keep workload-specific row models and metric logic local, but share
boring command and export policy through `measurement_tools/`:

- `measurement_tools.cli`: `LogFormat`, logging setup, and structured
  bad-input errors.
- `measurement_tools.tables`: `ExportFormat`, model-row table specs, manifest
  writing, and CSV/Parquet/JSONL output.
- `measurement_tools.stats`: small numeric helpers used by analysis groupers.

This is intentionally composition-based. New analysis tools should declare
their own row models and call the shared helpers rather than inheriting from a
common analyzer base class.

## Benchmark runner

`run_benchmarks.py` runs repeatable Anonymizer workloads and writes the same
measurement JSONL format, one raw file per benchmark case plus a combined
`measurements.jsonl`.

```bash
uv run python tools/measurement/run_benchmarks.py suite.yaml --output benchmark-runs/suite
uv run python tools/measurement/run_benchmarks.py suite.yaml --dry-run --json
uv run python tools/measurement/run_benchmarks.py suite.yaml \
  --output benchmark-runs/suite \
  --dd-trace last_message
uv run python tools/measurement/run_benchmarks.py suite.yaml \
  --output benchmark-runs/suite \
  --dd-task-trace
```

The repo-data smoke suite can be run with DataDesigner traces enabled:

```bash
bash tools/measurement/examples/run-repo-data-smoke-with-dd-traces.sh
```

The script writes to `/tmp/anonymizer-repo-data-smoke-dd-traces` by default.
Pass a different output directory as the first argument, or set
`DD_TRACE_MODE=all_messages` when full chat history is needed:

```bash
DD_TRACE_MODE=all_messages \
  bash tools/measurement/examples/run-repo-data-smoke-with-dd-traces.sh \
  /tmp/anonymizer-repo-data-smoke-dd-traces-full
```

## Benchmark CI

`.github/workflows/benchmark-ci.yml` runs the same benchmark runner from a
manual GitHub Actions dispatch. It targets the self-hosted
`anonymizer-evals` runner, checks out the requested ref, installs the project
environment, runs a suite, appends a short case summary to the GitHub step
summary, and uploads the full output directory as a workflow artifact.

The job is intentionally manual. It runs only through `workflow_dispatch`; it
does not run on `push`, `pull_request`, `schedule`, or the default PR CI path.
GitHub exposes manual dispatch only after the workflow file exists on the
repository default branch. After that, launch it from the Actions UI, GitHub
CLI, or API.

The default suite is `tools/measurement/examples/repo-data-smoke.yaml`. Dispatch
inputs let operators choose the ref, suite path, output directory,
DataDesigner message trace mode, sanitized scheduler task traces, and
fail-fast behavior. The workflow requires the repository secret
`NVIDIA_API_KEY` because the default model configuration uses NVIDIA-hosted
models.

The `ref` input defaults to `main`. To benchmark a PR or experiment branch, set
`ref` to that branch name or commit SHA. The workflow checks out that ref and
uses the benchmark runner and suite files from the checkout, so the selected ref
must contain `tools/measurement/run_benchmarks.py` and the requested suite path.

Benchmark suites are YAML files with three parts:

- `workloads`: input datasets and text-column metadata.
- `configs`: Anonymizer replace or rewrite configurations.
- `matrix`: optional workload/config pairs and repetition counts. When omitted,
  every workload is crossed with every config once.

Example:

```yaml
suite_id: biography-smoke
model_configs: ./model-configs.yaml
model_providers: ./providers.yaml
run_tags:
  anonymizer_ref: main
  commit_sha: abc123
  pipeline_id: "456"
case_retries: 1
case_retry_backoff_sec: 10
workloads:
  - id: biographies
    source: ./data/biographies.csv
    text_column: text
    row_limit: 25
  - id: support
    source: ./data/support.csv
    text_column: body
    id_column: ticket_id
    row_offset: 100
    row_limit: 50
configs:
  - id: redact-default
    replace: redact
    evaluate: true
  - id: hash-agent-labels
    detect:
      entity_labels: [person, email, api_key, password]
    replace:
      strategy: hash
      digest_length: 12
  - id: rewrite-low-risk
    rewrite:
      risk_tolerance: low
      max_repair_iterations: 1
matrix:
  - workload: biographies
    config: redact-default
    repetitions: 3
  - workload: support
    config: hash-agent-labels
```

Use `row_limit` and `row_offset` to create cheap, repeatable slices of a local
CSV or Parquet workload. The runner materializes a per-case sliced input under
`raw/inputs/` before calling Anonymizer, so each case keeps a stable input file
even when the matrix has multiple configs or repetitions. Slicing is rejected
for URL-like sources because the runner cannot safely materialize a local
subset without downloading the whole dataset first.

Relative paths in suite files are resolved from the suite file's directory.
The runner refuses to write into a non-empty output directory unless
`--overwrite` is set. By default it also exports Parquet tables into `tables/`;
pass `--no-export` when only raw measurement JSONL is needed.

Set `model_configs` and `model_providers` explicitly in checked-in or CI suites.
Relying on Anonymizer defaults makes a run depend on the caller's installed
defaults and provider environment. In provider YAML, put environment variable
names such as `NVIDIA_API_KEY` in `api_key`; do not commit raw keys. The bundled
`repo-data-smoke.yaml` follows this pattern with adjacent model/provider files.

Use `run_tags` for stable suite-level metadata copied into every measurement
record, such as source refs, commit SHAs, CI pipeline IDs, topology labels, or
benchmark-suite revisions. The runner reserves `suite_id`, `workload_id`,
`config_id`, `repetition`, and `case_id` for its own case identity tags.

Set `evaluate: true` on a replace config when the benchmark should run
`Anonymizer.evaluate()` after `run()` and capture the LLM-as-judge work in the
same case. This is intentionally replace-only for now; rewrite runs already
perform their internal evaluation/repair loop during `run()`.

When evaluation is enabled, the safe measurement log includes
`evaluation_record` rows with judge verdict booleans and invalid-item counts.
It does not persist the evaluated result dataframe or trace dataframe. Those
dataframes can contain original text, entity values, replacement values, raw
judge outputs, prompts, and model responses.

Before starting a real run, the benchmark runner performs cheap preflight
checks: suite/config parsing, local dataset existence, CSV/Parquet text-column
metadata, provider YAML shape, and active model-alias references. `--dry-run`
runs those same checks, expands the planned matrix, and skips output-dir writes
and model work.

Use `case_retries` and `case_retry_backoff_sec` for long-running suites on
shared model endpoints. Retries are disabled by default. When enabled, a failed
case is retried with the same `case_id` and output paths; the final case still
records `attempt_count` and `attempt_errors` in `summary.json`. `--fail-fast`
remains fail-fast and bypasses retries.

## DataDesigner traces

For debugging DataDesigner calls, pass `--dd-trace last_message` or
`--dd-trace all_messages`. Trace records are written separately from sanitized
measurements, under `traces/{case_id}.jsonl` by default. Use `--trace-dir` to
choose another directory. `last_message` stores only the final prompt message
for each DataDesigner model call; `all_messages` stores the full message list.

DataDesigner traces may contain raw input text, prompts, model outputs, entity
values, replacement values, secrets, and PII. Treat them as debug artifacts:
keep them out of shared benchmark bundles unless they have been reviewed or
redacted.

Anonymizer requests standard LLM-column traces through DataDesigner native LLM
column trace side effects. That covers `LLMTextColumnConfig` and
`LLMStructuredColumnConfig`. Model-backed `CustomColumnConfig` generator
functions are traced through a temporary Anonymizer shim that instruments the
per-run DataDesigner model registry and returned model facades. This is a
brittle bridge over private DataDesigner internals until DataDesigner exposes a
public model-call trace sink.

Safe measurement output includes a `dd_trace_coverage` record with native,
private-facade, and unsupported column counts so trace-enabled runs can detect
which path covered each workflow.

## DataDesigner Scheduler Task Traces

Pass `--dd-task-trace` to collect sanitized DataDesigner async scheduler task
timing records. The benchmark runner writes one sidecar per case under
`task-traces/{case_id}.jsonl` by default; use `--task-trace-dir` to choose
another directory.

Task trace records are separate from raw message traces. They include scheduler
metadata such as workflow name, column, row group, row index, task type, status,
relative dispatch/slot-acquired/completion offsets, queue wait time, execution
time, total time, and whether an error was present. They intentionally do not
store raw DataDesigner error strings because those can contain prompts, outputs,
or source values.

Offsets are relative to the earliest positive `dispatched_at` timestamp in each
DataDesigner workflow trace batch written into the case sidecar. They are meant
for timeline analysis without storing host-specific wall-clock timestamps.

```bash
uv run python tools/measurement/run_benchmarks.py \
  suite.yaml \
  --output benchmark-runs/suite \
  --dd-task-trace
```

## Benchmark analysis

`analyze_benchmark_output.py` joins `measurements.jsonl`, optional
DataDesigner traces, and detection artifact sidecars into richer case/group
tables:

```bash
uv run python tools/measurement/analyze_benchmark_output.py \
  benchmark-runs/suite-id \
  --output benchmark-runs/suite-id/analysis \
  --format csv
```

Important outputs:

- `case_analysis.*`: one row per benchmark case.
- `group_analysis.*`: median and aggregate metrics grouped by workload/config.
- `model_usage.*`: one row per measured model usage entry.
- `model_usage_group_analysis.*`: model usage rolled up by workflow/model.

Use `--detection-artifacts` to provide an explicit detection artifact JSONL
sidecar. Otherwise, the analyzer reads `detection-artifacts.jsonl` in the
benchmark directory when present.

## Pandas patterns

Analysis tables are regular CSV/Parquet files. A typical local workflow:

```python
import pandas as pd

cases = pd.read_parquet("benchmark-runs/suite/analysis/case_analysis.parquet")
groups = pd.read_parquet("benchmark-runs/suite/analysis/group_analysis.parquet")

cols = [
    "workload_id",
    "config_id",
    "median_pipeline_elapsed_sec",
    "median_observed_total_requests",
    "median_observed_total_tokens",
    "median_artifact_final_entity_signature_count",
]
print(groups[cols].sort_values(["workload_id", "median_pipeline_elapsed_sec"]))

failures = cases[
    (cases["case_failed"]) |
    (cases["observed_failed_requests"] > 0) |
    (cases["dd_trace_error_count"] > 0)
]
print(failures[["case_id", "config_id", "observed_failed_requests", "dd_trace_error_count"]])
```

## Metric interpretation

Use metrics as signals, not as a single score.

Latency and throughput:

- `elapsed_sec`: wall time for a measured stage or workflow.
- `pipeline_elapsed_sec`: end-to-end Anonymizer wall time for a case.
- `records_per_pipeline_sec`: completed input records per pipeline second.
- `input_text_tokens_per_pipeline_sec`: input text tokens processed per
  pipeline second.

Model work:

- `observed_total_requests`: measured model requests from DataDesigner or direct
  model workflow records.
- `observed_total_tokens`: measured input plus output tokens.
- `observed_failed_requests`: provider-level failed requests.
- `observed_bridge_fallback_requests`: sync-client fallback requests recorded
  from DataDesigner traces.
- `observed_non_bridge_failed_requests`: failed requests after subtracting
  sync-client bridge fallbacks. Prefer this field when judging endpoint
  reliability from trace-enabled runs.

Detection artifacts:

- `seed_entity_count`: detector or direct-seed candidate count before
  validation.
- `seed_validation_candidate_count`: candidates sent to validation.
- `estimated_seed_validation_chunk_count`: estimated validator chunks from the
  active validation chunk size.
- `augmented_entity_count`: augmenter suggestions.
- `augmented_new_final_value_count`: augmenter suggestions that add values not
  already present in the seed/final set.
- `artifact_final_detector_entity_count` and
  `artifact_final_augmenter_entity_count`: final entity source counts derived
  from detection artifact sidecars.
- `artifact_final_entity_signature_count` and
  `artifact_final_entity_signature_hashes`: opaque final-span signatures derived
  from detection artifacts. These do not include raw entity values.

Safety and replacement:

- `original_value_leak_count`: count of protected original values still present
  in replaced output.
- `replacement_missing_final_entity_count`: final entity occurrences whose
  original value has no replacement-map entry.
- `replacement_missing_final_value_count`: unique final entity values with no
  replacement-map entry.
- `replacement_synthetic_original_collision_count`: final entity occurrences
  whose original value was reused as a synthetic replacement value elsewhere in
  the same record.

Replace judge evaluation:

- `detection_valid`, `type_fidelity_valid`,
  `relational_consistency_valid`, and `attribute_fidelity_valid`: per-record
  judge verdicts when `evaluate: true` is enabled.
- `detection_invalid_entity_count`,
  `type_fidelity_invalid_replacement_count`,
  `relational_consistency_invalid_relation_count`, and
  `attribute_fidelity_invalid_entity_count`: counts of invalid judge findings.
  These fields count structures returned by the judges but do not include raw
  values, replacement strings, or judge reasoning text.
- `case_analysis` also includes per-case rollups for each judge family:
  `{family}_judged_record_count`, `{family}_valid_record_count`,
  `{family}_valid_rate`, and the corresponding invalid-count field.
- `group_analysis` includes grouped micro-rate rollups:
  `sum_{family}_judged_record_count`, `sum_{family}_valid_record_count`,
  `micro_{family}_valid_rate`, and `sum_{invalid_count_field}`. These rates are
  computed from summed counts, not medians of case-level rates.
