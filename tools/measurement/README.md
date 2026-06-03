<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Measurement Tools

`export_measurements.py` converts Anonymizer measurement JSONL into one table
per `record_type`.

Run these tools inside the project environment, either with an activated venv
or through `uv run`.

```bash
uv run python tools/measurement/export_measurements.py measurements.jsonl --output tables
```

By default it writes Parquet files plus `manifest.json`:

- `run.parquet`
- `stage.parquet`
- `record.parquet`
- `ndd_workflow.parquet` when adapter records are present

Use `--format csv` or `--format jsonl` for non-Parquet output, and
`--overwrite` to replace existing output files.

`run_benchmarks.py` runs repeatable Anonymizer workloads and writes the same
measurement JSONL format, one raw file per benchmark case plus a combined
`measurements.jsonl`.

```bash
uv run python tools/measurement/run_benchmarks.py suite.yaml --output benchmark-runs/suite
uv run python tools/measurement/run_benchmarks.py suite.yaml --dry-run --json
```

Benchmark suites are YAML files with three parts:

- `workloads`: input datasets and text-column metadata.
- `configs`: Anonymizer replace or rewrite configurations.
- `matrix`: optional workload/config pairs and repetition counts. When omitted,
  every workload is crossed with every config once.

Example:

```yaml
suite_id: shell-and-biography-smoke
model_configs: ./model-configs.yaml
model_providers: ./providers.yaml
workloads:
  - id: biographies
    source: ./data/biographies.csv
    text_column: text
  - id: support
    source: ./data/support.csv
    text_column: body
    id_column: ticket_id
configs:
  - id: redact-default
    replace: redact
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

The runner refuses to write into a non-empty output directory unless
`--overwrite` is set. By default it also exports Parquet tables into
`tables/`; pass `--no-export` when you only want the raw measurement JSONL.

## Output Layout

A benchmark run writes one raw measurement file per case, then combines them:

```text
benchmark-runs/suite-id/
  raw/
    biographies__redact-default__r000.jsonl
    support__hash-agent-labels__r000.jsonl
  measurements.jsonl
  summary.json
  tables/
    manifest.json
    run.parquet
    stage.parquet
    record.parquet
    ndd_workflow.parquet
```

Use `summary.json` to inspect case status and errors. Use `measurements.jsonl`
when you need the original structured records. Use `tables/` for analysis.

The exporter groups records by `record_type`:

- `run`: one row per Anonymizer run, with sanitized config, workload, model, and
  runtime metadata.
- `stage`: one row per measured pipeline stage, with elapsed time, row counts,
  and throughput fields.
- `record`: one row per input row when record-level measurement is enabled,
  with text-size buckets, entity counts, replacement counts, rewrite scores,
  and estimated nominal LLM call counts.
- `ndd_workflow`: one row per DataDesigner adapter call, with model aliases,
  elapsed time, row counts, failed-record counts, and observed token/request
  usage when DataDesigner exposes it.

The tables never store raw text, prompts, generated outputs, entity values, or
replacement maps. `record_hash` is a run-scoped HMAC, so it can join rows within
one run but should not be treated as a durable dataset identifier.

## Analysis Patterns

Start with these questions:

- Which workload/config pair is fastest at the same quality target?
- Which stage dominates wall time: detection, replacement, rewrite, or a
  DataDesigner sub-workflow?
- Does latency scale with text length, entity count, or rewrite repair work?
- Do token counts, request counts, and failed records explain latency outliers?
- Are quality metrics worse on one data shape, such as legal text, biographies,
  support tickets, shell history, or mixed natural-language/code records?

Most analyses join `stage`, `record`, and `ndd_workflow` back to `run` through
`run_id`, then group by run tags:

- `run_tags.suite_id`
- `run_tags.workload_id`
- `run_tags.config_id`
- `run_tags.repetition`
- `run_tags.case_id`

Prefer medians and percentiles over averages when comparing latency. LLM calls
usually have long tails, and one retry or provider stall can distort a mean.

## Pandas Examples

Load exported tables:

```python
from pathlib import Path

import pandas as pd

tables = Path("benchmark-runs/shell-and-biography-smoke/tables")
run = pd.read_parquet(tables / "run.parquet")
stage = pd.read_parquet(tables / "stage.parquet")
record = pd.read_parquet(tables / "record.parquet")
ndd = pd.read_parquet(tables / "ndd_workflow.parquet")
```

Compare end-to-end stage latency by workload and config:

```python
stage_group_cols = ["run_tags.workload_id", "run_tags.config_id", "stage"]

stage_summary = (
    stage
    .groupby(stage_group_cols)
    .agg(
        runs=("run_id", "nunique"),
        median_sec=("elapsed_sec", "median"),
        p95_sec=("elapsed_sec", lambda s: s.quantile(0.95)),
        rows_per_sec=("rows_per_sec", "median"),
    )
    .reset_index()
    .sort_values(["run_tags.workload_id", "stage", "median_sec"])
)

print(stage_summary)
```

Find slow records and relate them to text size and entity count:

```python
record_view = record[
    [
        "run_tags.workload_id",
        "run_tags.config_id",
        "record_hash",
        "text_length_tokens",
        "text_length_tokens_bucket",
        "final_entity_count",
        "nominal_llm_call_count",
        "utility_score",
        "leakage_mass",
    ]
].copy()

shape_group_cols = [
    "run_tags.workload_id",
    "run_tags.config_id",
    "text_length_tokens_bucket",
]

by_shape = (
    record_view
    .groupby(shape_group_cols)
    .agg(
        records=("record_hash", "count"),
        median_entities=("final_entity_count", "median"),
        median_nominal_calls=("nominal_llm_call_count", "median"),
        median_utility=("utility_score", "median"),
        median_leakage=("leakage_mass", "median"),
    )
    .reset_index()
)

print(by_shape)
```

Summarize DataDesigner token and request usage:

```python
workflow_group_cols = ["run_tags.workload_id", "run_tags.config_id", "workflow_name"]

token_summary = (
    ndd
    .groupby(workflow_group_cols)
    .agg(
        calls=("workflow_name", "count"),
        median_sec=("elapsed_sec", "median"),
        total_input_tokens=("observed_input_tokens", "sum"),
        total_output_tokens=("observed_output_tokens", "sum"),
        total_requests=("observed_total_requests", "sum"),
        failed_records=("failed_record_count", "sum"),
    )
    .reset_index()
    .sort_values(["run_tags.workload_id", "run_tags.config_id", "median_sec"])
)

print(token_summary)
```

Join run metadata to stage timing:

```python
run_meta = run[
    [
        "run_id",
        "mode",
        "strategy",
        "detect.entity_label_count",
        "detect.validation_max_entities_per_call",
    ]
]

stage_with_config = stage.merge(run_meta, on="run_id", how="left")

config_group_cols = ["mode", "strategy", "detect.entity_label_count", "stage"]

print(stage_with_config.groupby(config_group_cols)["elapsed_sec"].median())
```

For quick interactive work, CSV can be easier than Parquet:

```bash
uv run python tools/measurement/export_measurements.py \
  benchmark-runs/suite-id/measurements.jsonl \
  --output /tmp/suite-csv \
  --format csv \
  --overwrite
```

## Metric Interpretation

Use metrics as signals, not as a single score.

Latency and throughput:

- `elapsed_sec`: wall time for a measured stage or DataDesigner workflow.
- `rows_per_sec`: completed output rows per second for the measured block.
- `tokens_per_sec`: observed total tokens per second when token usage exists.
- `text_length_tokens_bucket`: a coarse text-size bucket for comparing similar
  inputs without storing text.

LLM usage:

- `observed_input_tokens`, `observed_output_tokens`, and
  `observed_total_tokens`: provider-reported usage when available. Missing or
  zero values mean the provider path did not expose usage, not necessarily that
  no tokens were consumed.
- `observed_total_requests`, `observed_successful_requests`, and
  `observed_failed_requests`: request counts when DataDesigner exposes them.
- `nominal_llm_call_count`: an internal estimate based on the Anonymizer
  pipeline shape. Treat it as expected work, not observed provider traffic.

Entity and quality metrics:

- `final_entity_count`: entities that survive detection and validation.
- `final_entity_label_counts`: per-label entity counts serialized as JSON in
  exported tabular files.
- `ground_truth_*`: precision, recall, F1, false positives, and false negatives
  when the input includes one of the supported ground-truth entity columns.
- `utility_score`, `leakage_mass`, `weighted_leakage_rate`,
  `needs_repair`, and `needs_human_review`: rewrite-mode evaluation fields.
  These are null for replace-mode runs.

Error and reliability metrics:

- `failed_record_count`: records dropped by a DataDesigner workflow.
- `status`: completion state for a stage or workflow.
- `summary.json` case errors: runner-level failures, such as invalid inputs or
  model endpoint failures.

## Reading Results Safely

Compare like with like. A shell-history workload, a support-ticket workload,
and a legal-document workload stress different parts of Anonymizer. Group by
`workload_id` before drawing conclusions about model routing, speculative
decoding, validation chunk size, or rewrite repair settings.

Record-level rows describe input shape and output quality, not per-record wall
time. Stage and workflow rows carry timing. To explain a slow run, first find
the slow stage, then inspect the records in that run for text length, entity
count, nominal call count, and rewrite repair signals.

When token or request fields are missing, check `ndd_workflow.model_usage`.
The measurement layer records deeper provider usage only when DataDesigner
returns it.
