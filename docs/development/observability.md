<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Observability

Anonymizer keeps local run measurement in the `anonymizer.measurement` package.
Measurement hooks record timings, counts, model-call summaries, and safety
metrics without changing anonymization behavior. Benchmark tools convert those
records into tables for latency, reliability, model usage, and quality analysis.

Measurement is separate from anonymous NVIDIA telemetry. Telemetry can report
one product event per run or preview. Users can opt out as described in
[Telemetry and Privacy](../index.md#telemetry-and-privacy). Measurement records
are local artifacts. They are written only when developer tooling or caller code
activates a measurement session.

## Model

Instrumentation is passive unless a `MeasurementCollector` is active in the
current context:

```python
from anonymizer.measurement import MeasurementConfig, configured_measurement_session

measurement = MeasurementConfig(output_path="benchmark-runs/case/measurements.jsonl")

with configured_measurement_session(measurement):
    result = anonymizer.run(config=config, data=data)
```

Instrumentation uses these entry points:

- `stage_timer(...)` wraps pipeline phases and records elapsed time.
- `record_run_metadata(...)` records config, input, model, and runtime metadata
  once per run, without raw source values.
- `record_record_metrics(...)` records per-row counts and safety metrics from
  the trace DataFrame.
- `record_ndd_workflow(...)` records DataDesigner workflow summaries at the
  `NddAdapter` boundary.
- `record_model_workflow(...)` records benchmark-only direct model calls that do
  not use DataDesigner.

The public API and CLI do not read measurement environment variables by default.
Benchmark and developer tools opt into measurement explicitly.

## Record Types

Measurement output is JSONL by default. Each row has a `record_type` and shared
run metadata.

| Record type | Meaning |
| --- | --- |
| `run` | One anonymization call: mode, strategy, input shape, config metadata, model aliases, runtime metadata. |
| `stage` | Pipeline phase timing, status, row counts, and row throughput. |
| `record` | Per-input-row counts, text-length buckets, entity counts, ground-truth comparison metrics when present, replacement coverage, leakage flags, and estimated LLM calls. |
| `ndd_workflow` | DataDesigner workflow summary: workflow name, model aliases, row counts, failures, elapsed time, usage summary, and throughput. |
| `model_workflow` | Direct model workflow summary for benchmark-only paths outside DataDesigner. |
| `dd_trace_coverage` | Trace coverage summary for DataDesigner columns when message tracing is enabled. |

Use `tools/measurement/export_measurements.py` to convert raw measurement JSONL
into Parquet, CSV, or JSONL tables.

## Output and Sinks

`MeasurementConfig` controls output:

| Field | Purpose |
| --- | --- |
| `output_path` | Destination for measurement records. |
| `output_format` | `jsonl` or `json`; defaults to `jsonl`. |
| `record_level` | Include per-row `record` entries; defaults to `True`. |
| `streaming` | Write JSONL records as they are emitted instead of collecting them in memory. |
| `keep_records` | Keep emitted records in memory for caller access. |
| `run_id` | Optional stable run ID. |
| `run_tags` | Caller-supplied tags copied to every record. |
| `fail_on_write_error` | Raise output write/close failures when the run body succeeded. |

Streaming mode supports JSONL only. Use it for long benchmark suites where
holding all measurement records in memory is unnecessary.

`MeasurementConfig.from_env()` can read `ANONYMIZER_MEASUREMENT_*` settings for
developer tooling. Product entry points do not call it automatically.

| Environment variable | Field |
| --- | --- |
| `ANONYMIZER_MEASUREMENT_OUTPUT_PATH` | `output_path` |
| `ANONYMIZER_MEASUREMENT_OUTPUT_FORMAT` | `output_format` |
| `ANONYMIZER_MEASUREMENT_RECORD_LEVEL` | `record_level` |
| `ANONYMIZER_MEASUREMENT_STREAMING` | `streaming` |
| `ANONYMIZER_MEASUREMENT_KEEP_RECORDS` | `keep_records` |
| `ANONYMIZER_MEASUREMENT_DD_TRACE` | `dd_trace` |
| `ANONYMIZER_MEASUREMENT_DD_TRACE_PATH` | `dd_trace_path` |
| `ANONYMIZER_MEASUREMENT_DD_TASK_TRACE_PATH` | `dd_task_trace_path` |
| `ANONYMIZER_MEASUREMENT_FAIL_ON_WRITE_ERROR` | `fail_on_write_error` |
| `ANONYMIZER_MEASUREMENT_RUN_ID` | `run_id` |
| `ANONYMIZER_MEASUREMENT_RUN_TAGS` | `run_tags` |

## DataDesigner Message Traces

DataDesigner message traces are optional sidecar artifacts for model-call
debugging:

```python
measurement = MeasurementConfig(
    output_path="benchmark-runs/case/measurements.jsonl",
    dd_trace="last_message",
    dd_trace_path="benchmark-runs/case/traces.jsonl",
)
```

`last_message` stores the final prompt message for each traced DataDesigner
model call. `all_messages` stores the full message list.

Message traces are separate from measurement records. They may contain raw input
text, prompts, generated output, entity values, replacement values, secrets, and
PII. Do not share them unless they have been reviewed or redacted.

Anonymizer requests these traces through DataDesigner native LLM column trace
side effects. That covers `LLMTextColumnConfig` and
`LLMStructuredColumnConfig`. It does not cover model calls made inside
`CustomColumnConfig` generator functions. When tracing is enabled, the
measurement stream records a `dd_trace_coverage` row so benchmark analysis can
see unsupported columns.

## DataDesigner Task Traces

Scheduler task traces are a separate sidecar:

```python
measurement = MeasurementConfig(
    output_path="benchmark-runs/case/measurements.jsonl",
    dd_task_trace_path="benchmark-runs/case/task-traces.jsonl",
)
```

Task traces capture DataDesigner scheduler timing metadata: workflow, column,
row group, row index, task type, status, queue wait time, execution time, total
time, and whether an error was present. They do not store raw DataDesigner error
strings because those strings can contain prompts, outputs, or source values.

## Safety Rules

Measurement records must not contain raw text, entity values, prompts, generated
outputs, replacement maps, provider secrets, or API keys.

Use counts, labels, lengths, buckets, model aliases, status flags, elapsed time,
token counts, request counts, and run-scoped HMACs instead. The collector hashes
record identity with a per-run key. Record hashes can join artifacts from one
run, but they are not stable identifiers across unrelated runs unless the caller
supplies the same hash key deliberately.

When adding instrumentation:

- Put timing around stable phase boundaries, not every helper call.
- Record metadata at the boundary where the information is known.
- Keep raw debug payloads in explicit sidecars, never in measurement records.
- Prefer `run_tags` for benchmark context such as suite ID, case ID, workload,
  config, or experimental strategy.
- Keep benchmark-only strategy switches in `tools/measurement`, not product
  defaults.

## Key Files

| File | Purpose |
| --- | --- |
| `src/anonymizer/measurement/` | Collector, config, context managers, safe record builders, and trace sidecar hooks. |
| `src/anonymizer/interface/anonymizer.py` | Run-level and per-record measurement integration. |
| `src/anonymizer/engine/ndd/adapter.py` | DataDesigner workflow measurement, native message trace capture, and scheduler task trace capture. |
| `tools/measurement/run_benchmarks.py` | Benchmark suite runner that activates measurement sessions and writes per-case artifacts. |
| `tools/measurement/README.md` | Detailed benchmark and analysis command reference. |
