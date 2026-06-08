<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Measurement tools

This directory contains developer tools for measuring Anonymizer runs, exporting
measurement JSONL to tables, and comparing benchmark strategies. Run the tools
inside the project environment, either with an activated venv or through
`uv run`.

```bash
uv run python tools/measurement/export_measurements.py measurements.jsonl --output tables
```

By default, `export_measurements.py` writes Parquet files plus
`manifest.json`:

- `run.parquet`
- `stage.parquet`
- `record.parquet`
- `ndd_workflow.parquet` when DataDesigner adapter records are present
- `model_workflow.parquet` when direct model workflow records are present

Use `--format csv` or `--format jsonl` for non-Parquet output, and
`--overwrite` to replace existing output files.

## Benchmark runner

`run_benchmarks.py` runs repeatable Anonymizer workloads and writes the same
measurement JSONL format, one raw file per benchmark case plus a combined
`measurements.jsonl`.

```bash
uv run python tools/measurement/run_benchmarks.py suite.yaml --output benchmark-runs/suite
uv run python tools/measurement/run_benchmarks.py suite.yaml --dry-run --json
uv run python tools/measurement/run_benchmarks.py suite.yaml \
  --output benchmark-runs/suite \
  --dd-trace last-message
```

The repo-data smoke suite can be run with DataDesigner traces enabled:

```bash
bash tools/measurement/examples/run-repo-data-smoke-with-dd-traces.sh
```

The script writes to `/tmp/anonymizer-repo-data-smoke-dd-traces` by default.
Pass a different output directory as the first argument, or set
`DD_TRACE_MODE=all-messages` when full chat history is needed:

```bash
DD_TRACE_MODE=all-messages \
  bash tools/measurement/examples/run-repo-data-smoke-with-dd-traces.sh \
  /tmp/anonymizer-repo-data-smoke-dd-traces-full
```

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
dd_parser_compat: none
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

Before starting a real run, the benchmark runner performs cheap preflight
checks: suite/config parsing, local dataset existence, CSV/Parquet text-column
metadata, provider YAML shape, native runtime requirements, and active
model-alias references. `--dry-run` runs those same checks, expands the planned
matrix, and skips output-dir writes and model work.

Use `case_retries` and `case_retry_backoff_sec` for long-running suites on
shared model endpoints. Retries are disabled by default. When enabled, a failed
case is retried with the same `case_id` and output paths; the final case still
records `attempt_count` and `attempt_errors` in `summary.json`. `--fail-fast`
remains fail-fast and bypasses retries.

## Benchmark-only detection strategies

Configs may set `experimental_detection_strategy` for benchmark-only pipeline
probes. These values are not public `Detect` config fields, and they should not
be treated as safe defaults across arbitrary data.

```yaml
configs:
  - id: native-single-pass
    experimental_detection_strategy: native_single_pass
    replace: redact
```

Supported values:

- `default`: run the normal Anonymizer detection pipeline.
- `no_augment`: run GLiNER detection and validation, but skip LLM augmentation.
- `detector_only`: run only GLiNER detection and local finalization. This skips
  LLM validation and LLM augmentation.
- `native_candidate_validate_no_augment`: run a benchmark-only native staged
  detector without DataDesigner using direct OpenAI-compatible calls for seed
  extraction and validation, then skip augmentation.
- `detector_native_validate_no_augment`: run the normal GLiNER detector seed
  through Anonymizer/DataDesigner, then bypass DataDesigner validation and
  augmentation with direct OpenAI-compatible validation calls.
- `detector_native_validate_native_augment`: run the normal GLiNER detector seed
  through Anonymizer/DataDesigner, then bypass DataDesigner validation and
  augmentation with direct OpenAI-compatible validation and augmentation calls.
- `gliner_native_validate_no_augment`: run a direct hosted-GLiNER seed without
  DataDesigner, validate those detector candidates with direct
  OpenAI-compatible calls, and skip augmentation.
- `gliner_native_validate_native_augment`: run a direct hosted-GLiNER seed
  without DataDesigner, validate those detector candidates with direct
  OpenAI-compatible calls, then run direct native augmentation.
- `native_single_pass`: run a benchmark-only native detector without
  DataDesigner using one direct OpenAI-compatible provider call per row. The
  model must return exact values plus `start`/`end` offsets; local code
  validates offsets, resolves overlaps, and records parser/runtime failures as
  `model_workflow` errors.
- `native_single_pass_recall`: the same one-call native detector with a
  recall-oriented prompt that includes Anonymizer's label examples and stronger
  high-recall guidance.
- `native_single_pass_values`: the same one-call native detector, but with the
  value-only prompt shape from `direct_detection_probe.py`. The model returns
  exact values and labels only; local code resolves every occurrence of each
  returned value into spans.
- `native_single_pass_values_recall`: the value-only one-call detector with the
  recall-oriented prompt from `direct_detection_probe.py`.

Native benchmark strategies require an explicit runtime. Set top-level
`native_runtime.endpoint` and `native_runtime.model`, set the standard
`ANONYMIZER_BENCH_NATIVE_ENDPOINT` and `ANONYMIZER_BENCH_NATIVE_MODEL`
environment variables, or override runtime fields per config with
`configs[].native_runtime`. GLiNER-seeded native strategies also require
`native_runtime.gliner_endpoint` and `native_runtime.gliner_model`, or the
standard `ANONYMIZER_BENCH_GLINER_ENDPOINT` and
`ANONYMIZER_BENCH_GLINER_MODEL` environment variables. The runner records
runtime id, alias, provider, model, and env-variable names as run tags; raw
endpoint URLs are not emitted into measurement tables.

```yaml
native_runtime:
  runtime_id: local-vllm-json
  endpoint_env: ANONYMIZER_BENCH_NATIVE_ENDPOINT
  model_env: ANONYMIZER_BENCH_NATIVE_MODEL
  provider: local-vllm
  alias: native-direct
configs:
  - id: native-single-pass
    experimental_detection_strategy: native_single_pass
    replace: redact
```

Use `detector_only` only as a lower-bound ablation. It skips the LLM validation
pass that drops false positives and reclassifies ambiguous spans. A faster run
that loses baseline signatures is a rejection.

Use staged native strategies when the question is "can direct provider calls
replace part of DataDesigner orchestration?" They still need repeated signature,
leak, label-mismatch, parser, and reliability gates before any workload-specific
promotion.

Use one-call native strategies for the more aggressive "collapse detection to
one call" experiment. They are often faster when the prompt works, but they are
more parser- and recall-sensitive. Any malformed JSON response becomes a failed
case in analysis, and any missed baseline signature should be treated as a
rejection rather than a latency win.

## DataDesigner traces

For debugging DataDesigner calls, pass `--dd-trace last-message` or
`--dd-trace all-messages`. Trace records are written separately from sanitized
measurements, under `traces/{case_id}.jsonl` by default. Use `--trace-dir` to
choose another directory. `last-message` stores only the final prompt message
for each DataDesigner model call; `all-messages` stores the full message list.

DataDesigner traces may contain raw input text, prompts, model outputs, entity
values, replacement values, secrets, and PII. Treat them as debug artifacts:
keep them out of shared benchmark bundles unless they have been reviewed or
redacted.

Summarize traced calls without copying raw prompts or responses into analysis
output:

```bash
uv run python tools/measurement/analyze_dd_traces.py \
  benchmark-runs/suite-id/traces \
  --output benchmark-runs/suite-id/trace-analysis \
  --format csv
```

This writes `trace_analysis.*` and `trace_group_analysis.*`. The row table
captures run tags, workflow/model metadata, status, elapsed time, prompt and
response lengths, token counts, and response-shape flags such as `raw_json`,
`fenced_json`, `embedded_json`, `text`, and `none`.

## Direct probes

`direct_detection_probe.py` calls a local OpenAI-compatible endpoint directly
for a small slice of records. It is useful for prompt experiments before adding
a benchmark strategy.

```bash
uv run python tools/measurement/direct_detection_probe.py \
  docs/data/NVIDIA_synthetic_biographies.csv \
  --text-column text \
  --endpoint http://gpu-dev-pod-serve-svc:8000/v1 \
  --model nvidia/nemotron-3-super \
  --labels person,email,api_key,password \
  --row-limit 5 \
  --output /tmp/direct-detection-probe
```

`staged_detection_probe.py` runs a no-DataDesigner staged detector outside the
main benchmark harness. It can compare seed extraction, validation, and
augmentation boundaries before integrating a strategy into `run_benchmarks.py`.

```bash
uv run python tools/measurement/staged_detection_probe.py \
  docs/data/NVIDIA_synthetic_biographies.csv \
  --text-column text \
  --endpoint http://gpu-dev-pod-serve-svc:8000/v1 \
  --model nvidia/nemotron-3-super \
  --labels person,email,api_key,password \
  --row-limit 5 \
  --output /tmp/staged-detection-probe
```

Useful staged options:

- `--seed-source direct-llm`: use direct LLM seed extraction.
- `--seed-source gliner`: use direct hosted GLiNER seeding.
- `--skip-augmentation`: disable augmentation for any seed source. This is an
  ablation for measuring how much recall the augmentation phase carries.
- `--validation-prompt-mode chunked-excerpt`: split seed validation candidates
  into chunks of `--validation-max-entities-per-call` and send each chunk with a
  tagged local excerpt bounded by `--validation-excerpt-window-chars`.

The staged tool writes `staged-detection-cases.jsonl`,
`staged-detection-artifacts.jsonl`, and `summary.json`. Case rows include
per-phase usage for seed extraction, validation, and augmentation, true case
wall time in `elapsed_sec`, model-call time in `model_elapsed_sec`,
`phase_model_work`, `phase_skip_reasons`, `phase_model_requests`,
`model_phase_count`, `model_request_count`, total usage, and optional baseline
signature deltas. Treat `summary.json` as a sensitive debug artifact because it
records the resolved endpoint/model runtime used for the probe.

Summarize staged probe outputs:

```bash
uv run python tools/measurement/analyze_staged_detection_output.py \
  /tmp/staged-detection-probe \
  --output /tmp/staged-detection-probe/analysis \
  --format csv
```

The analyzer accepts either the staged output directory or the
`staged-detection-cases.jsonl` file directly. It writes per-case, per-seed
source group, and label-delta tables. Use `group_analysis.csv` for latency,
token, request, and signature-overlap totals; use `label_delta_analysis.csv` to
see which labels account for baseline-only misses or direct-only additions. The
analysis tables omit raw text and raw entity values.

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

`compare_strategy_pairs.py` compares baseline/candidate case rows:

```bash
uv run python tools/measurement/compare_strategy_pairs.py \
  benchmark-runs/suite-id/analysis/case_analysis.csv \
  --baseline-config default \
  --candidate-config native-single-pass \
  --output benchmark-runs/suite-id/analysis/default-vs-native-single-pass.csv
```

When one CSV does not contain both arms, pass `--candidate-case-analysis`:

```bash
uv run python tools/measurement/compare_strategy_pairs.py \
  baseline/analysis/case_analysis.csv \
  --candidate-case-analysis candidate/analysis/case_analysis.csv \
  --baseline-strategy default \
  --candidate-strategy detector_native_validate_no_augment \
  --output comparison.csv
```

`screen_strategy_comparisons.py` screens many comparison CSVs:

```bash
uv run python tools/measurement/screen_strategy_comparisons.py benchmark-runs/ \
  --output benchmark-runs/strategy-screen.csv
```

Use `--group-by strategy_workload_family` when the same candidate behaves
differently across workload families. Use `--config-aliases aliases.json` to
group related config IDs, such as temperature or validation-window variants of
the same strategy.

## Pandas patterns

Analysis tables are regular CSV/Parquet files. A typical local workflow:

```python
import pandas as pd

cases = pd.read_parquet("benchmark-runs/suite/analysis/case_analysis.parquet")
groups = pd.read_parquet("benchmark-runs/suite/analysis/group_analysis.parquet")

cols = [
    "workload_id",
    "config_id",
    "experimental_detection_strategy",
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

Compare a candidate against a baseline:

```python
comparison = pd.read_csv("benchmark-runs/suite/analysis/default-vs-native.csv")
candidate_rows = comparison[
    ["workload_id", "candidate_verdict", "safety_verdict", "performance_verdict", "flags"]
]
print(candidate_rows)
```

Find candidate-specific misses:

```python
loss_cols = [
    column for column in comparison.columns
    if column.startswith("baseline_only_final_entity_signature_label_counts.")
]
print(comparison[["workload_id", *loss_cols]].fillna(0))
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
- `baseline_only_candidate_covered_signature_count`,
  `baseline_only_candidate_overlapping_signature_count`, and
  `baseline_only_candidate_uncovered_signature_count`: comparison-only fields
  from `compare_strategy_pairs.py`. These split exact signature deltas into
  covered, boundary-overlap, and uncovered losses.
- `candidate_verdict`: `candidate_viable`, `review`, or `reject`.

Treat `candidate_viable` as a promotion candidate, not as an automatic default.
It means the sampled comparison passed the current gates and improved at least
one performance metric without regressing another. Re-run candidates on the
target workload family, with repetitions, before changing production defaults.
