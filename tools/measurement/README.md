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
- `model_workflow.parquet` when non-DataDesigner model workflow records are
  present

Use `--format csv` or `--format jsonl` for non-Parquet output, and
`--overwrite` to replace existing output files.

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

To rerun the repo-data smoke suite with DataDesigner traces enabled:

```bash
bash tools/measurement/examples/run-repo-data-smoke-with-dd-traces.sh
```

The script writes to `/tmp/anonymizer-repo-data-smoke-dd-traces` by default.
Pass a different output directory as the first argument, or set
`DD_TRACE_MODE=all-messages` when you need full chat history:

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
suite_id: shell-and-biography-smoke
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

Use `case_retries` and `case_retry_backoff_sec` for long-running suites on
shared model endpoints. Retries are disabled by default. When enabled, a failed
case is retried with the same `case_id` and output paths; the final case still
records `attempt_count` and `attempt_errors` in `summary.json`. `--fail-fast`
remains fail-fast and bypasses retries. Treat retried cases as reliability
signals during analysis, especially when failures come from provider health
checks or rate limits.

Configs may also set `experimental_detection_strategy` for benchmark-only
pipeline probes:

```yaml
configs:
  - id: shell-rules-only
    experimental_detection_strategy: rules_only
    detect:
      entity_labels: [api_key, email, http_cookie, password, pin, unique_id, url, user_name]
    replace:
      strategy: hash
      digest_length: 12
```

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

Supported values:

- `default`: run the normal Anonymizer detection pipeline.
- `rules_guardrail`: run the normal Anonymizer detection pipeline, then union
  deterministic high-confidence rule spans into the final entity set.
- `rules_filter_guardrail`: remove GLiNER candidates that are fully covered by
  same-label deterministic high-confidence rule spans before validation, add
  non-overlapping rule spans back before augmentation so the augmenter sees them
  as already tagged, then add non-overlapping rule spans into the final entity
  set. Different-label overlaps and longer detector spans remain validation
  candidates so contextual spans such as a multi-token political view,
  university, or organization name are not shadowed by a shorter or differently
  labeled rule span.
- `no_augment`: run GLiNER detection and validation, but skip LLM augmentation.
- `rules_seed_no_augment`: add deterministic high-confidence secret spans to
  the GLiNER seed set, validate those seeds, and skip LLM augmentation.
- `rules_guardrail_no_augment`: run GLiNER detection and validation, skip LLM
  augmentation, then union deterministic high-confidence rule spans into the
  final entity set.
- `rules_filter_guardrail_no_augment`: remove GLiNER candidates that are fully
  covered by same-label deterministic high-confidence rule spans before
  validation, skip LLM augmentation, then add non-overlapping rule spans into
  the final entity set.
- `rules_guardrail_detector_only`: run only GLiNER detection and local
  finalization, then union deterministic high-confidence rule spans into the
  final entity set.
- `detector_only`: run only GLiNER detection and local finalization. This skips
  LLM validation and LLM augmentation.
- `rules_only`: use only deterministic high-confidence rules for the detection
  stage.
- `rules_covered_or_default`: if explicit `detect.entity_labels` are entirely
  inside the structured-secret fast lane (`api_key`, `email`, `http_cookie`,
  `password`, `pin`, `unique_id`, `url`, `user_name`), use deterministic rules
  for rows whose structured assignments are covered and route suspicious
  uncovered rows through the normal Anonymizer detection pipeline. Label sets
  outside the fast lane always use normal detection.
- `native_rules_router`: run a benchmark-only native staged detector without
  DataDesigner. Rule-covered label sets short-circuit through deterministic
  rules with no model calls; other label sets use direct OpenAI-compatible
  provider calls for seed extraction, validation, and augmentation.
- `native_candidate_validate_no_augment`: run a benchmark-only native staged
  detector without DataDesigner using direct OpenAI-compatible calls for seed
  extraction and validation, then skip augmentation. This isolates the cost and
  recall impact of removing the augmentation phase from the native executor.
- `detector_native_validate_no_augment`: run the normal GLiNER detector seed
  through Anonymizer/DataDesigner, then bypass DataDesigner validation and
  augmentation with direct OpenAI-compatible validation calls. This isolates
  whether native validation can replace DataDesigner validation when candidate
  quality is held closer to the default detector path.
- `detector_native_validate_native_augment`: run the normal GLiNER detector
  seed through Anonymizer/DataDesigner, then bypass DataDesigner validation and
  augmentation with direct OpenAI-compatible validation and augmentation calls.
  This keeps the default detector candidate source while testing whether direct
  provider calls can replace the two downstream DataDesigner LLM phases.
- `gliner_native_validate_no_augment`: run a direct hosted-GLiNER seed without
  DataDesigner, validate those detector candidates with direct
  OpenAI-compatible calls, and skip augmentation. This isolates DataDesigner
  detector orchestration overhead while keeping a GLiNER-style candidate source.
- `gliner_native_validate_native_augment`: run a direct hosted-GLiNER seed
  without DataDesigner, validate those detector candidates with direct
  OpenAI-compatible calls, then run direct native augmentation. This is the
  fully staged no-DataDesigner detector/validator/augmenter lane for contextual
  recall experiments.
- `native_single_pass`: run a benchmark-only native detector without
  DataDesigner using one direct OpenAI-compatible provider call per row. The
  model must return exact values plus `start`/`end` offsets; local code validates
  offsets, unions non-overlapping deterministic rule spans, resolves overlaps,
  and records parser/runtime failures as `model_workflow` errors.
- `native_single_pass_recall`: the same one-call native detector with a
  recall-oriented prompt that includes Anonymizer's label examples and stronger
  high-recall guidance.
- `native_single_pass_values`: the same one-call native detector, but with the
  value-only prompt shape from `direct_detection_probe.py`. The model returns
  exact values and labels only; local code resolves every occurrence of each
  returned value into spans.
- `native_single_pass_values_recall`: the value-only one-call detector with the
  recall-oriented prompt from `direct_detection_probe.py`.

These strategies exist to compare performance options. They are not public
`Detect` config fields, and they should not be treated as safe defaults across
arbitrary data. The rule-backed strategies only cover deterministic
high-confidence spans for `api_key`, `date_of_birth`, `email`,
`http_cookie`, `organization_name`, `password`, `pin`, `religious_belief`,
`street_address`, `unique_id`, `url`, and `user_name`; they will not replace
contextual detection for prose identifiers such as names in biographies or
legal documents. The prose rules (`date_of_birth`, `organization_name`,
`religious_belief`, and `street_address`) are narrow contextual patterns and
are not enough to opt into `rules_covered_or_default`; those labels fall back
to default detection unless `rules_only` is explicitly selected. The structured
identifier rules require keyed or command-style syntax such as
`Cookie:`, `pin=`, `trace-id:`, `user_name=`, or service-principal flags. They
are not general entity recognizers. `detector_only` is also unsafe as a default
because it skips the LLM validation pass that drops false positives and
reclassifies ambiguous spans. `rules_only` requires explicit `entity_labels`,
and every label must be covered by those deterministic rules. Use
`rules_covered_or_default` when a benchmark suite may include both fully
structured-secret scans and contextual workloads; it keeps the no-DataDesigner
short-circuit for the former and falls back to the default pipeline for prose
or legal labels.

Use `native_rules_router` when you want the same routing shape without
DataDesigner orchestration. It uses the resolved native runtime endpoint/model
from `native_runtime` or the standard benchmark runtime environment variables.
Treat it as a native-executor prototype: it can prove that DataDesigner overhead
is avoidable, but it must be compared against baseline signatures and
original-value leak metrics before any workload-specific promotion decision.

Use `native_candidate_validate_no_augment` when you want a narrower native
executor diagnostic: direct seed candidates plus direct validation, with no
augmentation. It is useful for proving how much speed comes from removing a
phase, but a faster run that loses baseline signatures is still a rejection.

Use `detector_native_validate_no_augment` when you want to keep the production
detector seed while testing a direct-provider validation path. It is not a
no-DataDesigner strategy because the detector seed still runs through the
adapter, but it tells you whether DataDesigner validation/augmentation is the
load-bearing part of a workload. The native validation shim preserves
`date_of_birth` over broader `date` reclassifications only when the local
candidate context contains birth/DOB language; generic filing or event dates can
still be reclassified to `date`.

Use `detector_native_validate_native_augment` for the same detector-seed
question when augmentation recall is expected to be load-bearing. This arm still
uses DataDesigner for the detector seed, but direct provider calls own both
validation and augmentation.

Use `gliner_native_validate_no_augment` or
`gliner_native_validate_native_augment` when the question is specifically
"what if GLiNER did not run through DataDesigner?" These strategies use the
staged direct executor's GLiNER seed client using
`native_runtime.gliner_endpoint`, `native_runtime.gliner_model`, or the standard
GLiNER runtime environment variables; the API key env var defaults to
`NVIDIA_API_KEY`. The no-augmentation arm is a lower-cost boundary; the
native-augmentation arm is the quality-oriented no-DataDesigner candidate. The
integrated benchmark strategies execute staged direct rows with bounded
parallelism so GLiNER and native validation/augmentation latency is not
serialized across records. These arms also normalize direct GLiNER `date` seeds
to `date_of_birth` only when the local seed context contains birth/DOB language.
Generic filing or event dates remain `date`. Both arms still need repeated
signature, leak, label-mismatch, and reliability gates before any
workload-specific promotion.

Use `native_single_pass`, `native_single_pass_recall`,
`native_single_pass_values`, or `native_single_pass_values_recall` for the more
aggressive "collapse detection to one call" experiment. The first pair asks the
model for `start`/`end` offsets and validates them before falling back to exact
value matching. The value-only pair uses the standalone direct-probe prompt and
lets local code recover spans from exact returned values. Recall variants spend
more prompt tokens on label examples and high-recall guidance. All one-call
variants are expected to be faster than staged native detection when the prompt
works, but they are also more parser- and recall-sensitive. Any malformed JSON
response becomes a failed case in analysis, and any missed baseline signature
should be treated as a rejection rather than a latency win.

Replacement-map generation has a separate benchmark-only knob:

```yaml
configs:
  - id: structured-local-substitute
    experimental_detection_strategy: rules_covered_or_default
    experimental_replacement_strategy: local_structured_substitute
    detect:
      entity_labels: [api_key, email, password, url]
    replace:
      strategy: substitute
```

Supported replacement strategy values:

- `default`: run normal replacement behavior. `Substitute` uses the configured
  `replacement_generator` role through DataDesigner.
- `local_structured_substitute`: for `replace: substitute`, build deterministic
  synthetic replacement maps locally for supported structured labels. Text
  replacement still uses Anonymizer's normal replacement-map application code.

`local_structured_substitute` requires `replace: substitute` and explicit
`detect.entity_labels`. Every label must be one of the structured substitute
labels: `api_key`, `date_of_birth`, `email`, `organization_name`, `password`,
`http_cookie`, `pin`, `religious_belief`, `street_address`, `unique_id`, `url`,
or `user_name`. The preflight rejects contextual labels such as `person`. This
is deliberate. The local substitute map generator does not understand names,
social relations, cultural consistency, or prose semantics; use the default
DataDesigner-backed `Substitute` path for those workloads.

The runner refuses to write into a non-empty output directory unless
`--overwrite` is set. By default it also exports Parquet tables into
`tables/`; pass `--no-export` when you only want the raw measurement JSONL.
Before starting a real run, the benchmark runner performs cheap preflight
checks: suite/config parsing, local dataset existence, CSV/Parquet text-column
metadata, provider YAML shape, native runtime requirements, and active
model-alias references. `--dry-run` runs those same checks, expands the planned
matrix, and skips output-dir writes and model work.

For debugging DataDesigner calls, pass `--dd-trace last-message` or
`--dd-trace all-messages`. Trace records are written separately from sanitized
measurements, under `traces/{case_id}.jsonl` by default. Use `--trace-dir` to
choose another directory. `last-message` stores only the final prompt message
for each DataDesigner model call; `all-messages` stores the full message list.

DataDesigner traces may contain raw input text, prompts, model outputs, entity
values, replacement values, secrets, and PII. Treat them as debug artifacts:
keep them out of shared benchmark bundles unless they have been reviewed or
redacted.

To summarize traced calls without copying raw prompts or responses into the
analysis output, run:

```bash
uv run python tools/measurement/analyze_dd_traces.py \
  benchmark-runs/suite-id/traces \
  --output benchmark-runs/suite-id/trace-analysis \
  --format csv
```

This writes `trace_analysis.*` and `trace_group_analysis.*`. The row table
captures run tags, workflow/model metadata, status, elapsed time, prompt and
response lengths, token counts, and response-shape flags such as `raw_json`,
`fenced_json`, `embedded_json`, `text`, and `none`. The grouped table rolls those
fields up by workload, config, workflow, model, provider, status, error type,
and response shape. Use this when diagnosing local provider behavior, parser
compatibility, unexpected thinking text, or retry-heavy workflows.

Some OpenAI-compatible local endpoints return raw JSON when their model config
uses `response_format: {type: json_object}`. DataDesigner structured recipes
currently prompt for markdown-fenced JSON, so those raw JSON responses can be
valid but still fail parsing. Set top-level `dd_parser_compat: raw_json` when a
benchmark suite needs this provider compatibility mode:

```yaml
dd_parser_compat: raw_json
```

This is benchmark-only behavior. The runner patches DataDesigner structured
parser builders for the duration of a case, restores them afterward, and records
the mode in `run_tags.dd_parser_compat`. The fallback accepts either pure raw
JSON or a JSON object/array embedded after model reasoning text, then still
validates the extracted object against the requested schema. Keep the default
`none` unless a local provider or vLLM endpoint needs raw-JSON structured-output
compatibility.

## DD-Free Direct Detection Probe

Use `direct_detection_probe.py` to test a deliberately DD-free extraction path
against an OpenAI-compatible endpoint. This is a benchmark-only diagnostic: it
does not call DataDesigner, does not run GLiNER, and does not execute the
production detection graph. It sends one direct chat-completions request per
input row, then reuses Anonymizer's existing span postprocessing, occurrence
expansion, overlap resolution, and entity signature logic so results can be
compared against normal detection artifacts.

Pass `--endpoint` and `--model`, or set `ANONYMIZER_BENCH_NATIVE_ENDPOINT` and
`ANONYMIZER_BENCH_NATIVE_MODEL`.

Example biography probe:

```bash
uv run python tools/measurement/direct_detection_probe.py \
  docs/data/NVIDIA_synthetic_biographies.csv \
  --text-column biography \
  --labels age,city,company_name,degree,education_level,field_of_study,first_name,language,last_name,occupation,organization_name,place_name,political_view,race_ethnicity,religious_belief,state,university \
  --endpoint http://your-openai-compatible-endpoint/v1 \
  --model your-model-id \
  --baseline-artifacts "$BASELINE_ARTIFACTS" \
  --output /tmp/direct-detection-probe-biography \
  --overwrite \
  --json
```

Example legal probe:

```bash
uv run python tools/measurement/direct_detection_probe.py \
  docs/data/TAB_legal_sample25.csv \
  --text-column text \
  --labels application_number,city,country,date,date_of_birth,nationality,person \
  --endpoint http://your-openai-compatible-endpoint/v1 \
  --model your-model-id \
  --baseline-artifacts "$BASELINE_ARTIFACTS" \
  --output /tmp/direct-detection-probe-legal \
  --overwrite \
  --json
```

The tool writes `direct-detection-cases.jsonl`,
`direct-detection-artifacts.jsonl`, and `summary.json`. Case rows include model
usage, elapsed time, raw/allowed suggestion counts, final label counts, final
signature hashes, and optional baseline comparison counts. Artifact rows use
the same opaque signature fields as `analyze_detection_artifacts.py` and omit
raw entity values. For baseline comparison, pass a per-case sidecar or another
artifact file with one row per `row_index`; duplicate row indexes are rejected
to avoid ambiguous comparisons. Treat the probe `summary.json` as a sensitive
debug artifact because it records the resolved endpoint/model runtime used for
the probe.
so a combined multi-case artifact cannot silently select the wrong baseline.

When this probe shape is promising, move it into a normal benchmark suite with
`experimental_detection_strategy: native_single_pass_values` or
`native_single_pass_values_recall`. Those strategies use the same value-only
prompt family but run through `run_benchmarks.py`, measurement collection, case
retries, artifact capture, and pairwise strategy comparison.

Interpret this probe as a lower-friction model-call experiment, not a safe
replacement for detection. A local one-row smoke against
`nvidia/nemotron-3-super` with vLLM JSON mode and thinking disabled produced:

- Biography: 4.1s, 906 total tokens, 19 final signatures, 18/22 baseline
  signatures shared; misses included `field_of_study` and `place_name`.
- Legal: 4.9s, 1,308 total tokens, 21 final signatures, 19/22 baseline
  signatures shared; misses included `date`, `date_of_birth`, and
  `nationality`.

That result makes a DD-free native executor worth exploring, but only if it
preserves the production safety decomposition (`GLiNER/rules -> validate ->
augment -> finalize`). The one-shot direct prompt is useful as a speed/quality
boundary, not as a production candidate.

## DD-Free Staged Detection Probe

Use `staged_detection_probe.py` to test a more conservative DD-free route. This
probe still avoids DataDesigner, but it does not collapse detection into one
model response. It can run direct LLM seed extraction, direct GLiNER seeding,
deterministic rule seeding, trusted deterministic rule seeding, or rule-routed
DD-free execution. It then runs direct validation and direct augmentation unless
trusted rules or the rule router short-circuit are selected, where rule spans
bypass validation. It reuses Anonymizer's existing row-level postprocessing
helpers for validation application, augmentation merge, occurrence expansion,
overlap resolution, and artifact signatures.

Example biography probe:

```bash
uv run python tools/measurement/staged_detection_probe.py \
  docs/data/NVIDIA_synthetic_biographies.csv \
  --text-column biography \
  --labels age,city,company_name,degree,education_level,field_of_study,first_name,language,last_name,occupation,organization_name,place_name,political_view,race_ethnicity,religious_belief,state,university \
  --endpoint http://your-openai-compatible-endpoint/v1 \
  --model your-model-id \
  --baseline-artifacts "$BASELINE_ARTIFACTS" \
  --output /tmp/staged-detection-probe-biography \
  --overwrite \
  --json
```

Example legal probe:

```bash
uv run python tools/measurement/staged_detection_probe.py \
  docs/data/TAB_legal_sample25.csv \
  --text-column text \
  --labels application_number,city,country,date,date_of_birth,nationality,person \
  --endpoint http://your-openai-compatible-endpoint/v1 \
  --model your-model-id \
  --baseline-artifacts "$BASELINE_ARTIFACTS" \
  --output /tmp/staged-detection-probe-legal \
  --overwrite \
  --json
```

To replace the LLM seed phase with a direct GLiNER call, add
`--seed-source gliner` plus `--gliner-endpoint` and `--gliner-model`, or set
`ANONYMIZER_BENCH_GLINER_ENDPOINT` and `ANONYMIZER_BENCH_GLINER_MODEL`. The
probe reads the GLiNER API key from `--gliner-api-key-env`, which defaults to
`NVIDIA_API_KEY`.

To replace the LLM seed phase with deterministic local rules, add
`--seed-source rules`. This still sends rule candidates through the validator.
Use `--seed-source rules-trusted` to bypass validation for high-confidence rule
spans and run only augmentation afterward. The trusted mode is a diagnostic for
rule-covered workloads; it is not a general prose/legal safety default.
Use `--seed-source rules-plus-direct-llm` to add deterministic rule spans to
direct LLM seed spans while validating only the direct LLM seed candidates. This
tests a mixed native path where obvious structured secrets are trusted locally
without giving up contextual model seeding for the rest of the record.
Use `--seed-source rules-router` to make that split explicit: if every requested
label is supported by deterministic rules, the probe runs trusted local rules
with no model calls; otherwise it falls back to `rules-plus-direct-llm`.
When the requested labels are all covered by deterministic rules, add
`--skip-augmentation-when-rule-covered` to measure a fully local short-circuit
with no model calls.
Use `--skip-augmentation` to disable augmentation for any seed source. This is
only a diagnostic for measuring how much recall the augmentation phase carries;
signature loss should reject the candidate even when latency improves.

To test whether direct validation can preserve the phase boundary with less
prompt text, add `--validation-prompt-mode chunked-excerpt`. This splits seed
validation candidates into chunks of `--validation-max-entities-per-call` and
sends each chunk with a tagged local excerpt bounded by
`--validation-excerpt-window-chars`. The default remains `full-text`, which
keeps the prior one-call behavior. Treat this as a request-count/token tradeoff:
chunked excerpts can reduce prompt payload, but they also create more validator
requests and can remove context needed for labels such as legal roles,
education, demographics, or prose locations.

The tool writes `staged-detection-cases.jsonl`,
`staged-detection-artifacts.jsonl`, and `summary.json`. Case rows include
per-phase usage for seed extraction, validation, and augmentation, true case
wall time in `elapsed_sec`, model-call time in `model_elapsed_sec`, plus
`phase_model_work`, `phase_skip_reasons`, `phase_model_requests`,
`model_phase_count`, `model_request_count`, total usage, and optional baseline
signature deltas. Use these fields to distinguish local work, provider latency,
and a provider that returned no token accounting.
Treat the staged probe `summary.json` as a sensitive debug artifact because it
records the resolved endpoint/model runtime used for the probe.
For example, a fully local rule-covered run should show `model_phase_count: 0`,
`model_request_count: 0`, `rule_covered_label_set: true`, and
`phase_skip_reasons.augmentation: "rule_covered_labels"`; `elapsed_sec` should
still capture the local rule/postprocess wall time while `model_elapsed_sec`
remains `0.0`. A chunked-excerpt validation run should usually keep
`model_phase_count` unchanged while raising `phase_model_requests.validation`.

To summarize those staged probe outputs without hand-written `jq`, run:

```bash
uv run python tools/measurement/analyze_staged_detection_output.py \
  /tmp/staged-detection-probe-biography \
  --output /tmp/staged-detection-probe-biography/analysis \
  --format csv
```

The analyzer accepts either the staged output directory or the
`staged-detection-cases.jsonl` file directly. It writes per-case, per-seed
source group, and label-delta tables. Use `group_analysis.csv` for latency,
token, request, and signature-overlap totals; use `label_delta_analysis.csv` to
see which labels account for baseline-only misses or direct-only additions. The
analysis tables still omit raw text and raw entity values.

The grouped table also includes a conservative `fast_lane_verdict`:

- `fast_lane_candidate`: every case completed, every case was fully
  rule-covered, the seed-source group has at least three cases, model requests
  were zero, and baseline comparison found no missing signatures.
- `reject`: at least one case errored or the candidate lost any baseline
  signature.
- `review`: baseline comparison is missing, fewer than three cases were
  analyzed, the candidate still used model calls, or not every case was fully
  rule-covered.

Use `fast_lane_candidate` only as a workload-scoped promotion signal. It does
not prove that the same no-DataDesigner path is safe for prose/legal labels or
for data shapes outside the sampled suite.

A refreshed local one-row smoke against `nvidia/nemotron-3-super` with vLLM JSON
mode and thinking disabled produced:

- Biography: 13.7s, 4,550 total tokens, 24 final signatures, 20/22 baseline
  signatures shared. The staged path recovered two signatures missed by the
  one-shot direct probe, but still missed an `age` and a `place_name` signature
  and added four direct-only signatures.
- Legal: 17.5s, 6,425 total tokens, 21 final signatures, 19/22 baseline
  signatures shared. This did not improve signature overlap over the one-shot
  direct probe and was materially slower.

A direct hosted GLiNER seed smoke reached NVIDIA's endpoint but failed before
local validation with `DEGRADED function cannot be invoked` for
`nvidia/gliner-pii`. Keep the `--seed-source gliner` mode as a native executor
option, but do not treat hosted GLiNER availability as stable for local
performance conclusions.

Rules seeding changed the tradeoff. On biography row 0, `rules` took 6.1s and
1,565 tokens but shared only 17/22 baseline signatures; `rules-trusted` took
5.2s and 1,019 tokens and shared 18/22. On legal row 0, `rules` took 7.1s and
2,213 tokens with 20/22 shared signatures; `rules-trusted` took 6.4s and 1,431
tokens with the same 20/22 shared signatures. On the three-row shell-secrets
slice, `rules` exposed a validation regression: the validator reclassified a
database URL as a password, leaving row 1 with 2/3 shared baseline signatures.
`rules-trusted` preserved all shell baseline signatures and reduced each row to
one augmentation call, but that no-op augmentation still consumed 398-533 tokens
per row. With `--skip-augmentation-when-rule-covered`, the same trusted-rules
shell run preserved all 12 baseline signatures with zero model usage. Use this
as evidence for a native executor with rule-covered short circuiting, not as
evidence that trusted rules are safe for arbitrary text.

Interpret this as evidence for native orchestration, not as a ready strategy.
The staged shape is closer to Anonymizer's safety model than one-shot
extraction, but the naive direct prompts spend too many tokens. The next useful
experiment is a native executor that preserves the same phase boundaries while
using compact production-equivalent prompts, direct provider clients, and a
cheap deterministic or detector-backed seed phase instead of LLM-seeded
extraction.

## No-DataDesigner Strategy Pivot

The strongest current performance signal comes from not invoking
DataDesigner at all for records whose requested labels and text shape are
covered by deterministic structured-secret extractors. On a local shell/structured-secret slice,
the staged `rules-router` path preserved every compared baseline signature with
zero model requests and millisecond-level elapsed time. In full Anonymizer
benchmarks, `rules_covered_or_default` plus `local_structured_substitute`
reduced structured substitute workloads by 38-99% wall time and removed most or
all observed model tokens, depending on whether the run still fell back to
default detection.

The benchmark harness now has several integrated native strategies for that
next experiment. `native_rules_router` reuses the staged DD-free executor inside
Anonymizer's detection workflow, so benchmark cases still exercise the normal
replacement and measurement plumbing. `native_candidate_validate_no_augment`
removes augmentation to isolate the recall cost of that phase.
`detector_native_validate_no_augment` keeps the default detector seed and
switches only validation to direct provider calls. `native_single_pass` is the
more radical variant: it asks the local provider for all spans in one JSON
response and then lets Anonymizer validate offsets and finalize entities
locally. Use these arms to compare native provider calls against the
DataDesigner-backed `default` strategy on the same workloads.

Treat that as a workload router, not a global replacement. The same DD-free
direct LLM approach on biography and legal prose still lost roughly a quarter
to a third of baseline signatures in repeated local probes, even though it
avoided DataDesigner. That is not an anonymization-safe trade by itself. The
current evidence points to three separate lanes:

- **Structured fast lane:** if the explicit labels are all deterministic-rule
  labels and rule extraction covers the workload, skip DataDesigner, skip model
  calls, and use local redact/hash/substitute. This is the most promising path
  for shell history, secrets, config files, audit logs, and similarly keyed
  records.
- **Native model lane:** for prose or mixed records, preserve the production
  detection decomposition but call providers directly: seed, validate, augment,
  finalize. The prototype exists as `staged_detection_probe.py`, and the
  benchmark harness includes detector-seeded and native-seeded variants, but
  their current prompts are still research prompts and are too lossy/costly to
  promote.
- **Single-pass model lane:** for a sharper boundary test, collapse prose or
  mixed detection into one direct JSON span extraction call. This only becomes
  interesting if it preserves baseline signatures; parser errors, invalid
  offsets, or missed signatures should send the workload back to the default
  pipeline.
- **Safety fallback:** route unsupported labels, uncertain text shapes, direct
  parser failures, and signature-loss evidence back to the normal
  DataDesigner-backed pipeline until a native executor proves equal or better
  recall on repeated workload-specific comparisons.

This changes the performance strategy from "make every DataDesigner phase
faster" to "avoid DataDesigner when the safety case is trivial, and use
DataDesigner as the fallback for hard cases." The benchmark interpretation
should therefore privilege signature coverage, original-value leak checks,
source provenance, and reliability flags over raw latency wins. A no-DD result
that is faster but loses baseline signatures remains a reject; a no-DD result
that is fully rule-covered, leak-free, and stable across repetitions is a
candidate for a production fast lane.

## Output Layout

A benchmark run writes one raw measurement file per case, then combines them:

```text
benchmark-runs/suite-id/
  raw/
    inputs/
      biographies__redact-default__r000.csv
    biographies__redact-default__r000.jsonl
    biographies__redact-default__r000.detection-artifacts.jsonl
    support__hash-agent-labels__r000.jsonl
  artifacts/
    biographies__redact-default__r000/
  traces/
    biographies__redact-default__r000.jsonl
  measurements.jsonl
  summary.json
  detection-artifacts.jsonl
  tables/
    manifest.json
    run.parquet
    stage.parquet
    record.parquet
    ndd_workflow.parquet
```

Raw per-case JSONL files are streamed as measurement events are recorded, so a
long run leaves inspectable partial output before the case exits. The combined
`measurements.jsonl` is written after the completed and errored case files are
collected.

Use `summary.json` to inspect case status, retry attempts, and errors. If a
case succeeds after retry, the combined `measurements.jsonl` contains the final
successful attempt while `summary.json` preserves the earlier failure messages.
Use `measurements.jsonl` when you need the original structured records. Use
`tables/` for analysis.
Use `traces/` only when `--dd-trace` was enabled and you need raw
DataDesigner message-level debugging.

Treat `summary.json`, `raw/inputs/`, `artifacts/`,
`raw/*.detection-artifacts.jsonl`, and `traces/` as sensitive outputs. They can
contain source text, entity values, replacement values, prompts, model
responses, exception messages, or other PII-bearing debug data. The exported
measurement tables and detection signature ids are designed for analysis
without raw values, but debug sidecars are not sanitized bundles.

Detection workflow artifacts can be analyzed separately when you need to know
whether augmentation helped or only added cost. `run_benchmarks.py` writes
`detection-artifacts.jsonl` automatically when export is enabled and detection
artifacts are present. The automatic export analyzes each case immediately after
it runs, then combines per-case sidecars from `raw/`; rows include `suite_id`,
`workload_id`, `config_id`, `repetition`, `case_id`, and `run_id` so they can be
joined to `measurements.jsonl` and exported tables. `rules_only` cases do not
produce DataDesigner parquet artifacts, so the runner writes a synthetic
rules-only sidecar from the same deterministic rules. That sidecar includes
counts, source=`rule`, and opaque entity signatures, but not raw entity values.
Routed strategies whose final entity set can differ from raw DataDesigner
artifacts, including row-aware `rules_covered_or_default`, write sidecars from
the final trace dataframe so rule-routed and fallback-routed rows are both
represented.

Row-aware routed strategies also emit sanitized route telemetry into
`measurements.jsonl`, and `analyze_benchmark_output.py` surfaces it in
`case_analysis.*` and `group_analysis.*`. Use `route_total_row_count`,
`route_rule_row_count`, and `route_fallback_row_count` to confirm how many rows
used the zero-model rules lane versus the normal detection fallback before
interpreting request, token, or latency deltas.
You can also run the analyzer by hand against an artifact directory:

```bash
uv run python tools/measurement/analyze_detection_artifacts.py \
  benchmark-runs/suite-id/artifacts \
  --output benchmark-runs/suite-id/detection-artifacts.jsonl
```

The analyzer reads `entity-detection*` parquet artifacts and emits one row per
artifact row. It reports seed, augmentation, and final entity counts; duplicate
augmentation suggestions; new augmented values that survived into final
entities; final label/source counts; and weak `api_key` shape warnings. The
output intentionally omits raw entity values.

Use this alongside the exported measurement tables when comparing
`default` against `no_augment`:

- High `augmented_duplicate_seed_value_count` with low
  `augmented_new_final_value_count` means augmentation probably added cost
  without improving that case.
- High `augmented_new_final_value_count` means augmentation found spans that
  the detector+validator path missed.
- High `weak_api_key_shape_count` usually means the label set is mismatched to
  the workload. For example, legal prose constrained to
  `[person, email, api_key, password]` can force dates or case identifiers into
  `api_key` because better prose labels are unavailable.

For a ready-made case and grouped summary that joins `measurements.jsonl` with
`detection-artifacts.jsonl`, use:

```bash
uv run python tools/measurement/analyze_benchmark_output.py \
  benchmark-runs/suite-id \
  --output benchmark-runs/suite-id/analysis \
  --format csv
```

By default this joins `benchmark-runs/suite-id/measurements.jsonl` with
`benchmark-runs/suite-id/detection-artifacts.jsonl`. To use a refreshed or
relocated sidecar that still contains benchmark case metadata, pass it
explicitly:

```bash
uv run python tools/measurement/analyze_benchmark_output.py \
  benchmark-runs/suite-id \
  --detection-artifacts benchmark-runs/suite-id/current-analysis/detection-artifacts.jsonl \
  --output benchmark-runs/suite-id/current-analysis \
  --format csv
```

The override sidecar must include `case_id` or `run_id` values that match the
measurement rows. A raw artifact scan produced from only the DataDesigner
parquet directory can summarize detection artifacts, but it cannot be safely
joined to benchmark measurements unless benchmark case metadata is preserved.

This writes `case_analysis.*`, `group_analysis.*`, `model_analysis.*`, and
`model_group_analysis.*`. It keeps fully local cases with no model workflow
rows, such as rule-covered `rules_only` or `native_rules_router` cases, in the
comparison with zero observed requests/tokens. Native direct-call strategies
that bypass DataDesigner write `model_workflow` rows, so their provider request
and token counts still contribute to case, group, and model summaries. When the
benchmark was run with current sidecar export, `rules_only` also has
artifact-derived signatures and source counts; older runs may only have
record-level entity counts. The joined case/group tables include
successful/failed request counts, input/output token splits, record counts,
dataset input-token throughput, `seed_validation_candidate_count`,
`estimated_seed_validation_chunk_count`, and `observed_failed_request_rate`;
use these when testing
`detect.validation_max_entities_per_call` so you can distinguish a real chunk
count change from provider retry variance. The model tables split the same
usage by `workflow_name` and `model_name`, which is useful for separating local
detector cost from validator, augmenter, substitute, or rewrite model cost.
When record-level measurements include ground-truth entities, the joined tables
also expose exact and relaxed entity-quality metrics. The relaxed metrics count
span overlap, with small label-equivalence groups for common aliases such as
`user_name` / `username` and `api_key` / `auth_token`. Case and group tables
also count empty detections, including empty records that had ground-truth
entities. If your suite adds portable topology tags such as `endpoint_count`,
`gpu_count`, or `tensor_parallelism`, the analysis computes per-endpoint and
per-GPU input-token throughput; otherwise those normalized fields remain null.
The case/group tables also surface incomplete benchmark cases with
`case_failed`, `error_stage_count`, `error_ndd_workflow_count`,
`error_model_workflow_count`, `failed_case_count`, and `failed_case_rate`.
Check these before interpreting a fast candidate as a safe improvement; a
failed repetition can otherwise look like entity instability or a latency win.
The joined case/group tables also expose final entity source counts from
detection artifacts, including `artifact_final_detector_entity_count`,
`artifact_final_rule_entity_count`, and
`artifact_final_augmenter_entity_count`. Use these to verify whether a faster
strategy is still relying on contextual detector/validator spans, or whether it
has shifted a workload entirely onto deterministic rules.
They also include `artifact_final_entity_signature_count` and
`artifact_final_entity_signature_hashes`, which are opaque per-row identifiers
derived from the final entity label and offsets. They do not include raw or
normalized entity values. The companion
`artifact_final_entity_signature_labels` field maps each opaque hash to its
entity label. These fields do not expose raw entity values, but they let
analysis tools detect when two configs report the same entity count while
protecting different spans.

To compare a baseline and candidate strategy across common workloads, use:

```bash
uv run python tools/measurement/compare_strategy_pairs.py \
  benchmark-runs/suite-id/analysis/case_analysis.csv \
  --baseline-strategy no_augment \
  --candidate-strategy rules_filter_guardrail_no_augment \
  --output benchmark-runs/suite-id/analysis/strategy_comparison.csv
```

If the candidate was run in a separate benchmark directory, pass a second case
analysis file:

```bash
uv run python tools/measurement/compare_strategy_pairs.py \
  benchmark-runs/baseline-suite/analysis/case_analysis.csv \
  --candidate-case-analysis benchmark-runs/candidate-suite/analysis/case_analysis.csv \
  --baseline-strategy no_augment \
  --candidate-strategy rules_guardrail_no_augment
```

The comparison reports latency, request, token, entity-count, validation
candidate-count, augmentation-count, final source-count, and opaque
entity-signature deltas. It also reports original-value leak deltas from
`original_value_leak_count` and `original_value_leak_record_count`. The
`augmented_entity_count_delta` and
`augmented_new_final_value_count_delta` columns are especially useful for
no-augmentation and model-routing ablations: a faster candidate that removes
new final values from augmentation needs signature checks before promotion.
When signature labels are available, it also reports label counts for
baseline-only, candidate-only, and shared signatures. For repeated selector
runs, it also compares signatures that are stable across every repetition,
which catches cases where a candidate finds a sensitive span only
intermittently. It adds conservative flags such as
`baseline_case_failures`, `candidate_case_failures`, `entity_count_loss`,
`entity_signature_loss`, `span_boundary_mismatch`,
`covered_label_mismatch`,
`candidate_original_value_leak`,
`candidate_replacement_missing_final_entity`,
`candidate_duplicate_synthetic_replacement`,
`failed_request_increase`, `bridge_fallback_increase`,
`stable_entity_signature_loss`, `no_candidate_detector_entities`,
`candidate_uses_rule_entities`, `candidate_skips_llm_validation`, and
`replacement_only_detection_instability`, plus five verdict fields:

- `value_protection_verdict`: `pass`, `review`, or `fail`. This axis focuses on
  whether the candidate still protects the sensitive values. Candidate case
  failures, candidate original-value leaks, missing replacement-map entries,
  replacement collisions, and uncovered baseline signatures fail. Rule
  provenance, validation skipping,
  provider retry pressure, and covered boundary or label mismatches do not fail
  this axis by themselves; they are represented in the semantic and overall
  safety verdicts.
- `signature_parity_verdict`: `pass`, `review`, or `fail`. This axis focuses on
  exact baseline signature semantics. Covered label or boundary mismatches stay
  review-gated even when `value_protection_verdict` passes.
- `safety_verdict`: `pass`, `review`, or `fail`. Candidate case failures and
  entity/signature loss fail. Candidate original-value leaks also fail, even
  when entity signatures match. Baseline case failures, baseline
  original-value leaks, rule-only, rule-heavy, or validation-skipping
  candidates require review. Candidate provider failed-request increases or
  bridge-fallback increases also require review: they are reliability signals,
  not anonymization leaks.
- `performance_verdict`: `improved`, `mixed`, `regressed`, `unchanged`, or
  `unknown`, based on available latency, request, and token deltas.
- `candidate_verdict`: `candidate_viable`, `review`, or `reject`. A candidate
  is viable only when safety passes and measured performance improves.

Use verdicts for triage, then inspect the underlying flags and label-count
deltas before promoting a strategy beyond benchmark experiments.
For replacement-only comparisons where the detection strategy is unchanged,
`replacement_only_detection_instability` means the candidate and baseline were
still run through independent detection passes and their detection artifacts
drifted. Treat that as a prompt to consult fixed-trace replacement replay before
blaming or promoting the replacement-map backend.
In fixed-trace replacement replay,
`candidate_duplicate_synthetic_replacement` means the local replacement backend
protected every original value but collapsed at least two replacements in the
same row to the same synthetic value. That is review-gated as a substitute
quality and relational-consistency concern rather than treated as an immediate
privacy leak.
When the replay CSV contains
`candidate_covers_baseline_replacement_missing_final_entity`,
`candidate_covers_baseline_original_value_leak`, or
`candidate_covers_baseline_replacement_synthetic_original_collision`, the
candidate removed a defect observed in the DataDesigner-backed substitute arm
on the same fixed detection trace. In that case `value_protection_verdict` can
pass while `signature_parity_verdict` remains review-gated, because the
candidate covered more of the final-entity set than the flawed baseline.

For `rules_covered_or_default`, compare rule-covered configs by config ID so
the zero-model lane is checked against the same explicit label set:

```bash
uv run python tools/measurement/compare_strategy_pairs.py \
  benchmark-runs/suite-id/analysis/case_analysis.csv \
  --baseline-config rule-labels-default \
  --candidate-config rule-labels-covered-or-default \
  --output benchmark-runs/suite-id/analysis/rules-covered-comparison.csv
```

Promote the fast path only when
`baseline_only_candidate_uncovered_signature_count` is zero on the target
workload, `candidate_original_value_leak_count` is zero, `candidate_verdict` is
at least `review`, and the review flags are expected rule fast-lane flags such
as `candidate_uses_rule_entities`, `no_candidate_detector_entities`,
`entity_count_loss`, or `span_boundary_mismatch`. Exact
`baseline_only_final_entity_signature_count` can be nonzero when a candidate
protects the same sensitive value with a wider or slightly narrower keyed span;
use the covered/overlapping/uncovered columns to decide whether that is an
acceptable workload policy. A run that has uncovered signatures or leaks
original detected values should reject:
in the June 8, 2026 sudo-password smoke run, the pre-fix comparison rejected the
candidate with `lost_labels=password:1`; after the narrow sudo rule was added,
the same comparison had no baseline-only signatures and remained review-gated
only because the final spans were rule-sourced.

The command output also includes a rollup summary with verdict counts and the
workloads in each candidate-verdict bucket, which is useful for repeated runs
over larger suites.

To screen many comparison CSVs from one or more benchmark directories, use:

```bash
uv run python tools/measurement/screen_strategy_comparisons.py \
  benchmark-runs/ \
  --output benchmark-runs/strategy-screen.csv \
  --group-output benchmark-runs/strategy-groups.csv
```

When screening a scratch directory that contains older analysis outputs, filter
by source-path fragments:

```bash
uv run python tools/measurement/screen_strategy_comparisons.py \
  /tmp/anonymizer-benchmark-scratch \
  --source-include analysis-current-csv \
  --source-include analysis-failure-aware-csv \
  --output current-strategy-screen.csv \
  --group-output current-strategy-groups.csv
```

Use `--source-exclude` to omit known stale or exploratory subdirectories.
For example, if a scratch directory contains a pre-fix comparison and a rerun,
screen only current evidence by excluding the stale source-path fragment:

```bash
uv run python tools/measurement/screen_strategy_comparisons.py \
  /tmp/anonymizer-perf-goal \
  --source-include comparison \
  --source-exclude before-sudo \
  --source-exclude structured-secrets-varied-comparison.csv \
  --output /tmp/anonymizer-perf-goal/strategy-screen-current.csv \
  --group-output /tmp/anonymizer-perf-goal/strategy-screen-current-groups.csv
```

The screen walks CSV files recursively, ignores non-comparison tables such as
`case_analysis.csv` and `group_analysis.csv`, and combines rows produced by
`compare_strategy_pairs.py`. It deduplicates exact repeated rows from copied
analysis directories, then sorts viable candidates first, then review and reject
rows, preserving latency/token deltas, flags, lost-label summaries, and
augmentation deltas. It also preserves baseline/candidate case counts,
baseline/candidate detection strategies, baseline/candidate replacement
strategies, stable-signature evidence counts, and candidate original-value leak
counts and labels. For DataDesigner-free experiments, it also preserves
`value_protection_verdict`, `signature_parity_verdict`, and label-mismatch
label counts, so one-off candidate rows are visible as weak evidence even before
opening the source comparison CSV. This is the quickest way to check whether a
benchmark directory contains any candidate worth rerunning on a larger workload
slice.

Use the `evidence_level` column to separate current safety evidence from older
or weaker comparison rows. `split_verdicts` means the row has separate value
protection and signature-parity verdicts, `stable_signatures` means it has
stable-signature counts but not split verdicts, `signature_counts` means it only
has raw signature counts, and `legacy` means the screen can only use the older
aggregate verdict columns. The group output includes `evidence_level_counts` so
mixed scratch directories do not make a legacy row look as strong as a current
split-verdict rerun.

The optional group output aggregates rows by candidate strategy when the
candidate used a non-default experimental strategy, or by candidate config
otherwise. This keeps ordinary config experiments, such as model routing or
prompt-parameter changes, from being collapsed under `strategy:default`. When
the same experiment used multiple config IDs, pass a JSON alias map:

```json
{
  "biography-hybrid-augment-temp07": "biography-temp07-routing",
  "biography-augment-temp07": "biography-temp07-routing"
}
```

```bash
uv run python tools/measurement/screen_strategy_comparisons.py \
  benchmark-runs/ \
  --group-by strategy_workload_family \
  --config-aliases config-aliases.json \
  --group-output benchmark-runs/strategy-family-groups.csv
```

Aliases only affect default-strategy, default-replacement config grouping.
Non-default experimental detection strategies still group by strategy; when a
candidate also uses a non-default replacement strategy, the group key appends
`replacement:<strategy>`. If detection is default and only replacement changes,
the group key is `replacement:<strategy>`. Use the group output to find
candidates with conflicting evidence, such as a no-augmentation candidate that
passes one slice and rejects on another. The
group table includes both best and worst latency, token, and request deltas so a
single fast slice does not hide a slower or unsafe repeat. It also includes
minimum baseline/candidate case counts and the minimum shared stable-signature
count observed in the group, plus summed candidate original-value leak counts
and leak labels. The
`recommendation` column is deliberately conservative:
`single_slice_viable` means one viable row exists but needs repeat evidence,
`candidate_family_viable` requires two or more viable rows and no review or
reject rows, `promising_needs_review` means viable rows exist but review-gated
rows remain and at least one split-verdict row is also viable,
`needs_split_verdict_rerun` means viable-looking and review-gated rows exist but
the group has only older signature-count or stable-signature evidence, or a
review-only group mixes current split-verdict rows with older comparison rows
that should be rerun under the current verdict schema,
`needs_viable_split_verdict` means older viable rows exist and split-verdict
evidence exists, but every split-verdict row is still review- or reject-gated,
`replacement_replay_review` means an improved replacement-strategy group is
review-gated by detection artifact drift even though the detection strategy did
not change; use fixed-trace replacement replay to isolate replacement-map
behavior,
`reliability_review` means every row improved performance but one or more rows
are review-gated by provider reliability signals such as failed-request or
sync-bridge fallback increases,
`fast_lane_review` means a `rules_only` or
`rules_covered_or_default` group improved performance, had explicit zero
candidate original-value leaks, had no uncovered baseline signatures, and is
review-gated only by expected rule fast-lane provenance or span-boundary flags,
`label_policy_review` means every row improved performance, passed
`value_protection_verdict`, and was review-gated on `signature_parity_verdict`
because the candidate protected a baseline value under a different label,
`review_only` means the family has no failures, still needs manual review, and
every review-gated row is `improved`,
`review_mixed_performance`
means the family has no failures but has mixed performance evidence,
`no_performance_win` means review-gated rows exist without an improvement
signal, `reject` means no viable rows survived, and `conflicting_evidence` means
at least one viable row and at least one rejected row exist for the same
candidate family.

When a strategy's safety depends on workload shape, group by workload family:

```bash
uv run python tools/measurement/screen_strategy_comparisons.py \
  benchmark-runs/ \
  --group-by strategy_workload_family \
  --output benchmark-runs/strategy-screen.csv \
  --group-output benchmark-runs/strategy-family-groups.csv
```

This keeps evidence from families such as shell-secret command logs, legal
records, and biographies separate. Use this mode before claiming a broad
performance improvement from a strategy that may only be safe on rule-covered
secret workloads. Use `--group-by strategy_workload` for an even stricter
per-workload grouping.

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
- `model_workflow`: one row per non-DataDesigner model-backed workflow, such as
  `native_rules_router`, `native_candidate_validate_no_augment`,
  `detector_native_validate_no_augment`,
  `detector_native_validate_native_augment`, `native_single_pass`, and the
  other `native_single_pass*` strategies, with the same sanitized usage fields
  as `ndd_workflow`.

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

Most analyses join `stage`, `record`, `ndd_workflow`, and `model_workflow` back
to `run` through `run_id`, then group by run tags:

- `run_tags.suite_id`
- `run_tags.workload_id`
- `run_tags.config_id`
- `run_tags.experimental_detection_strategy`
- `run_tags.experimental_replacement_strategy`
- `run_tags.dd_parser_compat`
- `run_tags.repetition`
- `run_tags.case_id`

Prefer medians and percentiles over averages when comparing latency. LLM calls
usually have long tails, and one retry or provider stall can distort a mean.

For staged DD-free detection probes, convert the probe output first:

```bash
uv run python tools/measurement/analyze_staged_detection_output.py \
  /tmp/anonymizer-perf-goal/no-dd-rules-plus-direct-biography-r5-current \
  --output /tmp/anonymizer-perf-goal/no-dd-rules-plus-direct-biography-r5-current/analysis \
  --format csv
```

Then read `analysis/group_analysis.csv` to compare `elapsed_sec_sum`,
`model_elapsed_sec_sum`, `model_request_count_sum`, `total_tokens_sum`,
`baseline_shared_signature_rate`, and
`baseline_only_final_entity_signature_count_sum`. Use `fast_lane_verdict` as
the first gate: `reject` means stop and inspect losses before running larger
slices; `fast_lane_candidate` means the sampled workload is a plausible
zero-model rule-covered lane with repeated evidence; `review` means the output
is incomplete, has too few cases, or still uses model work. The staged analyzer
requires at least three cases in a seed-source group before a clean zero-model
run can become `fast_lane_candidate`; one-row smokes remain `review` even when
they preserve all compared signatures. Read
`analysis/label_delta_analysis.csv` when the shared-signature rate is low; it
shows which labels drove the baseline-only losses or direct-only additions.

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
workflow_group_cols = [
    "run_tags.workload_id",
    "run_tags.config_id",
    "run_tags.experimental_detection_strategy",
    "run_tags.experimental_replacement_strategy",
    "run_tags.dd_parser_compat",
    "workflow_name",
]

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

Summarize provider usage by workflow and model:

```python
model_usage = pd.read_csv("benchmark-runs/suite-id/analysis/model_group_analysis.csv")

retry_sources = (
    model_usage
    .sort_values(
        ["sum_observed_failed_requests", "sum_observed_total_tokens"],
        ascending=[False, False],
    )
    [
        [
            "workload_id",
            "config_id",
            "workflow_name",
            "model_name",
            "sum_observed_total_requests",
            "sum_observed_failed_requests",
            "observed_failed_request_rate",
            "sum_observed_total_tokens",
        ]
    ]
)

print(retry_sources)
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

## Signature Delta Review

Use `extract_signature_deltas.py` when a fast candidate has fewer, more, or
different final entity signatures than a higher-recall reference run. The tool
compares two `detection-artifacts.jsonl` files and recovers local context from
the DataDesigner artifact parquet files. Entity values are masked by default:
the output stores label, source, span offsets, value length, signature id, and a
small context window with the entity replaced by a placeholder. It does not
emit a hash derived from the raw entity value.

Example: review spans found by a text/raw-parser reference but missed by a
hybrid candidate for one workload/config pair:

```bash
uv run python tools/measurement/extract_signature_deltas.py \
  /tmp/reference/detection-artifacts.jsonl \
  /tmp/candidate/detection-artifacts.jsonl \
  --baseline-artifact-root /tmp/reference/artifacts \
  --candidate-artifact-root /tmp/candidate/artifacts \
  --baseline-config legal-default \
  --candidate-config legal-hybrid-rules-guardrail \
  --workload legal-r2 \
  --output /tmp/legal-signature-deltas.csv \
  --format csv
```

Interpretation:

- `baseline_only` rows are spans the candidate missed relative to the
  reference.
- `candidate_only` rows are spans the candidate found that the reference did
  not.
- `resolution=parquet` means the span was recovered from DataDesigner's final
  detection artifacts.
- `resolution=artifact_details` means the span was reconstructed from
  sanitized final signature details plus the artifact row's source text. This
  is common for benchmark-only strategies that patch final entities from an
  in-memory dataframe after a seed-stage artifact is written.
- `resolution=rule` means the span was reconstructed from deterministic
  rule-guardrail logic because it was added after DataDesigner wrote parquet.
- `resolution=metadata_only` means only the opaque signature metadata was
  available; use this as a signal to rerun with trace/artifact capture if the
  delta matters.

## Current Local Findings

These findings come from small local vLLM runs against
`nvidia/nemotron-3-super`; treat them as triage signals, not defaults.

| Strategy | Latest local result | Status | Implication |
| --- | --- | --- | --- |
| `rules_only` on the three-row shell-secrets slice | Preserved all 12 stable signatures; median latency moved from 7.2s to 0.004s, requests from 12 to 0, and tokens from 11,019 to 0. | Review | Viable only for bounded secret scans where every requested label is covered by deterministic rules. |
| `rules_guardrail_detector_only` on the same shell-secrets slice | Preserved stable signatures and reduced model work, but one candidate repetition failed during GLiNER health checks. | Review | Useful as a structured-secret diagnostic, but less attractive than `rules_only` when labels are fully rule-covered. |
| `rules_filter_guardrail` on the same shell-secrets slice | Retry-enabled rerun completed all 6 cases. It preserved all 12 signatures, reduced seed validation candidates from 11 to 0, median pipeline latency from 8.0s to 3.9s, requests from 12.5 to 7.0, and tokens from 10,966 to 3,647. | Review | Useful as a mixed-workload probe; keep it review-gated because final entity provenance is rule-only for this slice. |
| `rules_filter_guardrail` on a mixed biography/legal/shell probe | After changing rule filtering to preserve different-label overlaps, repeated two-row biography, one-row legal, and three-row shell runs had no stable or unstable signature loss. Median pipeline latency moved from 28.5s to 20.0s on biography, 19.4s to 18.2s on legal, and 8.7s to 6.1s on shell. | Review | Historical positive probe only; the larger five-row non-shell repeat below did not preserve this signal. |
| `rules_filter_guardrail` on offset biography/legal slices | After hardening rule filtering so only fully covered same-label spans are skipped and rule reinsertion is additive, the five-row biography offset slice had no signature loss but moved into review because requests increased slightly while tokens decreased. The richer two-row legal offset slice rejected: latency, requests, and tokens regressed and one repetition missed three `court_name` signatures while adding one rule-backed `date_of_birth`. | Mixed | The hardened strategy is safer than the first version, but it still needs per-workload gates and is not a broad legal/prose default. |
| `rules_filter_guardrail` on current five-row biography/legal repeats | Biography preserved stable and unstable signatures but regressed latency from 37.8s to 45.9s and requests from 20.5 to 22.5. Legal improved latency from 60.6s to 51.2s and tokens from 63,072 to 61,568, but lost five stable `date` signatures and made two stable `person` signatures unstable. | Reject | Do not promote this as a prose/legal default; the safety and latency tradeoff is workload-dependent and fails the legal signature gate. |
| `rules_guardrail` on a five-row legal slice | Same-suite repeated comparison against default preserved stable and unstable signatures, but latency regressed from 39.6s to 47.1s, requests rose from 20.0 to 20.5, and tokens were roughly flat at 60,998 to 60,757. | Mixed | Deterministic date guardrails can improve coverage without signature loss, but they are not a legal-prose performance win on this slice. |
| `detector_only` and `rules_guardrail_detector_only` on prose/legal slices | Faster on one-row smoke checks, but lost baseline signatures on biography and legal samples. A current detector-only isolation rerun moved biography 27.3s → 0.9s and 8,416 → 526 tokens, but lost two `first_name` and one `organization_name` signatures. Legal moved 52.0s → 1.0s and 14,095 → 1,078 tokens, kept `date_of_birth`, but still lost one `date` and one `nationality` signature while adding many extra spans. | Reject | Local finalization alone is not a safe replacement for validation and augmentation on contextual text. The legal rerun is useful diagnostically because raw detector output kept `date_of_birth`, so a later native-validation miss likely came from validation behavior rather than detector seeding. |
| One-shot DD-free direct detection on biography/legal row 0 | Biography completed in 5.1s with 902 tokens but shared only 18/22 baseline signatures. Legal completed in 5.8s with 1,308 tokens but shared only 19/22 baseline signatures. | Reject as replacement | This is a useful speed boundary and prompt experiment, but a single extraction prompt drops core detections on non-shell workloads. |
| Standalone direct-detection five-row probe | A fresh local probe compared one direct extraction call per row against the current staged direct reference. On legal, compact direct detection moved from 62.3s and 15 requests to 17.1s and 5 requests, but shared only 75/147 reference signatures and missed 72. Recall prompting improved legal to 31.1s, 109 final entities, 102 shared, and 45 missed. On biographies, compact direct detection moved from 85.7s and 15 requests to 21.2s and 5 requests, with 91/102 shared signatures, 11 missed, and no extras; recall prompting regressed to 62 shared and 40 missed. Outputs: `/tmp/anonymizer-perf-goal/direct-detection-legal-r5-compact-after-guard`, `/tmp/anonymizer-perf-goal/direct-detection-legal-r5-recall-after-guard`, `/tmp/anonymizer-perf-goal/direct-detection-biography-r5-compact-after-guard`, `/tmp/anonymizer-perf-goal/direct-detection-biography-r5-recall-after-guard`. The benchmark harness can now run this value-only prompt shape through `native_single_pass_values` and `native_single_pass_values_recall`. | Mixed diagnostic | The one-call path is the clearest lower-bound latency test, but it is not a general anonymization-safe replacement. Compact one-call extraction may deserve workload-specific follow-up for biographies; legal still needs augmentation or a stronger candidate source. Recall prompting is not monotonic across domains. |
| Staged DD-free detection on biography/legal row 0 | Biography improved to 20/22 shared signatures but took 13.7s and 4,550 tokens. Legal stayed at 19/22 shared signatures while taking 17.5s and 6,425 tokens. Hosted GLiNER seeding was unavailable due a `DEGRADED function cannot be invoked` response for `nvidia/gliner-pii`. | Mixed diagnostic | A native no-DataDesigner executor is still plausible, but only if it preserves phase boundaries with much cheaper seed/validation prompts or deterministic code. Naive direct LLM phases are not enough. |
| Chunked-excerpt validation in staged DD-free detection | On current one-row reruns, biography preserved the same 20/22 shared signatures as full-text validation but moved from 10.8s, 4,527 tokens, and 3 model requests to 13.3s, 5,648 tokens, and 6 requests. Legal preserved the same 19/22 shared signatures but moved from 14.7s, 6,425 tokens, and 3 requests to 17.2s, 7,727 tokens, and 7 requests. | Reject | Splitting direct validation into local excerpts increases repeated instruction overhead and request count on these non-shell rows. Do not pursue validator excerpting as a standalone no-DD speedup unless longer records show a different request/token crossover. |
| Rules-seeded staged DD-free detection | `rules` improved biography/legal latency but still lost baseline signatures; legal row 0 reached 20/22 shared signatures at 7.1s and 2,213 tokens. On shell-secrets, validation reclassified a database URL as a password and lost one baseline URL signature. `rules-trusted` fixed that shell loss and preserved all 12 shell signatures with one augmentation call per row, but still missed biography/legal signatures. With `--skip-augmentation-when-rule-covered`, trusted rules preserved all 12 shell signatures with zero model usage. | Mixed diagnostic | Deterministic seed spans are useful, but rule-covered spans should not always go through LLM validation. A native executor needs workload gates and should short-circuit locally when every requested label is rule-covered. |
| Rules + direct LLM staged DD-free detection | `rules-plus-direct-llm` preserved all 12 shell-secrets signatures while avoiding validation, but still used two model calls per row and 726-938 tokens because the direct seed and augmentation phases still ran. On row-0 smokes it looked like the most plausible mixed no-DD path: biography shared 20/22 signatures at 10.8s and 4,465 tokens, and legal shared 20/22 at 11.5s and 5,929 tokens. The five-row gate rejected it: biography shared only 80/114 baseline signatures, lost 34 signatures, and took 85.7s versus the DD baseline's 32.9-47.8s; legal shared 108/145, lost 37 signatures, and took 62.3s versus the DD baseline's ~39.5s. | Reject for contextual workloads | Trusting deterministic structured spans locally is still useful, but direct LLM seed/validation/augmentation is not a safe or faster replacement for DataDesigner-backed contextual detection on prose/legal slices. Keep no-DD promotion limited to fully rule-covered structured-secret lanes unless a new native executor passes repeated signature gates. |
| Rules router staged DD-free detection | `rules-router` preserved all 12 shell-secrets signatures with no seed, validation, or augmentation model calls. The mixed/contextual fall-through did not generalize: on five biographies it shared 96/114 default signatures and lost 18 baseline signatures; on five legal rows it shared 86/145 default signatures and lost 59 baseline signatures. The benchmark-safe expression of this result is `rules_covered_or_default`, which short-circuits only fully rule-covered label sets and otherwise runs default detection. | Mixed | Keep the router only for the zero-model rule-covered structured-secret lane. Do not use the direct local LLM fall-through as a prose/legal replacement; use default Anonymizer or another signature-gated strategy for contextual rows. |
| Integrated `native_rules_router` benchmark with corrected direct-call metering | A five-row benchmark-harness run on biography/legal confirmed the staged finding. Biography moved from 32.9s to 85.6s, requests from 20 to 15, and tokens from 43,354 to 26,644, but entities fell from 114 to 102 and 34 baseline signatures were uncovered. Legal moved from 54.3s to 62.3s, requests from 21 to 15, and tokens from 60,649 to 31,894, but 37 baseline signatures were uncovered. Both workloads rejected. | Reject for contextual workloads | Direct native calls can reduce request and token counts while still losing safety and wall-time. Treat lower token counts as insufficient evidence; contextual promotion requires signature preservation and latency improvement together. |
| Integrated `native_candidate_validate_no_augment` smoke | One-row biography/legal benchmark-harness smoke proved the no-augmentation native executor is much cheaper but unsafe. Biography moved from 24.8s to 5.9s, requests from 4 to 2, and tokens from 8,092 to 2,000, but entities fell from 15 to 12 and lost `age`, `first_name`, and `organization_name` signatures. Legal moved from 49.8s to 10.9s, requests from 4 to 2, and tokens from 13,791 to 3,823, but entities fell from 23 to 21 and lost `date`, `date_of_birth`, and `nationality` signatures. Both rows had zero original-value leaks. | Reject for contextual workloads | Removing augmentation from the native executor gives the expected speed boundary, but augmentation or a stronger candidate source remains load-bearing for contextual recall. Keep this arm as a diagnostic, not a promotion candidate. |
| Integrated `detector_native_validate_no_augment` smoke | Keeping the detector seed and replacing DataDesigner validation/augmentation with direct validation is much cheaper, but quality remains workload-dependent. Biography still rejects: latest one-row rerun moved 26.6s -> 6.7s and 8,398 -> 2,347 tokens, but entities fell 15 -> 14 and two augmenter-sourced child `first_name` signatures were uncovered. A focused one-row legal repeat improved median latency from 15.0s to 11.0s, requests from 4 to 3, and tokens from 9,516 to 4,150 with zero leaks. After row-parallel direct validation plus deterministic DOB-context label normalization, a wider three-row, two-repeat legal gate moved median elapsed from 40.6s to 21.3s, requests from 12.5 to 6.5, and tokens from 37,972 to 17,902 with zero original-value leaks. The split verdicts were `value_protection=pass`, `signature_parity=review`, and `performance=improved`: a filing-date span that baseline labeled `date_of_birth` was protected as `date`, while separate birth-context years were added as `date_of_birth`. | Mixed: biography reject, legal label-policy review | The promising shape is not "remove DataDesigner everywhere"; it is "keep DD as fallback, use deterministic fast lanes where provably covered, and only replace validation when a native validator preserves both coverage and label semantics across repeated gates." The legal repeats now show a real latency win, but a DD-free candidate may protect values correctly while disagreeing with DataDesigner label semantics. That should stay review-gated until label policy says whether such covered reclassification is acceptable. |
| Integrated `detector_native_validate_no_augment` substitute gate | A one-row legal substitute smoke first showed the same review shape as the redact gate: latency moved from 21.1s to 15.2s, requests from 5 to 3, and tokens from 12,192 to 6,871 with zero original-value leaks. The wider three-row, two-repeat substitute gate still improved performance, but rejected on safety: median pipeline latency moved from 44.0s to 33.4s, requests from 15.0 to 9.0, tokens from 47,958 to 28,465.5, and failed requests from 3.0 to 0.0, while both baseline and candidate leaked two original `date` values across two row-runs. Replacement-map coverage was complete; local replay showed the leak was a substitute collision where one synthetic date reused another protected original date in the same record. The candidate added 11 stable signatures, but had covered label mismatches including a stable `date_of_birth` -> `date` mismatch. | Reject for substitute promotion | Native validation reduces detection cost even when substitute still uses normal replacement-map generation, but speed cannot promote a substitute strategy while original values survive in replaced output. The leak appears in the default substitute arm too, so this is a baseline substitute safety issue separate from the native validator. |
| Integrated `gliner_native_validate_*` no-DataDesigner gate | A one-row biography/legal smoke tested direct hosted GLiNER seeding outside DataDesigner plus direct native validation, with and without direct native augmentation. Biography no-augment rejected despite improving latency/tokens because it lost two `first_name` signatures. Biography with native augmentation passed the one-row gate: latency 13.7s -> 10.2s, requests 4 -> 3, tokens 8,033 -> 5,035, entities 22 -> 24, zero leaks, and only candidate-only additions. After bounded per-row parallelism and targeted label-boundary guidance in the integrated no-DD executor, a repeat-3 five-row biography gate improved median wall time 40.7s -> 25.5s, requests 21 -> 15, and tokens 43,371 -> 27,643 with zero original-value leaks and no case failures. The guidance removed the earlier `first_name` label mismatches, but repeat comparison rejected the candidate: four baseline signatures were only covered with mismatched labels (`degree`: 1, `last_name`: 2, `place_name`: 1), and six stable baseline signatures became unstable (`degree`: 1, `last_name`: 2, `organization_name`: 2, `place_name`: 1). Legal improved latency/tokens in both one-row arms, but stayed review-gated because a generic filing date that baseline labeled `date_of_birth` was protected as `date` by the candidate; the seed guardrail correctly does not promote dates without birth/DOB context. | Reject for contextual biographies | Direct GLiNER outside DataDesigner is a useful performance diagnostic, but repeated stable-signature gates block promotion on this biography slice. Lower requests/tokens plus faster wall time are insufficient if label semantics are unstable. |
| Integrated `native_single_pass` benchmark smoke | One-row benchmark-harness smoke on biography/legal showed the speed boundary for collapsing detection into one direct provider call. Biography improved latency 10.3s → 1.7s, requests 4 → 1, and tokens 5,059 → 597, but found 4 entities versus 7 and lost three `person` signatures, so it rejected. Legal improved latency 19.2s → 1.1s, requests 5 → 1, and tokens 7,107 → 838 while preserving both signatures, so that single row was viable. | Mixed diagnostic | The one-call native extractor is worth keeping as a benchmark arm, but it is not safe for broad contextual use. Promotion needs repeated workload-specific signature gates; a legal-row win does not cancel the biography miss. |
| Integrated `native_single_pass` five-row gate | After adding a local deterministic rule guardrail, the larger biography/legal run still rejected both contextual workloads. Biography moved from 24.1s to 8.3s, requests from 21 to 5, and tokens from 26,759 to 3,078, but entities fell from 36 to 21 and it lost 16 `person` signatures. Legal moved from 35.7s to 6.1s, requests from 21 to 5, and tokens from 38,569 to 5,781, but entities fell from 14 to 12 and it lost three `person` signatures. | Reject for contextual workloads | Local rules can cheaply protect deterministic secret shapes, but they do not fix contextual recall. Collapsing detection to one direct call remains a useful lower-bound latency experiment, not a safe contextual replacement. |
| Integrated `native_single_pass_recall` five-row gate | The recall prompt improved raw recall, especially on legal text, but still rejected both workloads. Biography moved from 23.0s to 10.2s, requests from 21 to 5, and tokens from 26,730 to 4,072, but entities fell from 36 to 26 and it still lost 16 `person` signatures. Legal moved from 32.2s to 8.7s, requests from 21 to 5, and tokens from 38,085 to 6,885; entity count rose from 14 to 20, but two baseline `person` signatures were still uncovered. | Reject for contextual workloads | Prompt recall can improve counts without satisfying anonymization safety. One-call contextual extraction remains below the signature gate even when it is much faster and cheaper than default. |
| Integrated `native_single_pass_values*` value-only five-row gate | Two repetitions on five NVIDIA biography rows and five TAB legal rows confirmed the value-only one-call prompt shape is only a speed boundary. Compact values mode improved latency by 55.6% on biographies and 68.9% on legal, with 15-15.5 fewer requests and 31,770-60,491 fewer tokens, but rejected both workloads after losing 45 biography and 123 legal baseline-only signatures. Recall values mode still rejected: it improved latency by 31.9% on biographies and 38.7% on legal, but lost 40 biography and 96 legal baseline-only signatures. Output: `/tmp/anonymizer-native-values-paired-r5`. | Reject for contextual workloads | Returning values instead of offsets makes parsing cheaper but does not solve contextual recall. Keep this arm in the harness as a lower-bound diagnostic; do not promote one-call extraction on biographies or legal text without a different seed source or repeated signature parity. |
| Structured fast-lane router tightening | `rules_covered_or_default` now short-circuits only the structured-secret labels `api_key`, `email`, `http_cookie`, `password`, `pin`, `unique_id`, `url`, and `user_name`. Narrow prose rule labels such as `date_of_birth`, `organization_name`, `religious_belief`, and `street_address` fall back to default detection unless `rules_only` is explicitly selected. A shell-secret smoke still found 12 entities across 3 records with 0 model rows, 0 requests, and 0 tokens. | Review | This preserves the no-DataDesigner win without assuming all inputs are shell logs. Local prose rules remain useful as explicit experiments or guardrails, but they are not complete enough for automatic contextual anonymization. |
| Narrow prose-label augmentation skip probe | On one synthetic `organization_name` + `street_address` row, `rules_covered_or_default` correctly fell back to model-backed detection instead of using the zero-model fast lane. A repeat-3 comparison then found `rules_guardrail_no_augment` preserved the same two signatures with zero leaks while moving median latency 3.0s → 2.6s, requests 4 → 3, and tokens 3,069 → 2,133. | Candidate | Skipping augmentation can be viable for tightly scoped prose-label slices when detector+validator already recover the needed spans. This is not a broad prose default; promote only through repeated signature gates, especially on biographies and legal text where augmentation may carry recall. |
| Real biography/legal no-augmentation check | On two NVIDIA biography rows, pure `no_augment` rejected: latency regressed 24.1s → 28.8s, entities fell 48 → 46, and two `first_name` signatures were lost. `rules_guardrail_no_augment` improved biography latency/tokens (24.1s → 18.3s, 17,992 → 11,905 tokens) but still rejected after losing the same two `first_name` signatures and using rule-sourced spans. On two TAB legal rows at offset 2, `no_augment` preserved signatures and reduced tokens but regressed latency (27.2s → 38.6s) and increased failed-request rate; `rules_guardrail_no_augment` preserved signatures with modest latency/token gains but remained review-gated because it introduced rule-sourced spans. | Mixed: biography reject, legal review | The synthetic augmentation-skip win does not generalize to biography prose. Augmentation remains load-bearing for contextual name recall, and legal gains need repeated runs plus failed-request scrutiny before promotion. |
| `rules_covered_or_default` mixed benchmark harness run | A two-row synthetic shell-secret run initially exposed a rule hole: default found one sudo-stdin password that the rule-only path missed. After adding a narrow `echo "..." | sudo -S` password rule, the rerun preserved all 9 shell signatures with detection latency 21.4s → 0.004s, requests 8 → 0, and tokens 9,854 → 0. One-row biography and legal contextual configs included `person`, so they fell back to model-backed detection and matched default entity counts. | Review | This is the safest implementation shape for the no-DataDesigner idea: use local rules only where labels and observed signatures prove coverage, and treat every missed signature as a rule-quality bug or a reason to fall back. |
| `rules_covered_or_default` current mixed fallback run | Current-code rerun completed all 6 cases. Shell secrets preserved all 9 signatures with pipeline latency 23.1s → 0.005s, requests 8 → 0, and tokens 10,173 → 0. The biography and legal configs included `person`, so both candidate cases fell back to model-backed detection and matched default entity counts and signatures: biography 7/7, legal 2/2. | Review | The router is behaving as designed after the rule-only `tagged_text` contract fix: structured secret configs can short-circuit locally, while contextual non-shell configs stay on default detection. |
| `rules_covered_or_default` repeated shell-secret run | A three-repetition shell-only suite completed all 6 cases. The candidate preserved all 9 final signatures in every repetition with median detection latency 29.4s → 0.004s, requests 9 → 0, and tokens 10,112 → 0. Default detection was unstable on this tiny slice: one repetition missed one `api_key`, so stable signatures were 8 for default and 9 for the rules path. The comparison remained review-gated, not viable, because all candidate spans were rule-sourced. | Review | Repeated evidence strengthens the structured-secret fast path but also shows why promotion should use stable-signature comparisons rather than treating default as perfectly deterministic on every repetition. |
| `rules_covered_or_default` on non-shell structured secrets | A four-row JSON/env/HTTP-header/YAML-style suite initially rejected after exposing two deterministic-rule gaps: URLs swallowed trailing semicolon separators and `session_id=...` cookie values were not protected. After tightening URL boundaries and adding a narrow `session_id` assignment rule, the rerun preserved all 17 default signatures while moving detection latency 25.8s → 0.010s, requests 16 → 0, and tokens 19,167 → 0. A repeat-3 run then kept all candidate signatures stable: default produced 15, 15, and 16 entities with median 18,822 tokens, while the rules path produced 17 entities every time with zero model requests and zero tokens. | Review | The no-DataDesigner fast lane is not shell-specific, but it must remain rule-coverage and signature-gated. Treat every structured-secret miss as either a narrow rule bug with tests or a reason to fall back to default detection. |
| `local_structured_substitute` on non-shell structured secrets | A four-row JSON/env/HTTP-header/YAML-style substitute suite preserved the same 17 final entities with zero original-value leaks. In a repeat-3 run, DataDesigner-backed substitute had median pipeline latency 38.1s, 4 requests, and 13,967 tokens for replacement-map generation; individual DD-backed runs ranged from 30.7s to 62.4s. `local_structured_substitute` had median latency 0.005s, 0 requests, and 0 tokens while preserving the same 17 replacements. | Review | Replacement-map generation is now another defensible no-DataDesigner lane for structured labels. Keep it benchmark-only until repeated gates and a policy decision define which structured labels deserve public API support. |
| `local_structured_substitute` with model-backed detection fallback | A one-row audit-style structured-identifier suite requested `api_key`, `http_cookie`, `pin`, `unique_id`, and `user_name`, so `rules_covered_or_default` fell back to normal model-backed detection in both arms. Both arms found the same 5 final entities with zero original-value leaks. Local replacement removed the replacement-map workflow, moving pipeline latency 53.6s → 33.0s, requests 5 → 4, and tokens 11,547 → 7,694. The pairwise comparison marked the candidate viable. | Candidate | Local replacement-map generation can help even when detection still needs DataDesigner. This is a cleaner promotion path than rule-only detection because contextual detection provenance is preserved; keep rejecting contextual replacement labels such as `person`. |
| `local_structured_substitute` with default detection on varied audit/config/HTTP identifiers | A four-row repeat-3 suite isolated replacement-map generation by keeping default model-backed detection in both arms. After adding a local synthetic-original collision guard, the guarded rerun kept value protection clean: zero original leaks, zero missing replacement-map entries, and zero synthetic-original collisions. Local substitute moved median pipeline latency 18.8s -> 12.7s, requests 21 -> 17, and tokens 24,324 -> 17,015. A current fixed-trace replay held detection constant at 21 entities and measured replacement only: DataDesigner substitute took 6.15s while local structured substitute took 0.003s, with 21/21 replacements and zero leaks/collisions in both arms. Regenerating the older repeat comparison with split verdicts moved the strategy-screen group out of `needs_split_verdict_rerun`; adding the fixed-trace replay comparison moved it to `promising_needs_review`. All three rows have `value_protection=pass`; the replay row has `signature_parity=pass` and `candidate_verdict=candidate_viable`, while one full-pipeline pairwise row has `signature_parity=review` because two covered signatures used different labels (`api_key`, `unique_id`). The comparison now tags this drift as `replacement_only_detection_instability` because detection strategy did not change. | Promising needs review | This is the cleanest structured-label promotion path because detector provenance stays model-backed in full-pipeline runs and the replacement backend passes fixed-trace replay. It is not fully promoted because normal pairwise runs still need monitoring for provider reliability and detection-run label drift. |
| `local_structured_substitute` fixed-trace replay on biography structured labels | A five-row NVIDIA synthetic biography replay used model-backed detection for `date_of_birth`, `organization_name`, `religious_belief`, and `street_address`, then replayed both substitute backends three times on the same 56 detected entities. After making local replacement-map generation avoid per-record duplicate synthetic values, both arms produced 159 replacements across 15 replacement attempts with zero duplicate synthetics, zero missing replacement-map entries, zero original-value leaks, and zero synthetic-original collisions. DataDesigner substitute took 23.59s for replacement-map generation and local structured substitute took 0.006s. The replay comparison marks `value_protection=pass`, `signature_parity=pass`, `safety=pass`, and `candidate_verdict=candidate_viable`. Output: `/tmp/anonymizer-perf-goal/biography-supported-structured-replacement-replay-repeat3.json`; comparison: `/tmp/anonymizer-perf-goal/biography-supported-structured-replacement-replay-repeat3-comparison.csv`; screen: `/tmp/anonymizer-perf-goal/strategy-screen-local-substitute-with-biography-replay-groups.csv`. | Candidate | This broadens the replacement-only result beyond shell or config logs without claiming DD-free contextual detection. The speed and leak profile are strong, the duplicate-collapse issue is fixed for this slice, and repeated replacement-only evidence shows the local backend can preserve replacement-map coverage when detection is held fixed. The remaining gate is policy: decide which structured labels and text shapes are eligible for deterministic substitute generation in production-facing configuration. |
| Expanded `rules_covered_or_default` + `local_structured_substitute` on an audit-style structured identifier record | After adding narrow keyed coverage for `http_cookie`, `pin`, `unique_id`, and `user_name`, the candidate protected all baseline signatures, found one additional `unique_id`, had zero original-value leaks, and moved pipeline latency 9.2s → 0.005s, requests 5 → 0, and tokens 6,075 → 0. | Review | This extends the no-DataDesigner fast lane beyond shell logs into keyed audit/config/HTTP-style structured records. It remains review-gated because every final span is rule-sourced and this run used one row. |
| Expanded `rules_covered_or_default` + `local_structured_substitute` on varied audit/config/HTTP identifiers | A four-row repeat-3 suite preserved every baseline-only signature through containing or overlapping candidate spans, with zero original-value leaks. Median pipeline latency moved 21.1s → 0.006s, requests 21 → 0, and tokens 24,332 → 0. The comparison records 8 exact baseline-only signatures, 8 candidate-covered signatures, 2 span-boundary mismatches, and 0 uncovered signatures. | Review | This is the strongest no-DataDesigner result so far for non-shell structured records. It is still not a default: all final spans are rule-sourced, and two protected values had different span boundaries such as `token=<value>` versus `<value>`, so promotion needs a workload policy gate. |
| Row-aware `rules_covered_or_default` + local substitute smoke | A four-row JSON/env/HTTP-header/YAML-style suite initially rejected because quoted JSON `user`/`pin` keys were not rule-covered. After adding quoted-key coverage and changing the router to fall back per row on suspicious uncovered structured assignments, the structured candidate moved pipeline latency 9.7s -> 0.0s, requests 20 -> 0, and tokens 20,080 -> 0 while matching entity count 10 -> 10 with zero original-value leaks. One-row biography and legal controls included `person`, used default detection in both arms, and passed comparison gates. | Review | The no-DataDesigner path is now safer: eligible labels are necessary but not sufficient, and rows with uncovered structured fields go through normal detection. The structured candidate still stays review-gated because one `HF_TOKEN` value was protected under a different label/boundary than the default `http_cookie` span. |
| Row-aware `rules_covered_or_default` + local substitute repeat gate | A focused repeat-3 split-verdict suite reran the same four structured rows after the row-aware router change. All 6 cases completed. Default substitute had median pipeline latency 12.4s, 21 requests, and 20,071 tokens; the row-aware rules/local candidate had median latency 0.006s, 0 requests, and 0 tokens. Both arms found 10 entities in every repetition and had zero original-value leaks or synthetic-original collisions. The split-verdict comparison has `value_protection=pass` but remained `safety=review` and `signature_parity=review`: one stable baseline `http_cookie` signature was protected by the candidate under an `api_key` label with a span/boundary mismatch. Output: `/tmp/anonymizer-perf-goal/structured-fastlane-split-r3`. | Needs viable split verdict | This is a large structured fast-lane performance win, but not promotion-ready. The next decision is whether the covered `http_cookie` -> `api_key` mismatch is acceptable value protection for this workload or whether the deterministic rules need to match baseline label semantics more closely. |
| `bio-vmax10-w80` validator window tuning | Rejected on biography rows 6-10: latency, requests, and tokens regressed, and stable `field_of_study` and `state` signatures were lost. | Reject | Smaller validation windows need per-workload proof; prompt-size savings can be outweighed by more calls and lost context. |
| Text augmenter routing at `temperature: 0.3` | A one-row biography smoke test passed, but repeated five-row slices did not: rows 0-4 preserved signatures while latency regressed from 40.4s to 45.9s and requests from 21.0 to 21.5; rows 5-9 rejected after latency regressed from 41.0s to 52.1s and two stable `state` signatures became unstable. | Reject | JSON-validator/text-augmenter routing at the default text temperature is not a reliable prose speedup on these slices. |
| Text augmenter routing at `temperature: 0.7` | Passed the first biography slice, then failed on rows 6-10 by losing a stable `university` signature and regressing latency. | Reject | Do not promote the routing pattern from a single positive slice. |
| `rules_guardrail_no_augment` on legal prose | Improved latency/tokens on legal rows 2-3, but lost two stable `first_name` signatures. | Reject | Augmentation remains load-bearing for contextual names, even when aggregate entity counts look acceptable. |

No broad replacement for the default prose/legal detection path has passed the
current repeated signature checks. The only strong performance result so far is
workload-scoped: deterministic rules for tightly bounded, rule-covered secret
scans.

When DataDesigner message traces are enabled, interpret failed request counts
through `observed_non_bridge_*` metrics before drawing provider-reliability
conclusions. Across 13 local trace files, the local-vLLM
`SyncClientUnavailableError` rows were 104 near-zero-latency sync-to-async
bridge fallbacks with zero token usage; they are adapter accounting, not model
work. GLiNER `ProviderError` rows are different: the same trace set had 20 real
detector failures, which can invalidate otherwise faster detector-heavy
candidates.

Do not expand deterministic rules into contextual names merely to recover the
failed candidates above. The rejected prose and legal runs lost labels such as
`first_name`, `field_of_study`, `state`, and `university`; these require context
and separate precision evidence. The rule layer should stay narrow unless a new
label has high-confidence syntax and false-positive tests.

## Validator Chunk Tuning

The detector validator can dominate replace-mode latency on records with many
candidate entities. Tune `Detect.validation_max_entities_per_call` and
`Detect.validation_excerpt_window_chars` together:

- `validation_max_entities_per_call` controls how many candidate entities go
  into each validator call. Lower values create more calls, but Anonymizer can
  overlap those calls through the validator pool.
- `validation_excerpt_window_chars` controls how much text surrounds each
  validation chunk. Lower values reduce prompt size, but can hide context the
  validator needs for labels such as `date_of_birth`, `race_ethnicity`, or
  legal roles.

Run these sweeps per workload. A window that is safe for short biographies may
drop legal identifiers, and a legal-safe window may erase the speedup on short
records.

Example config fragment:

```yaml
configs:
  - id: legal-vmax10-w160
    detect:
      validation_max_entities_per_call: 10
      validation_excerpt_window_chars: 160
      entity_labels: [first_name, last_name, court_name, date, date_of_birth]
    replace:
      strategy: hash
      digest_length: 12
```

Use the aggregate analysis first:

```bash
uv run python tools/measurement/analyze_benchmark_output.py \
  benchmark-runs/legal-window-sweep \
  --json
```

Then compare every faster candidate against a higher-context reference:

```bash
uv run python tools/measurement/extract_signature_deltas.py \
  /tmp/reference/legal__default-window__r000.detection-artifacts.jsonl \
  /tmp/candidate/legal__vmax10-w160__r000.detection-artifacts.jsonl \
  --baseline-artifact-root /tmp/reference/artifacts \
  --candidate-artifact-root /tmp/candidate/artifacts \
  --baseline-config default-window \
  --candidate-config vmax10-w160 \
  --workload legal \
  --output /tmp/legal-vmax10-w160-deltas.csv \
  --format csv
```

Treat a candidate as unsafe until signature deltas are clean on repeated runs.
In one local vLLM check with two repetitions, a biography sample went from
24.6s with the default window to 17.8s with `vmax10/w80`, with all 50 stable
entity signatures preserved. A one-row legal sample went from 21.2s with the
default window to 13.2s with `vmax10/w160`, with all 28 stable signatures
preserved. Both candidates increased request and token counts, so the comparison
tool marks them for review instead of as automatic wins.

The biography `vmax10/w80` result did not hold on the next five biography rows.
With `row_offset: 5`, median latency regressed from 31.8s to 33.6s, requests
from 20.0 to 43.0, and tokens from 44,367.0 to 68,407.5. The comparison also
lost stable `field_of_study` and `state` signatures, with an additional
unstable `university` loss, so the tool rejected the candidate. Recheck this
tuning on the target dataset because smaller windows can miss sensitive
attributes and because the extra parallel validator calls can overwhelm any
prompt-size savings.

## Augmentation Ablation

Use `experimental_detection_strategy: rules_guardrail_no_augment` to measure
what happens when the detector keeps GLiNER, validation, and deterministic rule
guardrails, but skips LLM augmentation. Treat this as an ablation, not as a
replacement for the default pipeline.

In a local vLLM check with two repetitions, removing augmentation from the
two-row biography sample reduced work but consistently lost two stable
`first_name` signatures. The comparison tool rejected both the default-window
and `vmax10/w80` no-augmentation candidates. This indicates augmentation is
load-bearing for prose records where contextual names and quasi-identifiers
matter.

The same ablation preserved all 28 stable signatures on a one-row legal sample.
With the default validation window, latency moved from 21.2s to 18.3s, requests
from 5 to 4, and tokens from 11,327.5 to 7,654. With `vmax10/w160`, latency
moved from 13.2s to 9.5s, requests from 8 to 7, and tokens from 16,604 to
12,881. Compared directly against the default-window baseline, the combined
legal candidate is faster but still needs review because validator chunking
increases requests and tokens.

That legal no-augmentation result also failed to generalize to the next two
legal records. On `row_offset: 1` with two rows and two repetitions, comparing
`legal-noaugment-vmax10-w160` against the same-window full augmentation baseline
improved latency from 23.9s to 21.5s, requests from 28.0 to 26.0, and tokens
from 61,780.5 to 50,905.0, but the candidate lost two stable `first_name`
signatures and one unstable `date` signature. The comparison rejected it despite
the performance improvement.

Use this ablation when `augmented_new_final_value_count` is near zero for the
target workload and repeated signature deltas are clean. Do not generalize a
single legal row to the rest of a legal dataset, and do not generalize legal
results to biography, support-ticket, shell-history, or mixed prose data without
rerunning the comparison on that workload.

## Augmenter Routing and Temperature

The detection validator and augmenter do different jobs. Keep them separable in
model configs when testing local endpoints:

- validators benefit from deterministic JSON-oriented settings;
- augmenters may work better through a text alias, because DataDesigner
  structured parsing can be fragile on local OpenAI-compatible endpoints;
- augmenter temperature changes can alter retry pressure and output shape, so
  evaluate them with repeated signature comparison, not only entity counts.

In one local vLLM biography run with two repetitions, keeping the validator on
`local-nemotron-json` while routing the augmenter to a text alias with
`temperature: 0.7` was the first prose candidate that passed the current safety
gate and improved performance. Median latency moved from 24.2s to 21.6s,
requests from 8 to 6, and tokens from 17,938.5 to 11,921. The comparison had no
baseline-only or unstable-lost signatures across 48 stable signatures, so the
tool marked it `candidate_viable`.

The same routing/temperature candidate also held on a five-row biography slice
with two repetitions, though the gain was smaller. Median latency moved from
40.4s to 38.0s, requests from 21.0 to 20.5, and tokens from 43,367.5 to
43,043.0. It preserved all 114 stable baseline signatures; one candidate-only
`place_name` appeared in one repetition, so the comparison still marked the
candidate `candidate_viable`.

This result did not generalize cleanly to the next five biography rows. On a
second slice using `row_offset: 5`, the same candidate was rejected: median
latency moved from 41.0s to 47.5s, requests from 21.0 to 21.5, and tokens were
effectively unchanged at 44,708.0 to 44,670.0. More importantly, the comparison
lost one stable `university` signature and had unstable losses for
`field_of_study` and `university`. Treat this routing as an experiment to
retest on each workload, not as a default candidate yet.
When the two temp-0.7 config IDs are grouped with `--config-aliases`, the
biography family result is `conflicting_evidence`: three comparison rows, two
viable rows, one reject row, best latency -10.4%, worst latency +16.0%, and
stable losses for `field_of_study` and `university`.

On a two-row legal slice with two repetitions, the same augmenter routing did
not materially improve latency or requests: median latency moved from 27.3s to
27.5s, requests stayed at 8, and tokens moved from 24,460.0 to 24,296.5. It
preserved stable signatures, but the rule-guardrail legal strategy remains
review-gated and this routing should be treated as neutral for that sample.
Also compare it against prompt-only changes such as
`prose_augment_focus`: in the same biography slice, prose-focused augmentation
preserved signatures and reduced requests/tokens, but wall time increased from
24.2s to 26.4s, so the tool kept it in review.

Parser compatibility is a separate concern. A text-model suite without
`dd_parser_compat: raw_json` produced a failed biography case in local testing;
the raw-parser compatibility mode fixed that failure, but increased latency and
tokens on both biography and legal slices. Treat raw-parser compatibility as an
endpoint interoperability fix, not as a performance optimization.

## Detector-Only Ablation

Use `experimental_detection_strategy: detector_only` to measure the lower bound
of the detection phase when GLiNER output is trusted directly and only local
span finalization runs afterward. Use
`experimental_detection_strategy: rules_guardrail_detector_only` to measure the
same path with deterministic high-confidence rule spans unioned into the final
entity set. Both remove LLM validation and LLM augmentation from the detection
phase, so they are diagnostic ablations rather than deployable strategies.

The comparison tool marks these candidates with
`candidate_skips_llm_validation`, which forces `safety_verdict: review` even
when entity signatures match on the sampled records. The rule-guardrail variant
also gets `candidate_uses_rule_entities` when rule spans survive. Promote either
path only if independent precision checks show false positives are acceptable
for the target workflow and repeated signature deltas remain clean.

In a one-row cross-workload smoke check, detector-only improved latency and
token counts on biography, legal, and shell-secrets slices, but all three
candidates were rejected by signature comparison. Biography moved from 13.7s to
0.9s and lost two baseline `first_name` signatures; legal moved from 15.3s to
1.0s and lost one `nationality` signature while increasing final entity count
from 22 to 39; shell-secrets moved from 6.6s to 0.8s and still lost one
baseline `api_key` signature. This is a useful lower-bound measurement, but it
shows why validation/augmentation or deterministic rule coverage remain
load-bearing for anonymization.

The `rules_guardrail_detector_only` variant did not fix prose/legal losses in
the same one-row check: biography still lost two `first_name` signatures and
legal still lost one `nationality` signature. It did preserve all shell-secret
baseline signatures while moving latency from 4.6s to 0.8s, requests from 4 to
1, and tokens from 3,969 to 85. Treat that as a narrow structured-secret
candidate. It remains review-gated because it skips LLM validation and relies
on deterministic rules.

On the three-row shell-secrets slice with three successful candidate
repetitions, `rules_guardrail_detector_only` preserved all stable baseline
signatures while moving median latency from 7.2s to 3.2s, requests from 12 to 4,
and tokens from 11,019 to 198. The final entity set came from 9 detector spans
and 3 rule spans. It still had local GLiNER `ProviderError` health-check
failures and remains slower than `rules_only`, which used zero model calls and
zero tokens on the same fully rule-covered labels.

## Deterministic Rules for Structured Secrets

Use `experimental_detection_strategy: rules_only` only when the workload is a
bounded secret-scanning task and every requested label is covered by the
deterministic rules. Current rule coverage is intentionally narrow:
`api_key`, `date_of_birth`, `email`, `http_cookie`, `organization_name`,
`password`, `pin`, `religious_belief`, `street_address`, `unique_id`, `url`,
and `user_name`. The `http_cookie`, `pin`, `unique_id`, and `user_name` rules
cover keyed or command-style structured patterns only. They do not recognize
free-form names, narrative identifiers, or arbitrary prose mentions.

The zero-model detector is implemented by
`EntityDetectionWorkflow.detect_with_high_confidence_rules()`. The benchmark
strategy delegates to that internal engine method, but no user-facing config
selects it outside the benchmark harness.

Use `experimental_detection_strategy: rules_covered_or_default` for mixed
benchmark suites where some configs are structured-secret scans and others
include contextual labels such as `person`, `organization_name`, or
`street_address`. It runs the same zero-model path for structured fast-lane
cases, but does not attempt a DataDesigner-free replacement for contextual
prose or legal records.

A mixed local-vLLM smoke run on June 8, 2026 used two synthetic shell-secret
rows plus one biography and one legal row. The first shell run found that
`rules_covered_or_default` missed a sudo stdin password that default
augmentation caught; after adding a narrow `echo "..." | sudo -S` rule, the
rerun preserved all 9 shell signatures with zero model requests and zero tokens.
The biography and legal configs requested `person`, so they correctly fell back
to model-backed detection and matched default entity counts. Keep this strategy
signature-gated: a missed default signature is a rule-quality bug or a fallback
signal, not acceptable drift.

A follow-up three-repetition shell-only run kept all 9 candidate signatures
stable while default detection had 8 stable signatures because one `api_key`
was absent from one default repetition. The comparison still returned
`candidate_verdict=review` because the candidate had no detector-sourced final
spans. This is the intended behavior: repeated clean signatures can justify a
workload-scoped fast lane, but rule-only provenance should remain an explicit
review decision.

For substitute workloads, use
`experimental_replacement_strategy: local_structured_substitute` to bypass the
DataDesigner replacement-generator call. The local substitute generator writes a
normal replacement map and stamps `_replacement_map_source=local_structured` so
measurement estimates do not count a replacement-map LLM call. It only supports
structured labels. Pair it with `rules_covered_or_default` when all requested
labels are also rule-covered; otherwise detection can still use the default
model-backed path while replacement-map generation stays local. If a config
includes `person` or another contextual label, preflight fails instead of
silently producing poor local substitutes.

On the current four-row non-shell structured-secret suite,
DataDesigner-backed substitute preserved 17 entities with zero original-value
leaks but had repeat-3 median latency 38.1s, 4 requests, and 13,967 tokens in
replacement-map generation. The local structured substitute arm preserved the
same 17 entities, had zero original-value leaks, and had repeat-3 median latency
0.005s with 0 requests and 0 tokens. The repeat output used for this result is
`/tmp/anonymizer-perf-goal/structured-secrets-local-substitute-repeat3`.

The local substitute backend can also combine with model-backed detection. In
the first one-row audit-style structured-identifier smoke, `api_key`,
`http_cookie`, `pin`, `unique_id`, and `user_name` were not all rule-covered, so
detection fell back to the default model path in both arms. The local substitute
arm still removed the replacement-map DataDesigner workflow, moving pipeline
latency from 53.6s to 33.0s, requests from 5 to 4, and tokens from 11,547 to
7,694 while preserving the same 5 final entities and zero original-value leaks.
The output used for that result is
`/tmp/anonymizer-perf-goal/structured-identifiers-local-substitute`.

Use `replay_replacement_strategies.py` when you need to hold detection fixed and
isolate replacement-map generation:

```bash
uv run python tools/measurement/replay_replacement_strategies.py \
  /tmp/anonymizer-perf-goal/structured_identifiers_varied.csv \
  --text-column text \
  --labels api_key,http_cookie,password,pin,unique_id,user_name \
  --nrows 5 \
  --replacement-repetitions 3 \
  --model-configs /stable-cache/anonymizer/local-vllm-json-models.yaml \
  --model-providers /stable-cache/anonymizer/local-vllm-providers.yaml \
  --dd-parser-compat raw_json \
  --comparison-output /tmp/anonymizer-perf-goal/structured-identifiers-replacement-replay-comparison.csv \
  --json
```

The current fixed-trace replay detected 21 entities once, then ran both
substitute backends on that same trace. DataDesigner substitute took 6.04s for
replacement-map generation; local structured substitute took 0.003s. Both arms
produced 21 replacements, zero missing replacement-map entries, zero
original-value leaks, and zero synthetic-original collisions. The JSON output
used for this result is
`/tmp/anonymizer-perf-goal/structured-identifiers-replacement-replay.json`.
A rerun after adding an LLM replacement-map collision guard produced the same
21/21 complete, leak-free, collision-free result. In that rerun, DataDesigner
substitute took 6.22s and local structured substitute took 0.003s; the updated
JSON output is
`/tmp/anonymizer-perf-goal/structured-identifiers-replacement-replay-after-llm-guard.json`.
When `--replacement-repetitions` is greater than one, detection still runs once
and only the substitute backends repeat. The summary rows aggregate replacement
latency, missing-map counts, leaks, collisions, duplicate synthetics, and source
counts across those repeated backend passes. When `--comparison-output` is set,
the replay tool also writes a one-row comparison CSV with
`value_protection_verdict`, `signature_parity_verdict`, `safety_verdict`,
`performance_verdict`, and `candidate_verdict`. This lets
`screen_strategy_comparisons.py` include fixed-trace replacement evidence
alongside normal pairwise benchmark comparisons. Missing local replacement-map
entries, original-value leaks, and synthetic-original collisions fail the replay
candidate even if the elapsed-time delta is large.
If the DD substitute baseline misses replacement-map entries or leaks original
values while the local backend covers them, the replay comparison emits
candidate-covers-baseline flags and the strategy screen recommends
`candidate_covers_baseline_defects` for all-review groups of that shape. Treat
that as a baseline-independent safety-rule prompt: inspect the candidate's
missing, leak, collision, duplicate-synthetic, and supported-label columns
rather than requiring exact parity with a known-flawed substitute baseline.

After adding narrow keyed rules for `http_cookie`, `pin`, `unique_id`, and
`user_name`, the same audit-style label set can now short-circuit both
detection and local replacement for a structured record. In a one-row local
vLLM check, default detection plus DataDesigner substitute found 4 entities and
missed the `unique_id`; the rules/local arm found 5 entities, had zero
original-value leaks, and moved pipeline latency from 9.2s to 0.005s, requests
from 5 to 0, and tokens from 6,075 to 0. The pairwise comparison remains
`review`, not `candidate_viable`, because the candidate has rule-only
provenance and the evidence is a single row. The output used for this result is
`/tmp/anonymizer-perf-goal/structured-identifiers-expanded-rules`.

On a three-row shell-secrets slice with labels `[api_key, password, email, url]`,
`rules_only` preserved all 12 stable signatures across three repetitions while
moving median latency from 7.2s to 0.004s, requests from 12 to 0, and tokens
from 11,019 to 0 in the refreshed failure-aware comparison. The comparison tool
still marks the candidate for review because it has no contextual detector spans
and skips LLM validation. That is the right gate: a pure rule strategy is
acceptable only when missing contextual spans is part of the test contract.

`rules_seed_no_augment` preserved the same 12 signatures and reduced median
tokens from 11,017 to 7,732, but median latency moved from 8.0s to 8.5s on the
same slice. In this run, seeding rules into the validator path reduced token
work but did not improve end-to-end latency. Prefer `rules_only` for tightly
scoped secret scans; prefer rule guardrails plus contextual detection for prose,
legal text, support tickets, and mixed records.

Use `rules_filter_guardrail` as the mixed-workload version of that idea. It
keeps LLM augmentation, but rule-covered spans are not sent to the seed
validator. The rule spans are reinserted before augmentation so the augmenter
does not waste work rediscovering them. This is a candidate for datasets that
combine structured secrets with contextual prose; it still needs repeated
signature comparison because filtered detector spans no longer receive the
LLM validator's reclassification/drop pass. In a local shell-secrets smoke run,
the completed candidate repetition reduced seed validation candidates to zero
and preserved all stable signatures, but the repeated comparison rejected it
because a later candidate case hit a GLiNER health-check rate limit.

## Metric Interpretation

Use metrics as signals, not as a single score.

Latency and throughput:

- `elapsed_sec`: wall time for a measured stage or DataDesigner workflow.
  Staged DD-free detection cases report end-to-end case wall time here.
- `rows_per_sec`: completed output rows per second for the measured block.
- `tokens_per_sec`: observed total tokens per second when token usage exists.
- `text_length_tokens_bucket`: a coarse text-size bucket for comparing similar
  inputs without storing text.
- `record_count` and `input_text_tokens_total`: case-level workload size
  derived from record measurements. These are independent of provider-reported
  token usage.
- `records_per_pipeline_sec` and `input_text_tokens_per_pipeline_sec`: dataset
  throughput normalized by the measured Anonymizer pipeline stage. The matching
  `*_per_ndd_sec` fields use summed DataDesigner workflow wall time instead.
- `input_text_tokens_per_endpoint_sec` and
  `input_text_tokens_per_gpu_sec`: optional topology-normalized dataset
  throughput. These are populated only when benchmark run tags provide portable
  topology counts such as `endpoint_count` or `gpu_count`.

LLM usage:

- `observed_input_tokens`, `observed_output_tokens`, and
  `observed_total_tokens`: provider-reported usage when available. Missing or
  zero values mean the provider path did not expose usage, not necessarily that
  no tokens were consumed.
- `observed_total_requests`, `observed_successful_requests`, and
  `observed_failed_requests`: request counts when DataDesigner or a native
  benchmark model workflow exposes them.
- `observed_failed_request_rate`: failed requests divided by total requests.
  Case and group tables expose this as the end-to-end retry pressure for a
  strategy; model usage tables expose it per workflow/model. Sort by this
  together with total token count to find retry-heavy workflow/model pairs.
- `observed_bridge_fallback_requests`: DataDesigner sync-to-async bridge
  fallbacks, derived from message traces when `--dd-trace` is enabled. Treat
  these as adapter accounting, not provider/model failures.
- `model_elapsed_sec`: staged DD-free detection only; sum of direct model-call
  durations for seed, validation, and augmentation. This stays `0.0` for fully
  local rule-covered runs even when `elapsed_sec` records nonzero local work.
- `observed_non_bridge_total_requests`,
  `observed_non_bridge_failed_requests`, and
  `observed_non_bridge_failed_request_rate`: request metrics after subtracting
  sync-to-async bridge fallbacks. Prefer these fields over raw failed-request
  counts when diagnosing provider reliability from traced runs.
- `nominal_llm_call_count`: an internal estimate based on the Anonymizer
  pipeline shape. Treat it as expected work, not observed provider traffic.
- `seed_validation_candidate_count`: number of detector candidates sent to the
  seed validator, derived from detection artifacts without storing values.
- `estimated_seed_validation_chunk_count`: estimated validator chunk count after
  applying `detect.validation_max_entities_per_call`. If this does not change
  between benchmark configs, chunk-size experiments are not expected to reduce
  successful validator calls.

Entity and quality metrics:

- `final_entity_count`: entities that survive detection and validation.
- `original_value_leak_count`: number of final entity original values that
  still appear verbatim in the replaced or rewritten output text. This is a
  conservative replace/rewrite safety signal and stores only counts, not raw
  values.
- `original_value_leak_label_counts`: per-label counts for those surviving
  original values. The analysis tables aggregate these as
  `original_value_leak_record_count`, `sum_original_value_leak_count`,
  `leaking_case_count`, and `median_original_value_leak_count`.
- `replacement_missing_final_entity_count`: number of final entity occurrences
  whose original value has no entry in the replacement map. This is sanitized
  replacement-map coverage, not raw leakage text.
- `replacement_missing_final_value_count`: number of unique final entity values
  with no replacement-map entry. Compare it with
  `original_value_leak_count` to distinguish omitted replacement-map entries
  from replacement-application or metric issues.
- `replacement_missing_final_entity_label_counts`: per-label counts for missing
  replacement-map coverage.
- `replacement_synthetic_original_collision_count`: number of final entity
  occurrences whose original value was reused as a synthetic replacement value
  elsewhere in the same record. This is a substitute safety signal; map
  coverage can be complete while this is nonzero.
- `replacement_synthetic_original_collision_value_count`: number of unique
  protected original values reused as synthetic replacement values.
- `replacement_synthetic_original_collision_label_counts`: per-label counts for
  synthetic-original collisions.
- `artifact_final_detector_entity_count`,
  `artifact_final_rule_entity_count`, and
  `artifact_final_augmenter_entity_count`: final entity source counts derived
  from detection artifact sidecars. These are useful safety signals for
  rule-backed benchmark strategies.
- `artifact_final_entity_signature_count` and
  `artifact_final_entity_signature_hashes`: opaque final-span signatures derived
  from detection artifacts. `artifact_final_entity_signature_labels` maps each
  hash to a label, but still does not include raw entity values. Use these to
  catch and triage safety regressions where total entity count is unchanged but
  the candidate lost a baseline-protected span.
- `baseline_only_candidate_covered_signature_count`,
  `baseline_only_candidate_overlapping_signature_count`, and
  `baseline_only_candidate_uncovered_signature_count`: comparison-only fields
  from `compare_strategy_pairs.py`. These split exact signature deltas into
  baseline spans protected by a containing candidate span, protected by a
  high-overlap or small keyed-boundary candidate span, or not protected by any
  candidate span metadata. Overlapping coverage sets `span_boundary_mismatch`
  and keeps the candidate in review; uncovered signatures set
  `entity_signature_loss` and fail the safety verdict.
- `baseline_only_candidate_label_mismatch_signature_count`: comparison-only
  field for baseline signatures whose raw span is covered by the candidate, but
  under a different label. This sets `covered_label_mismatch` and keeps the
  candidate in review because the value is protected but label semantics may no
  longer match replacement/audit expectations.
- `value_protection_verdict`: comparison-only pass/review/fail verdict focused
  on whether candidate output still protects baseline values. Covered
  label-mismatch spans can still pass this axis because the sensitive value is
  protected, while uncovered signatures, candidate leaks, and candidate case
  failures fail it.
- `signature_parity_verdict`: comparison-only pass/review/fail verdict focused
  on exact baseline signature semantics. Covered label mismatches and boundary
  mismatches review-gate this axis even when `value_protection_verdict` passes.
  This split is useful for DataDesigner-free experiments: a candidate can be a
  plausible protection backend while still requiring label-policy review before
  it can replace a DataDesigner-backed baseline.
- `final_entity_label_counts`: per-label entity counts serialized as JSON in
  exported tabular files.
- `ground_truth_*` and `entity_*`: exact value+label precision, recall, F1,
  false positives, and false negatives when the input includes one of the
  supported ground-truth entity columns.
- `entity_relaxed_*`: span-overlap precision, recall, and F1. The
  label-compatible variants require both span overlap and equivalent labels,
  while the non-label-compatible relaxed metrics only ask whether a
  ground-truth span was protected by any detected span.
- `empty_detection_count`, `empty_detection_rate`,
  `empty_detection_with_ground_truth_count`, and
  `empty_detection_with_ground_truth_rate`: diagnostics for records where the
  detector returned no final entities. The ground-truth-specific fields are the
  important safety signal when a benchmark includes labels.
- `utility_score`, `leakage_mass`, `weighted_leakage_rate`,
  `needs_repair`, and `needs_human_review`: rewrite-mode evaluation fields.
  These are null for replace-mode runs.

Error and reliability metrics:

- `failed_record_count`: records dropped by a DataDesigner workflow.
- `status`: completion state for a stage or workflow.
- `case_failed`: true when a benchmark case has any error-status stage or
  DataDesigner workflow measurement.
- `error_stage_count`, `error_ndd_workflow_count`, and
  `error_model_workflow_count`: error-status measurement rows counted per case.
- `failed_case_count` and `failed_case_rate`: group-level failed-case count and
  rate for a workload/config/strategy.
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

When token or request fields are missing, check `ndd_workflow.model_usage` and
`model_workflow.model_usage`. The measurement layer records deeper provider
usage only when the underlying executor returns it.
