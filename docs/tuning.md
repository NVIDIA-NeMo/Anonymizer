<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tune quality before throughput

Treat quality and performance as separate decisions. First meet a quality gate on fixed, representative data; then benchmark one performance control at a time without changing the accepted quality configuration.

Use this guide to run that tuning cycle. For symptom-first diagnosis of a specific run, see [Troubleshooting](troubleshooting.md).

1. Define the quality gate.
2. Establish and freeze a quality configuration.
3. Design a representative performance benchmark.
4. Tune throughput one layer at a time.
5. Select the smallest stable production setting that still passes the quality gate.

## 1. Define the quality gate

Build an independently adjudicated quality set that covers the data you expect to process. Keep a manageable subset for rapid iteration, and include:

- sensitive values for every entity label the deployment must protect;
- benign values that resemble sensitive values;
- representative record lengths, formats, and entity densities;
- structured fields, free text, and mixed-format records when applicable;
- repeated values and ambiguous identifiers; and
- Unicode, punctuation, overlapping spans, and boundary cases.

Document which values should be protected and which should remain unchanged. An identifier can be harmless in one dataset and sensitive in another, so preserve known field-level policy before converting structured records to free text.

Set acceptance thresholds before tuning. Use both Anonymizer diagnostics and independent checks:

| Concern | Evidence | Example acceptance rule |
| --- | --- | --- |
| Missed entities | Adjudicated spans; `entity_coverage` and `missed_entities` as judge-based diagnostics | Minimum recall overall and for each required label |
| False positives | Adjudicated spans; optional `detection_valid` and `detection_invalid_entities` diagnostics | Minimum precision overall and for each required label |
| Output privacy | Deterministic checks for retained original values; Rewrite leakage metrics | Maximum permitted leakage |
| Utility | Task-specific review; Rewrite `utility_score` | Minimum permitted utility |
| Reliability | `failed_records`, retries, and request failures | Maximum permitted failure rate |

`Anonymizer.evaluate()` does not produce adjudicated span precision, relaxed matching scores, or an oversanitization rate. Calculate those measures separately when the policy requires them. LLM judges are useful diagnostics, not ground truth; see [Evaluation](concepts/evaluation.md) for the fields the SDK emits.

## 2. Establish and freeze a quality configuration

Replace and Rewrite share a core detection sequence: the GLiNER detector proposes spans, a large language model (LLM) validates them, and an LLM augmenter finds possible misses. Rewrite adds latent-entity detection and rewrite-specific stages. Changing `gliner_threshold` affects only the GLiNER proposals; it does not constrain the augmenter or correct validator decisions. See [Detect](concepts/detection.md) and [Rewrite](concepts/rewrite.md) for the canonical pipeline descriptions.

Choose the detection taxonomy, data context, anonymization mode, and model assignments before tuning performance:

- With `entity_labels=None` (permissive mode), the augmenter may infer labels beyond the defaults. An explicit list enables strict mode and limits output to those labels. Strict mode controls the taxonomy, not individual values or fields.
- `AnonymizerInput.data_summary` supplies context to detection prompts and, in Rewrite mode, rewrite prompts. It does not change local Replace strategies or Substitute replacement-map prompts, and it is not a deterministic exclusion rule.
- Assign models by measured quality, latency, and cost. Keep aliases in one validator pool behaviorally equivalent in quality, context limits, and safety settings.

See [Choosing a strategy](concepts/choosing-a-strategy.md) for label, `data_summary`, and mode configuration, and [Models](concepts/models.md) for model roles and validator pools.

### Inspect and evaluate detection

Use `Annotate` on a small preview to inspect detected spans and labels, then run the optional detection-validity judge. The example assumes `records.parquet` is a representative local input and that the default model credentials are configured.

```python
from anonymizer import (
    Annotate,
    Anonymizer,
    AnonymizerConfig,
    AnonymizerInput,
    Detect,
    EvaluateConfig,
)

anonymizer = Anonymizer()
data = AnonymizerInput(
    source="records.parquet",
    text_column="text",
    data_summary=(
        "Representative records from the target workload. Protect sensitive values "
        "defined by the deployment policy and treat structural syntax as metadata."
    ),
)
detect = Detect()

preview = anonymizer.preview(
    config=AnonymizerConfig(detect=detect, replace=Annotate()),
    data=data,
    num_records=50,
)
evaluated = anonymizer.evaluate(
    preview,
    config=EvaluateConfig(compute_detection_validity=True),
)
```

`Annotate` preserves the original sensitive text and is not a privacy-safe output. Use it only for inspection. Rewrite produces leakage and utility metrics during `run()` and `preview()`; a later `evaluate()` call adds judge-based results. Replace evaluation is post-hoc. See [Evaluation](concepts/evaluation.md) for both modes.

### Change one quality variable at a time

Run every candidate against the same quality set. Each experiment should answer one question:

| Variable | Question | Hold fixed |
| --- | --- | --- |
| `entity_labels` | Does this taxonomy match the deployment policy? | Models, threshold, validator settings, and prompts |
| `gliner_threshold` | Does a higher or lower proposal threshold improve the accepted precision-recall trade-off? | Labels, validator settings, and models |
| `validation_max_entities_per_call` | Does validator chunk size affect accuracy or failures on dense records? | Threshold, excerpt size, and validator models |
| `validation_excerpt_window_chars` | Does more or less surrounding context improve disambiguation? | Threshold, chunk size, and validator models |
| Model assignment | Does a different model meet the quality gate for this role? | Detection controls and all other role assignments |

After each change, inspect the retained output and rerun the acceptance checks. Reject changes that improve averages while failing a required label or edge case.

### Record the freeze point

Before performance testing, record:

- the quality-set revision and adjudication rules;
- the complete `AnonymizerConfig` and `AnonymizerInput.data_summary`;
- approved model identities, provider mappings, non-concurrency inference parameters, and validator aliases;
- baseline admission limits, pool topology, and scheduler settings;
- Anonymizer and model-service versions; and
- acceptance thresholds and observed results.

During each performance arm, change only the control named for that arm. A change to the corpus, entity policy, prompts, model identity, or non-concurrency inference parameters requires a new quality-gate run and freeze point.

## 3. Design a representative performance benchmark

Record count and total tokens do not fully describe a workload. Record length, entity density, format, and the number of independent rows determine which limits the pipeline exercises.

A benchmark with many independent records does not predict latency for one very long record. Test long records separately. Detection and validation can chunk work, but other LLM stages may process a whole record; do not assume end-to-end chunking for arbitrarily large inputs.

Run the benchmark in the target managed or self-hosted environment. Approve its resource and cost budgets, and define stop thresholds before generating load.

Build fixed benchmark slices across the workload's record-length and entity-density distributions. Include no-entity rows, duplicates, supported formats, and the edge cases from the quality set. Run one warm-up followed by at least three measured repetitions. If the provider or shared infrastructure can drift, run the low-concurrency baseline before and after each candidate.

For the baseline, keep alias topology fixed and set each available `max_parallel_requests` limit to `1`. Set `RunConfig(max_in_flight_tasks=1)` so DataDesigner holds one task lease at a time. A validator pool can still offer one lane per alias, so record the pool topology rather than describing this baseline as fully serial.

```python
from anonymizer import Anonymizer, RunConfig

anonymizer = Anonymizer(
    model_configs="models.yaml",
    model_providers="providers.yaml",
    data_designer_run_config=RunConfig(max_in_flight_tasks=1),
)
```

Capture at least:

- end-to-end and per-stage elapsed time;
- request queue and service time where available;
- configured and measured concurrency;
- requests, input and output tokens, retries, failures, and rate limits;
- host CPU and memory, plus accelerator utilization and memory when applicable; and
- quality results on the exact retained outputs, outside the timed block.

The local measurement API records stage timings, model usage, and failures. It can also write scheduler traces when task tracing is enabled. See [Observability](development/observability.md) for setup and artifact formats.

Do not include post-hoc `Anonymizer.evaluate()` calls in the primary timed run. Run them afterward on the retained output. Rewrite's built-in leakage and utility scoring remains part of the Rewrite execution path and should stay inside its end-to-end measurement.

For Replace, `Redact` can isolate detection cost because it does not generate substitutes. If production uses `Substitute`, benchmark that complete path separately before selecting a production setting.

## 4. Tune throughput one layer at a time

Compare each candidate with the current accepted setting, then keep or reject it before moving to the next layer:

| Layer | Change | Hold fixed |
| --- | --- | --- |
| Detector admission | Increase the detector alias's `max_parallel_requests` gradually | GLiNER batch policy and all LLM settings |
| Validator admission | Increase validator alias limits gradually | Pool topology, chunk size, excerpt size, and model |
| Validator topology | Compare one alias with multiple equivalent aliases | Same total client-side alias limits and provider quotas |
| Validator chunking | Vary `validation_max_entities_per_call` | Validator model, excerpt size, and total client-side admission |
| Augmenter admission | Increase the augmenter alias's `max_parallel_requests` gradually | Selected detector and validator settings |
| DataDesigner scheduling | Vary `max_in_flight_tasks` and `buffer_size` separately | Selected per-alias limits |

`max_parallel_requests` applies per model alias. A validator pool's client-side admission limit is bounded by the sum of its alias limits, but provider quotas, routing, and shared backends may cap actual concurrency. Treat the sum as offered load, not guaranteed model capacity. Multiple aliases that point to the same backend do not create capacity, although they may expose capacity that was previously idle. See [Validator pools](concepts/models.md#validator-pools).

After every candidate run, apply the same quality gate to the retained output. Stop increasing a layer when measured concurrency stops rising, additional concurrency produces little throughput improvement, tail latency or queues become unstable, resource use reaches its limit, or failures and rate limits increase.

### Optional deployment-specific controls

- **Self-hosted GLiNER:** The reference server can coalesce concurrent requests. Vary coalescing only after selecting detector-side admission, and measure queue delay and available host or accelerator utilization. See [Self-hosting GLiNER](concepts/self-hosting-gliner.md#reference-implementation) for its controls and defaults.
- **External detection execution:** `export_detection_config()` and `export_detection_builder_for_seed(..., job_index=..., num_jobs=...)` can export and partition detection for an external DataDesigner executor. They do not distribute the complete Replace or Rewrite pipeline. See the [`Anonymizer` API reference](reference/anonymizer/interface/anonymizer.md).

## 5. Select the production setting

Use the same workload shape that production will receive; many short records and a small number of long records require different tests.

Select the smallest configuration that satisfies all of these conditions:

- every quality threshold passes on the retained output;
- throughput improvement is repeatable across measured runs;
- tail latency, failures, retries, and rate limits remain within their limits;
- host, accelerator, and provider usage remain within the approved budget; and
- the complete production configuration and supporting measurements are recorded.

The goal is a stable setting near the point of diminishing returns, where more concurrency adds little throughput, not the largest concurrency value that completes.
