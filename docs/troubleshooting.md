<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Troubleshooting

Symptom-first guide to common problems and how to fix them. Each entry says how to diagnose, what knob to turn, and what to verify after.

When something looks wrong, **first confirm the run completed cleanly** (no `failed_records` — these are rows that didn't make it through the pipeline at all, usually a rate-limit / infra issue). Once you know the pipeline ran, **run `preview` and inspect rows with quality issues** (`needs_human_review=True`) — the trace columns it produces are where to start.

!!! note "This guide is written against the Python API"

    Diagnostic objects like `result.failed_records` and `result.trace_dataframe` only exist in Python. The CLI is for production batch runs and only emits a per-stage summary line on stderr (`📋 Detection complete — N entities … (K failed) [Xs]`); if `K > 0` you have a problem to investigate, but the CLI can't tell you *which* rows or *why*. Drop into Python (or import the same config from the agent's config file) for everything below.

---

## Diagnostics: what to read first

Most fixes start with one of these.

### Did the run actually complete cleanly?

Before debugging quality, confirm the run wasn't degraded by infra problems. Anonymizer never silently drops rows — every dropped record appears on `result.failed_records` with full provenance:

```python
for fr in result.failed_records:
    print(fr.record_id, fr.step, fr.reason)
```

Common patterns and what they mean:

| `reason` substring | Likely cause | Action |
|---|---|---|
| `Record missing from workflow output` at `step="detection"` | Validator pool exhausted on at least one chunk for this row — usually rate limiting (429s) burning through the AIMD throttler's retry budget | **Model/provider config issue, not strategy.** See [Rate limits / dropped rows](#rate-limits-dropped-rows) below |
| `Record missing from workflow output` at `step="replace-map-generation"` or `step="rewrite-*"` | Same as above but for the replace/rewrite stage | Same fix |
| `Output is missing required tracking column` | Internal — file an issue | — |

If `failed_records` is non-empty, **fix that first.** Strategy/prompt knobs (`risk_tolerance`, `protect`, `gliner_threshold`) won't help — those rows didn't fail because of bad output, they failed because the model never returned an output at all.

### Rate limits / dropped rows

If you have rows failing with `Record missing from workflow output`, the chain is:
GLiNER candidate set → chunked into validator calls → each call dispatched to a model in the validator pool → `ThrottledModelClient` does AIMD on 429s → if every alias in the pool fails on at least one chunk, the row drops.

Fix in this order:

1. **If failures are at `step="detection"`, add aliases to the validator pool** in `models.yaml`. The validator is the only role that supports a pool — set `entity_validator` to a list of aliases and chunked validation will round-robin across them, giving you failover when one provider rate-limits. Every other role (detector, augmenter, rewriter, evaluator, etc.) is a single alias.
2. **Lower `validation_max_entities_per_call`** (default `100`) on `Detect` so each call sends fewer tokens — easier on tight per-minute token budgets. Helps any stage that's hitting per-call token limits, but most useful for validation.
3. **Switch the heavy alias to a different `provider`** in `providers.yaml`. If you're hammering one tenant's quota, moving to a second deployment of the same model helps more than tuning batch sizes. This is the only lever for non-validator stages (rewrite, evaluate, etc.) since they don't have pools.
4. **Re-run on just the failed records** — filter the input dataframe to those `record_id`s and call `anonymizer.run` again. Failures are usually transient.

See [Models](concepts/models.md) and [Validator pools](concepts/models.md#validator-pools) for the config shape.

### Read the preview trace

Rewrite mode `preview` returns intermediate columns alongside the rewritten text. Inspect them via `result.trace_dataframe`:

```python
result = anonymizer.preview(config=config, data=data, num_records=5)
result.trace_dataframe[[
    "_domain",
    "_sensitivity_disposition",
    "leakage_mass",
    "utility_score",
    "needs_human_review",
]]
```

Key columns:

| Column | What it tells you |
|---|---|
| `_domain` | Which domain the classifier picked. Wrong domain → wrong supplement → poor rewrite |
| `_sensitivity_disposition` | Per-entity sensitivity assignments (high/medium/low) |
| `leakage_mass` | Confidence-weighted sum of leaked entities |
| `utility_score` | 0–1 quality preservation score |
| `weighted_leakage_rate` | Leakage normalized by maximum possible leakage |
| `any_high_leaked` | Whether any high-sensitivity entity leaked through |
| `needs_human_review` | Crossed the configured threshold |
| `_judge_evaluation` | Final-judge qualitative comments |

### Re-run with `Annotate` to see detection output

When you suspect a detection problem (missed entities, weird labels), run a tiny `preview` with `replace=Annotate()` against the same `Detect` config. The output text shows `<original, label>` for every detected entity in place — easier to eyeball than the trace columns.

```python
from anonymizer import Annotate, AnonymizerConfig

debug_config = AnonymizerConfig(detect=detect, replace=Annotate())
preview = anonymizer.preview(config=debug_config, data=data, num_records=5)
print(preview.dataframe.iloc[0][f"{data.text_column}_with_spans"])
```

---

## Detection problems

### Detection missed an entity I expected

Try in order:

1. **Lower `gliner_threshold`** from `0.3` to `0.2` (or `0.15`). False positives get caught downstream by validation.
2. **Extend the default list** with the entity's label if it's not in `DEFAULT_ENTITY_LABELS`. Setting `Detect.entity_labels` to a custom list switches detection to **strict mode** (only listed labels detected, augmenter can't invent), so to keep the defaults *plus* one extra label use:

   ```python
   from anonymizer import DEFAULT_ENTITY_LABELS, Detect
   detect = Detect(entity_labels=[*DEFAULT_ENTITY_LABELS, "clinical_facility"])
   ```

   Domain-specific labels (`clinical_facility`, `case_number`, `internal_project_codename`) won't be detected reliably without being listed this way.
3. **Set `AnonymizerInput.data_summary`** so the augmenter LLM has domain context. A line like `"De-identified pediatric oncology progress notes"` materially improves coverage.
4. **For rewrite mode**, latent entities are detected separately. If a piece of inferable information (e.g. "during her third round of chemo" → cancer treatment) is being preserved verbatim, the latent detector likely missed it — refine `Rewrite.privacy_goal.protect` to call out the inference category explicitly.

Verify by re-running `preview` with `Annotate` and confirming the entity now appears tagged.

### Too many false-positive entities

Symptoms: detected entities include obvious common words, dates that aren't dates, etc.

1. **Raise `gliner_threshold`** to `0.5`. The augmenter will pick up real misses, so this rarely costs recall.
2. **Lower `validation_excerpt_window_chars`** (default `500`) if context-driven validation is being misled by far-away sentences. Smaller per-chunk prompts trade context for precision.
3. **Sanity-check the validator with an `Annotate` preview.** A flaky validator (or a misconfigured alias) returns "keep" on almost everything, which presents as recall going way up — easiest spotted by eyeballing the entity list on a handful of rows.

### A new domain isn't being detected well

Symptom: rewrite output is generic-sounding even though the input is clearly in a specialized domain.

1. Inspect `_domain` in `result.trace_dataframe`. If it shows `general` or an unrelated domain, the classifier is missing the cue.
2. Set `AnonymizerInput.data_summary` to name the domain explicitly.
3. If your domain isn't represented in `DOMAIN_SUPPLEMENT_MAP`, the engine falls back to generic supplements and rewrite quality suffers. This is a code-level extension — file an issue, or add the domain to `src/anonymizer/engine/rewrite/domain_classification.py`.

---

## Rewrite quality

### `leakage_mass` is too high

`leakage_mass` is a confidence-weighted sum of leaked entities (high=1.0, medium=0.6, low=0.3). Targets vary by `risk_tolerance`:

| Tolerance | Repair triggers above | Flagged for review above |
|---|---|---|
| `minimal` | 0.6 | 1.0 |
| `low` | 1.0 | 2.0 |
| `moderate` | 1.5 | 2.5 |
| `high` | 2.0 | 3.0 |

If you're consistently above your threshold:

1. **Tighten `risk_tolerance`** one step (e.g. `low` → `minimal`). Cheapest knob.
2. **Refine `privacy_goal.protect`** to name the categories that are leaking. Inspect `_sensitivity_disposition` to see which entities the engine deemed protected — anything classified `low` may be slipping through.
3. **Set `strict_entity_protection=True`** to force every detected entity into a protective disposition.
4. **Increase `max_repair_iterations`** from `3` to `5` if the trace shows leakage shrinking across iterations but not finishing.
5. **Detection coverage** — leakage can't be fixed if the entity wasn't detected. Walk back through "[Detection missed an entity I expected](#detection-missed-an-entity-i-expected)" before giving up.

### `utility_score` is too low

`utility_score` measures how well meaning was preserved (0–1). Below ~0.5 is usually unusable; the human-review threshold depends on `risk_tolerance` (0.3–0.6).

Most common causes:

1. **`protect` is too aggressive** — it's removing things downstream tasks need. Move the over-suppressed content into `preserve`.
2. **`preserve` is too vague** — generic phrasing like "preserve meaning" gives the rewriter no signal. Name the specific facets that matter (clinical findings, argument structure, timeline, etc.).
3. **`risk_tolerance="minimal"` plus `strict_entity_protection=True`** is the most aggressive combination and can over-modify. Loosen one of the two if downstream task quality matters more than blanket coverage.
4. **Repair loop is over-correcting** — inspect repair iterations in the trace. If utility falls each iteration, lower `max_repair_iterations`.

### Most rows have `needs_human_review=True`

Three failure modes look the same in the column:

| Cause | Diagnosis |
|---|---|
| Leakage too high | `weighted_leakage_rate` near 1, or `any_high_leaked=True` |
| Utility too low | `utility_score` below `flag_utility_below` for your tolerance |
| Both | Almost always means `protect` and `preserve` are pulling in opposite directions |

For the third case, the fix is rewriting the privacy goal so it draws a cleaner line between "remove this" and "keep this." See [Choosing a strategy > Privacy goal](concepts/choosing-a-strategy.md#5-rewrite-privacy-goal).

### Repair runs every iteration but never converges

Symptom: every record uses all `max_repair_iterations` and still ends up flagged.

1. The leakage threshold for your `risk_tolerance` may be unreachable for your data. Look at the *floor* of `leakage_mass` across repair iterations — if it plateaus above the threshold, the data has more sensitive content than the threshold permits given current detection coverage.
2. Loosen `risk_tolerance` one step, **only after** confirming detection has caught everything you can see. Loosening before detection is solid just hides the leak.
3. Set `max_repair_iterations=0` for an audit pass. You'll get the metrics without paying for repair attempts that won't succeed, which makes it easy to see how far off you are.

---

## Pipeline / output issues

### Output rows are missing (`FailedRecord`s)

See [Did the run actually complete cleanly?](#did-the-run-actually-complete-cleanly) and [Rate limits / dropped rows](#rate-limits-dropped-rows). The short version: this is almost always a model/provider issue, not a strategy issue, and the fix is in `models.yaml` / `providers.yaml`.

### Replacement looks repetitive or unrealistic (Substitute)

Symptoms: every "Alice" becomes "Maya," every city becomes "Springfield," every email becomes `john@example.com`.

1. **Add domain hints to `Substitute.instructions`** — see [Choosing a strategy > Writing Substitute.instructions](concepts/choosing-a-strategy.md#writing-substituteinstructions).
2. **Check the `replacement_generator` model** — small models default-collapse to high-frequency names. Try a stronger model for this role.
3. **`Substitute` is consistent within a row, not across rows.** Repeated mentions of the same value in one row collapse to one replacement (so a person's name stays consistent in a single document). Across rows the LLM has no shared state, so "Alice" in different rows usually gets different replacements. If you need stable cross-row mappings, use `Hash` or post-process `result.trace_dataframe["_replacement_map"]`.

### Hash output isn't stable across runs

Hashes are deterministic given the same `algorithm`, `digest_length`, and input text. Instability almost always means:

- The detected text differs (different whitespace, casing, surrounding context). Look at the `<your_text_column>_with_spans` column to confirm.
- The label differs between runs (label normalization is lossy — e.g. `first_name` vs `FIRST_NAME` collapse to the same value, but `name` vs `first_name` do not).

The hash itself is a function of `text`, not of `label`. If the *original entity text* is identical, the digest is identical.

### Validation passed but `preview` errors at LLM call

Configuration is structurally valid but a runtime model call failed. Check:

1. The provider for the model alias has an API key set in your environment.
2. The base URL is reachable (corporate VPN / proxy).
3. The model alias actually exists at the provider — `anonymizer validate` checks the alias is in your config; it doesn't dial out to confirm the model is live.

