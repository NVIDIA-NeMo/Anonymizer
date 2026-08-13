---
date:
  created: 2026-08-07
readtime: 10
authors:
  - memadi-nv
---

# **After Anonymization — Part II: Evaluating Rewrite Mode**

<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Let's return to the customer biographies from Part 1. Replace mode changed the explicit identifiers. Now suppose the data must meet a stricter privacy requirement: even after names and addresses are replaced, a rare occupation, an exact sequence of life events, or a distinctive combination of hometown and employer may still identify someone.

With that stricter goal in mind, we recommend to use NeMo Anonymizer's Rewrite mode. It can account for latent entities and combinations of identifying clues by transforming the full record—not just explicit identifiers—while preserving the meaning that still matters. The challenge is to reduce identifying detail without losing useful information.

During `run()` or `preview()`, Rewrite checks each generated record for privacy leakage and meaning preservation. Leakage results determine whether a record enters the repair loop, while both leakage and utility contribute to the final human-review flag. After anonymization, an optional `evaluate()` call reviews entity coverage and the rewrite's privacy, quality, and style. Detection validity is separately opt-in.

This is Part 2 of a two-part series on evaluation in Anonymizer. Part 1 covers Replace mode; this article explains the two evaluation layers used by **Rewrite mode**, what each score means, and how to inspect the results.

<!-- more -->

---

## Rewrite Evaluation Has Two Layers

Unlike Replace mode, Rewrite mode evaluates every generated rewrite as part of `preview()` or `run()`. A separate `evaluate()` call adds independent LLM-as-judge feedback:

```mermaid
flowchart TD
    A[Source data] --> B[Anonymizer.preview / run\nDetect and rewrite]
    B --> C[Evaluate–repair loop\nUtility · leakage · repair]
    C --> D[Rewrite result]
    D --> E[Optional Anonymizer.evaluate]
    E --> F[Post-hoc judge report\nCoverage · privacy · quality · style]
```

Rewrite first scores each record for privacy leakage and utility. If leakage is too high, it rewrites the record and checks it again. When the evaluate–repair loop ends, it flags records that still need human review. Post-hoc judge evaluation is an optional, separate step that you run by calling `evaluate()` after anonymization. It adds informational judge outputs without rewriting the text again or changing the existing human-review decision, and you can configure a different LLM for each judge role. See [Part 1](evaluation-anonymizer-replace.md#anonymization-and-evaluation-as-separate-steps) for more about this workflow.

```python
import pickle

from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Rewrite

anonymizer = Anonymizer()

result = anonymizer.run(
    config=AnonymizerConfig(rewrite=Rewrite()),
    data=AnonymizerInput(source="records.csv", text_column="text"),
)

# Rewrite metrics are already present after run().
rewrite_metric_columns = [
    "utility_score",
    "leakage_mass",
    "weighted_leakage_rate",
    "needs_human_review",
]
rewrite_scores = result.dataframe[rewrite_metric_columns]

# To run the optional post-hoc evaluation immediately, use:
# evaluated = anonymizer.evaluate(result)

# To evaluate in a later session, save the complete result.
with open("rewrite-result.pkl", "wb") as f:
    pickle.dump(result, f)

# In the later session, reload the complete result before evaluating it.
with open("rewrite-result.pkl", "rb") as f:
    saved_result = pickle.load(f)

evaluated = anonymizer.evaluate(saved_result)
posthoc_scores = evaluated.dataframe[["entity_coverage", "judge_evaluation"]]
```

As in Replace mode, the completed result is reusable. You can run `evaluate()` immediately or save the complete `AnonymizerResult` or `PreviewResult` with Python's `pickle` module and evaluate it later. Save the result object—not just its `dataframe`—so the Rewrite configuration and evaluation context remain available.

---

## Evaluate–Repair Loop: Did the Rewrite Balance Privacy and Utility?

During `run()` or `preview()`, Anonymizer creates quality questions from the original record and privacy questions from the detected entities and their sensitivity dispositions. It answers those questions against the rewritten text, computes per-record metrics, and repairs rows that exceed the configured privacy threshold.

### Utility Score: Was Important Meaning Preserved?

`utility_score` measures how well the rewritten text answers questions about the original record. Scores range from `0.0` to `1.0`; higher is better.

Before rewriting, Anonymizer extracts pieces of information to preserve, called meaning units, and labels each one as `critical` or `important`. When computing `utility_score`, a critical meaning unit receives twice the weight of an important one:

```text
utility_score = weighted mean of per-question answer scores
```

A high score means the rewrite retained the important facts and relationships. It does not require verbatim wording or every identifying detail to survive.

### Leakage Mass: What Identifying Information Remains?

The privacy check tests whether each protected value can still be identified or inferred from the rewrite. Every leaked item contributes its sensitivity weight multiplied by the judge's confidence:

```text
leakage_mass = Σ(sensitivity_weight × leak_confidence)
```

High-, medium-, and low-sensitivity items carry weights of `1.0`, `0.6`, and `0.3`. Lower leakage mass is better, but it is not bounded at `1.0` because several items can leak in one record.

`weighted_leakage_rate` normalizes that mass by the maximum possible leakage for the record:

```text
weighted_leakage_rate = leakage_mass / maximum_possible_leakage_mass
```

It ranges from `0.0` to `1.0`. `any_high_leaked` separately records whether at least one high-sensitivity item leaked.

<div class="output-columns-table" markdown>

| Output column | Meaning |
|---|---|
| `utility_score` | Meaning preservation from `0.0` to `1.0`; higher is better |
| `leakage_mass` | Confidence-weighted sum of leaked items; lower is better |
| `weighted_leakage_rate` | Leakage normalized to the record's maximum possible mass |
| `any_high_leaked` | Whether any high-sensitivity item leaked |

</div>

### Repair Loop: Can a Failing Rewrite Be Improved?

After the initial checks, rows above the repair threshold—or rows with a high-sensitivity leak when the selected risk tolerance requires it—enter the repair loop. Only failing rows are repaired and checked again. The loop stops when they pass or reach `max_repair_iterations`.

```mermaid
flowchart LR
    A[Evaluate rewrite] --> B{Needs repair?}
    B -->|No| C[Keep result]
    B -->|Yes| D[Repair failing row]
    D --> E[Re-evaluate]
    E --> B
```

`risk_tolerance` controls the repair and review thresholds. The default is `low`; use `minimal` for a stricter posture or `moderate` / `high` for more tolerance. Setting `max_repair_iterations=0` disables repair but still computes the rewrite metrics.

### Human Review: Which Records Still Need Attention?

After repair ends, `needs_human_review` becomes `True` when the rewrite is missing, a high-sensitivity item still leaks, utility falls below the selected preset's threshold, or leakage mass exceeds its review threshold.

<div class="output-columns-table" markdown>

| Output column | Meaning |
|---|---|
| `needs_human_review` | Final review flag based on the rewrite metrics |

</div>

Records with no detected entities skip rewriting and pass through unchanged with `utility_score=1.0`, `leakage_mass=0.0`, and `needs_human_review=False`.

---

## Post-Hoc Evaluation: What Does an Independent Judge See?

The post-hoc report includes entity coverage and a holistic rewrite judgment, with detection validity available as an opt-in audit.

### Entity Coverage: Were All In-Scope Entities Detected?

Entity coverage works the same way in Rewrite and Replace modes: an independent judge identifies in-scope candidates in the original text and measures how many Anonymizer detected. See [Part 1: Entity Coverage](evaluation-anonymizer-replace.md#entity-coverage-were-all-in-scope-entities-detected) for the calculation, output columns, and interpretation guidance.

### Privacy: Did the Rewrite Reduce Linkage Risk?

The privacy rubric compares the rewritten record with the original and estimates whether a realistic attacker could link them. It considers surviving direct identifiers and distinctive combinations of quasi-identifiers.

- `high` — direct identifiers are removed and remaining details create low linkage risk.
- `medium` — no obvious direct identifier remains, but a distinctive bundle creates noticeable risk.
- `low` — the record remains easily or near-certainly linkable.

### Quality: Was the Record's Meaning Preserved?

The quality rubric evaluates important facts, relationships, chronology, and conclusions independently from privacy and writing style.

- `high` — important meaning, facts, and structure are fully preserved.
- `medium` — most content is preserved with minor loss or distortion.
- `low` — important information is materially lost, contradicted, or distorted.

### Style: Does the Rewrite Read Naturally?

The style rubric evaluates fluency, grammar, clarity, coherence, and whether the result reads like human-written prose.

- `high` — fluent, coherent, and natural.
- `medium` — readable with isolated awkward phrasing.
- `low` — noticeably unnatural, broken, or placeholder-like.

All three rubric results are stored together in `judge_evaluation`. To retrieve the value for the first record:

```python
judge_evaluation = evaluated.dataframe.loc[0, "judge_evaluation"]
```

An example value looks like this:

```json
{
  "privacy": {"score": "high", "reasoning": "..."},
  "quality": {"score": "medium", "reasoning": "..."},
  "style": {"score": "high", "reasoning": "..."}
}
```

The `reasoning` field explains why the selected rubric level applies, using evidence from the original and rewritten text.

These rubric scores are informational. They do not trigger repair and do not modify `needs_human_review`.

### Detection Validity: Were the Detected Entities Valid in Context?

Detection validity uses the same opt-in judge as Replace mode to check whether detected `(value, label)` pairs are valid in context. Rewrite reports the fraction that passed from `0.0` to `1.0`, aligning it with Rewrite's numeric scoring, whereas Replace reports whether all detections passed as a boolean. See [Part 1: Detection Validity](evaluation-anonymizer-replace.md#detection-validity-were-the-detected-entities-valid-in-context) for what it flags and how to enable it.

---

## Reading the Report

To inspect the first record (row `0`):

```python
evaluated.display_record(0)
```

The Rewrite report shows the original and rewritten text, rewrite utility and leakage metrics, the human-review flag, post-hoc privacy / quality / style scores, entity coverage, and the entity disposition used to guide protection. When detection validity is enabled, that score and any flagged detections also appear.

<div style="text-align: center;" markdown>

![Screenshot of display_record output for a Rewrite result showing original and rewritten text, rewrite metrics, post-hoc judge scores, entity coverage, and entity disposition.](assets/evaluate-rewrite-display-record.png){ loading=lazy }

</div>

For a tabular summary across records:

```python
evaluated.dataframe[[
    "utility_score",
    "leakage_mass",
    "weighted_leakage_rate",
    "needs_human_review",
    "entity_coverage",
    "detection_valid",    # Disabled by default, added only when compute_detection_validity=True
    "judge_evaluation",
]]
```

---

## Comparing Judge Models

Judge models are configured and compared the same way as in [Part 1: Comparing Judge Models](evaluation-anonymizer-replace.md#comparing-judge-models): run `evaluate()` on the same result with different model configurations, without rerunning anonymization. See [Evaluation: Model Roles](../../concepts/evaluation.md#model-roles) for the default model used by each evaluation step, which you can treat as the baseline when comparing candidate models.

---

## What the Scores Do Not Prove

The metrics and judges answer bounded questions. They do not certify that a rewrite is anonymous.

- **Utility score** measures preservation through generated questions; it can miss meaning that no question captured.
- **Leakage metrics** test known protected values and inferences; they do not model every possible attacker or external dataset.
- **Entity coverage** depends on one judge's candidate extraction and is not ground truth.
- **Detection validity** measures precision of detected entities, not whether all sensitive information was found.
- **Privacy, quality, and style** are coarse holistic judgments, not guarantees.

Use these signals together. Review representative records, inspect failures and entity dispositions, evaluate against annotated data when available, and apply human review where the risk requires it.

---

## The Bottom Line

Rewrite evaluation is not a single final score. The evaluate–repair loop measures utility and leakage, repairs failing rows, and flags unresolved risk. The optional post-hoc judge evaluation adds independent evidence about detection coverage and the rewrite's overall privacy, quality, and style.

The two layers serve different purposes: rewrite metrics guide repair during anonymization, while post-hoc judges provide a broader assessment of the completed results.

Skipping post-hoc evaluation can leave detection gaps and holistic privacy, quality, or style problems unnoticed until affected records reach downstream systems.

For API details and the complete output schema, see [Evaluation](../../concepts/evaluation.md). For Rewrite configuration and risk tolerance, see [Rewrite](../../concepts/rewrite.md).
