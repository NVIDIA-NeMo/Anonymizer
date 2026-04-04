<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Rewrite

Instead of replacing individual entities, rewrite mode generates a privacy-safe transformation of the entire text, preserving semantic meaning while obscuring sensitive information.

---

## How it works

[Detection](detection.md) runs first (same as [replace mode](replace.md), plus latent entity detection for context-inferable information). Then the text is classified by domain, each entity gets a sensitivity disposition, and an LLM transforms the text accordingly. The result is evaluated for quality and privacy leakage, with automatic repair if thresholds are exceeded. A final judge sets a `needs_human_review` flag.

---

## Basic usage

```python
from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Rewrite

anonymizer = Anonymizer()

config = AnonymizerConfig(rewrite=Rewrite())
data = AnonymizerInput(source="data.csv", text_column="text")

preview = anonymizer.preview(config=config, data=data, num_records=3)
preview.display_record()
```

---

## Configuration

| Field | Default | Description |
|-------|---------|-------------|
| `privacy_goal` | Auto-populated | What to protect and what to preserve. |
| `instructions` | `None` | Additional instructions for the rewrite LLM. |
| `risk_tolerance` | `low` | Preset controlling repair and review thresholds: `minimal`, `low`, `moderate`, `high`. |
| `max_repair_iterations` | `2` | Maximum repair rounds. Set to 0 to disable repair. |

### Privacy goal

`PrivacyGoal` tells the rewriter what matters. Defaults work for general-purpose data. Override for domain-specific needs:

```python
from anonymizer.config.rewrite import PrivacyGoal

config = AnonymizerConfig(
    rewrite=Rewrite(
        privacy_goal=PrivacyGoal(
            protect="All patient identifiers and clinical facility names",
            preserve="Clinical findings, treatment plans, and medical terminology",
        )
    )
)
```

!!! tip "Be specific"

    The more precise the `protect` and `preserve` fields, the better the rewriter targets sensitive content while retaining what matters.

### Risk tolerance

Controls the automated repair loop and human review flagging. Each preset bundles a coherent set of behaviors:

| Preset | Repair threshold | Repair on any high leak | Flag utility below | Flag leakage above |
|--------|-----------------|------------------------|--------------------|--------------------|
| `minimal` | 0.6 | Yes | 0.6 | 1.0 |
| `low` | 1.0 | Yes | 0.5 | 2.0 |
| `moderate` | 1.5 | Yes | 0.4 | 2.5 |
| `high` | 2.0 | No | 0.3 | 3.0 |

The **repair threshold** is the leakage mass above which a record is sent for repair.
> Leakage mass is a confidence-weighted sum of leaked entities, where each entity's weight reflects its sensitivity (high=1.0, medium=0.6, low=0.3).
> A leakage mass of 1.0 roughly equals one high-sensitivity entity leaked at full confidence.

```python
config = AnonymizerConfig(
    rewrite=Rewrite(
        risk_tolerance="minimal",
        max_repair_iterations=3,
    )
)
```

---

## Output columns

| Column | Description |
|--------|-------------|
| `{text_col}_rewritten` | The privacy-safe rewritten text. |
| `utility_score` | Quality preservation (0.0--1.0). Higher is better. |
| `leakage_mass` | Weighted privacy leakage. Lower is better. |
| `weighted_leakage_rate` | Normalized leakage (0.0--1.0) relative to the maximum possible leakage mass. |
| `any_high_leaked` | Whether any high-sensitivity entity leaked through. |
| `needs_human_review` | Flag for records that may need manual review. |

Use `preview.trace_dataframe` for the full pipeline trace (domain, disposition, QA pairs, repair iterations, judge evaluation).

!!! note "No entities? No rewrite."

    Records with no detected entities pass through unchanged with `utility_score=1.0` and `leakage_mass=0.0`.

---

## Working with flagged records

Records with `needs_human_review=True` exceeded automated thresholds for leakage or utility. To investigate and resolve:

**Diagnose:** Use `trace_dataframe` to inspect the flagged record's intermediate columns — disposition, leakage breakdown, repair iterations, and judge evaluation.

```python
flagged = result.trace_dataframe[result.trace_dataframe["needs_human_review"] == True]
flagged[["utility_score", "leakage_mass", "any_high_leaked"]].head()
```

**Tune and re-run:** Adjust settings and re-run on flagged records:

- Increase `max_repair_iterations` to give the rewriter more attempts.
- Refine `privacy_goal` with more specific `protect` / `preserve` instructions for the domain.
- Lower `risk_tolerance` (e.g. `minimal`) to trigger more aggressive repair.

**Last resort:** Manually edit or exclude records that resist automated repair — some text is inherently difficult to rewrite without losing utility or leaking identifiers, and requires your judgement as the expert. 

---

## Model roles

Rewrite uses multiple LLM roles. All default to models in the [default config](models.md):

| Role | Default | Purpose |
|------|---------|---------|
| `domain_classifier` | `gpt-oss-120b` | Classifies text domain. |
| `disposition_analyzer` | `gpt-oss-120b` | Assigns sensitivity levels. |
| `meaning_extractor` | `gpt-oss-120b` | Extracts meaning units. |
| `qa_generator` | `gpt-oss-120b` | Generates QA pairs for evaluation. |
| `rewriter` | `gpt-oss-120b` | Generates the rewritten text. |
| `evaluator` | `nemotron-30b-thinking` | Evaluates quality and leakage. |
| `repairer` | `gpt-oss-120b` | Repairs high-leakage rewrites. |
| `judge` | `nemotron-30b-thinking` | Final quality/privacy judge. |
