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
| `evaluation` | `EvaluationCriteria()` | Thresholds and criteria for leakage/utility scoring. |

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

### Evaluation criteria

Controls the repair loop and human review flagging:

| Field | Default | Description |
|-------|---------|-------------|
| `risk_tolerance` | `conservative` | Preset leakage threshold: `strict` (0.6), `conservative` (1.0), `moderate` (1.5), `permissive` (2.0). |
| `max_leakage_mass` | `None` | How much `leakage_mass` is still okay before repair should try again; if you set this, it overrides `risk_tolerance` and domain-based limits. |
| `auto_adjust_by_domain` | `False` | Tighten thresholds automatically for high-risk domains ([domain → risk level map](https://github.com/NVIDIA-NeMo/Anonymizer/blob/main/src/anonymizer/config/rewrite.py#L28)). |
| `repair_any_high_leak` | `True` | Trigger repair if any high-sensitivity entity leaks. |
| `max_repair_iterations` | `2` | Maximum repair rounds. Only applicable when `auto_repair_privacy=True` |
| `auto_repair_privacy` | `True` | Enable automatic repair loop. |
| `flag_utility_below` | `0.50` | Flag for human review if utility score is below this. |
| `flag_leakage_mass_above` | `2.0` | Flag for human review if leakage mass exceeds this. |

---

## Output columns

| Column | Description |
|--------|-------------|
| `{text_col}_rewritten` | The privacy-safe rewritten text. |
| `utility_score` | Quality preservation (0.0--1.0). Higher is better. |
| `leakage_mass` | Weighted privacy leakage. Lower is better. |
| `any_high_leaked` | Whether any high-sensitivity entity leaked through. |
| `needs_human_review` | Flag for records that may need manual review. |

Use `preview.trace_dataframe` for the full pipeline trace (domain, disposition, QA pairs, repair iterations, judge evaluation).

!!! note "No entities? No rewrite."

    Records with no detected entities pass through unchanged with `utility_score=1.0` and `leakage_mass=0.0`.

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
