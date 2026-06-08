<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Evaluation

Anonymizer provides LLM-as-judge evaluation for both modes, replace and rewrite, but they work differently:

| Mode | How evaluation runs |
|------|---------------------|
| **Replace** | Post-hoc, via a separate `Anonymizer.evaluate()` call after `run()` / `preview()`. |
| **Rewrite** | Built into the anonymization pipeline — runs automatically as part of every `run()` / `preview()` call. |

---

## Rewrite evaluation

Rewrite evaluation is part of the pipeline and runs automatically — there is no separate call. After the rewritten text is generated, an evaluate–repair loop scores each record for **utility** (how much semantic content was preserved) and **leakage mass** (how much sensitive information survived). Records that exceed the leakage threshold are sent back for repair, up to `max_repair_iterations` times. A final judge then produces a qualitative assessment and flags records that still need human review.

The key output columns are `utility_score`, `leakage_mass`, `weighted_leakage_rate`, `any_high_leaked`, and `needs_human_review`. See [Rewrite](rewrite.md) for the more details.

---

## Replace evaluation

Replace evaluation is **optional and post-hoc** — you call `Anonymizer.evaluate()` on a result from `run()` or `preview()`. The replace mode is read directly from the result object, so you don't restate it:

```python
from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Substitute

anonymizer = Anonymizer()
cfg = AnonymizerConfig(replace=Substitute())
src = AnonymizerInput(data="data.csv", text_column="text")

result = anonymizer.run(config=cfg, data=src)
evaluated = anonymizer.evaluate(result)
evaluated.display_record(0)
```

You can also save a `preview()` result and evaluate it in a separate session:

```python
import pickle

preview = anonymizer.preview(config=cfg, data=src, num_records=15)

with open("/tmp/preview.pkl", "wb") as f:
    pickle.dump(preview, f)

# … later …
with open("/tmp/preview.pkl", "rb") as f:
    loaded = pickle.load(f)

evaluated = anonymizer.evaluate(loaded)
```

Four LLM judges run: one that scores detection quality and three that score replacement quality (Substitute mode only). Note that all 4 scores are assigned per record.

---
### Entity Detection judge:

### Detection validity

> "Are the detected entities actually correct (value, label) pairs in context?"

This judge runs during replace evaluation regardless of which replace mode was used. It looks at each detected span and flags:

- **false_positive** — the span is not actually identifying or sensitive in this context (common word, generic phrase, boilerplate).
- **wrong_label** — the span is sensitive but the label sits in a clearly different domain (e.g. a company name labeled `first_name`). Sibling labels within the same broad domain are treated as valid.
- **not_in_text** — the literal value does not appear in the original text.
- **wrong_boundary** — the span is a clear partial or over-extended capture (omits part of the actual value, or absorbs surrounding function words). Descriptive words in natural prose around a bare entity value are not a boundary error.
- **contextual_mismatch** — the span refers to something other than the labeled entity type in this context (e.g. "Apple" as fruit labeled `company_name`).

| Output column | Type | Description |
|---|---|---|
| `detection_valid` | `bool \| None` | `True` if all detections pass; `None` if the judge was unavailable. |
| `detection_invalid_entities` | `list` | Each flagged detection with value, label, and one-sentence reasoning. |

---

### Entity Replacment judges

When the source result used the **Substitute** mode, three additional LLM judges run in parallel — one per quality dimension.

### Type fidelity

> "Does each synthetic value still belong to the same entity class and match the expected format for that class?"

The judge checks that replacements are shape-compatible with their originals — same granularity and character class — anchored by what the original itself looks like. It does **not** check semantic attributes (gender, age bucket) or cross-entity consistency; those are separate metrics.

| Output column | Type | Description |
|---|---|---|
| `type_fidelity_valid` | `bool \| None` | `True` if all replacements pass; `None` if the judge was unavailable. |
| `type_fidelity_invalid_replacements` | `list` | Each failing replacement with label, original, synthetic, and reasoning. |

### Attribute fidelity

> "Does each synthetic value preserve the salient within-entity attributes of the original?"

The judge checks attributes including:

- **Gender of name** — applies to `first_name`, `last_name`, `user_name`. Only checked when the original name clearly implies a gender. Adjacent or ambiguous cases pass.
- **Age bucket** — applies to `age` and `date_of_birth`. Buckets: child (0–12), teen (13–19), young adult (20–29), adult (30–44), middle-aged (45–64), senior (65+). Adjacent buckets pass; only clear flips (adult → child) fail.

All other labels are skipped — their attributes are either handled by other metrics or too unreliable to judge automatically.

| Output column | Type | Description |
|---|---|---|
| `attribute_fidelity_valid` | `bool \| None` | `True` if all checked attributes pass; `None` if unavailable. |
| `attribute_fidelity_invalid_entities` | `list` | Each failing entity with attributes checked and reasoning. |

### Relational consistency

> "Do the synthetic entities preserve the same relational coherence with each other that the originals had?"

The judge inspects cross-entity relationships within a record — for example, whether a synthetic city is actually located in the synthetic state, or whether a synthetic date of birth is consistent with a synthetic age. Records with no checkable relationships always pass.

Relationships inspected include geographic pairings (city ↔ state, city ↔ postcode), temporal coherence (date of birth ↔ age), and name–email alignment.

| Output column | Type | Description |
|---|---|---|
| `relational_consistency_valid` | `bool \| None` | `True` if all relations pass; `None` if unavailable. |
| `relational_consistency_invalid_relations` | `list` | Each failing relation with participants and reasoning. |

---

## Reading replace evaluation results

`display_record()` renders a formatted per-record view that includes all four judge verdicts alongside the replacement map:

```python
evaluated.display_record(0)
```

For a tabular overview across all records:

```python
evaluated.dataframe[
    [
        "detection_valid",
        "type_fidelity_valid",
        "attribute_fidelity_valid",
        "relational_consistency_valid",
    ]
]
```

Use `trace_dataframe` for the full internal trace including raw judge outputs.

---

## Model roles

All four judges default to `gpt-oss-120b`. Defaults are defined in [`evaluate.yaml`](https://github.com/NVIDIA-NeMo/Anonymizer/blob/main/src/anonymizer/config/default_model_configs/evaluate.yaml). Override them by passing a `model_configs` YAML to `Anonymizer(model_configs=...)` — see [Models](models.md) for the full override pattern.

| Role | Default | Purpose |
|------|---------|---------|
| `detection_validity_judge` | `gpt-oss-120b` | Checks detected (value, label) pairs for correctness. |
| `replace_type_fidelity_judge` | `gpt-oss-120b` | Checks entity class and format preservation. |
| `replace_attribute_fidelity_judge` | `gpt-oss-120b` | Checks within-entity attribute preservation. |
| `replace_relational_consistency_judge` | `gpt-oss-120b` | Checks cross-entity coherence within a record. |

```yaml
# my_models.yaml
selected_models:
  evaluate:
    detection_validity_judge: your-model-alias
    replace_type_fidelity_judge: your-model-alias
    replace_attribute_fidelity_judge: your-model-alias
    replace_relational_consistency_judge: your-model-alias
```

