<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Evaluation

Anonymizer provides LLM-as-judge evaluation for both modes, replace and rewrite, but they work differently:

| Mode | How evaluation runs |
|------|---------------------|
| **Replace** | Post-hoc, via a separate `Anonymizer.evaluate()` call after `run()` / `preview()`. |
| **Rewrite** | Automatic leakage/utility scoring runs as part of every `run()` / `preview()` call. A separate `Anonymizer.evaluate()` call adds LLM-as-judge quality scoring. |

---

## Replace Evaluation

Replace evaluation is **optional and post-hoc** — you call `Anonymizer.evaluate()` on a result from `run()` or `preview()`:

```python
from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Substitute

anonymizer = Anonymizer()
cfg = AnonymizerConfig(replace=Substitute())
src = AnonymizerInput(source="data.csv", text_column="text")

result = anonymizer.run(config=cfg, data=src)
evaluated = anonymizer.evaluate(result)
evaluated.display_record(0)
```

Both `run()` and `preview()` results can be saved and evaluated in a separate session:

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

Four LLM judges run per record: one that scores detection quality and three that score replacement quality (Substitute mode only).

---

### Entity Detection Judge

#### Detection Validity

> "Are the detected entities actually correct (value, label) pairs in context?"

This judge runs regardless of which replace mode was used. It looks at each detected span and flags:

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

### Entity Replacement Judges

When the source result used the **Substitute** mode, three additional LLM judges run in parallel — one per quality dimension.

#### Type Fidelity

> "Does each synthetic value still belong to the same entity class and match the expected format for that class?"

The judge checks that replacements are shape-compatible with their originals — same granularity and character class — anchored by what the original itself looks like. It does **not** check semantic attributes (gender, age bucket) or cross-entity consistency; those are separate metrics.

| Output column | Type | Description |
|---|---|---|
| `type_fidelity_valid` | `bool \| None` | `True` if all replacements pass; `None` if the judge was unavailable. |
| `type_fidelity_invalid_replacements` | `list` | Each failing replacement with label, original, synthetic, and reasoning. |

#### Attribute Fidelity

> "Does each synthetic value preserve the salient within-entity attributes of the original?"

The judge checks two attributes:

- **Gender of name** — applies to `first_name`, `last_name`, `user_name`. Only checked when the original name clearly implies a gender. Adjacent or ambiguous cases pass.
- **Age bucket** — applies to `age` and `date_of_birth`. Buckets: child (0–12), teen (13–19), young adult (20–29), adult (30–44), middle-aged (45–64), senior (65+). Adjacent buckets pass; only clear flips (adult → child) fail.

All other labels are outside the scope of this metric.

| Output column | Type | Description |
|---|---|---|
| `attribute_fidelity_valid` | `bool \| None` | `True` if all checked attributes pass; `None` if unavailable. |
| `attribute_fidelity_invalid_entities` | `list` | Each failing entity with attributes checked and reasoning. |

#### Relational Consistency

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

The four roles are `detection_validity_judge`, `replace_type_fidelity_judge`, `replace_attribute_fidelity_judge`, and `replace_relational_consistency_judge`.

```yaml
# my_models.yaml
selected_models:
  evaluate:
    detection_validity_judge: your-model-alias
    replace_type_fidelity_judge: your-model-alias
    replace_attribute_fidelity_judge: your-model-alias
    replace_relational_consistency_judge: your-model-alias
```

---

## Rewrite Evaluation

Rewrite evaluation has two layers:

1. **Automatic (always runs)** — leakage mass, utility score, weighted leakage rate, and `needs_human_review` are computed as part of every `run()` / `preview()` call. See [Rewrite](rewrite.md) for the repair loop and output columns.

2. **Post-hoc LLM judges (optional)** — call `Anonymizer.evaluate()` on a completed rewrite result to add the entity detection judge and three holistic quality rubrics.

```python
from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Rewrite

anonymizer = Anonymizer()
cfg = AnonymizerConfig(rewrite=Rewrite())
src = AnonymizerInput(source="data.csv", text_column="text")

result = anonymizer.run(config=cfg, data=src)
evaluated = anonymizer.evaluate(result)
evaluated.display_record(0)
```

Both `run()` and `preview()` results can be saved and evaluated in a separate session:

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

---

### Entity Detection Judge

Same judge as in replace mode — see [Entity Detection Judge](#entity-detection-judge) above. In rewrite mode, `detection_valid` is returned as a **0–1 fraction** (the share of detected entities that passed), rather than a boolean. A value of `1.0` means all detections are valid; lower values indicate the fraction of entities the judge flagged as incorrect.

| Output column | Type | Description |
|---|---|---|
| `detection_valid` | `float \| None` | 1.0 if all detections pass; fraction of valid entities otherwise; `None` if the judge was unavailable. |
| `detection_invalid_entities` | `list` | Each flagged detection with value, label, and one-sentence reasoning. |

---

### Rewrite Quality Judges

Three rubrics evaluate the holistic quality of the rewritten text. All three run as a single LLM judge call and are stored together under `judge_evaluation`.

#### Privacy

> "Does the rewritten text adequately remove linkage risk to the original record?"

Scores residual linkage risk after the rewrite — comparing rewritten values to originals, distinguishing direct identifiers from quasi-identifiers, and assessing whether remaining details narrow the candidate set of plausible matches.

| Score | Meaning |
|-------|---------|
| `high` | Original direct identifiers removed; remaining quasi-identifiers create low linkage risk. |
| `medium` | No obvious direct identifiers remain, but a distinctive quasi-identifier bundle creates noticeable linkage risk. |
| `low` | Easily or near-certainly linkable — direct identifiers remain or enough detail survives that re-identification requires minimal effort. |

#### Quality

> "How well does the rewritten text preserve important meaning, facts, and structure?"

Evaluates content preservation independent of privacy and style. Changes made for privacy reasons are not penalized when the core meaning is intact.

| Score | Meaning |
|-------|---------|
| `high` | Important meaning, facts, and structure fully preserved. |
| `medium` | Most content preserved; minor details lost or slightly distorted. |
| `low` | Material loss of important information, contradictions, or distorted core meaning. |

#### Style

> "Does the rewritten text read as fluent, coherent, and human-written prose?"

Evaluates readability, grammatical correctness, clarity, and phrasing — independent of content changes.

| Score | Meaning |
|-------|---------|
| `high` | Fluent, coherent, human-written prose. |
| `medium` | Mostly readable; isolated awkward phrasing or stiff transitions. |
| `low` | Noticeably unnatural; broken grammar, placeholder-like language, or machine-generated feel. |

The three rubric scores are stored together under the `judge_evaluation` column as a dict:

```python
# Example judge_evaluation value for a single record
{
    "privacy": {"score": "high",   "reasoning": "All direct identifiers removed..."},
    "quality": {"score": "medium", "reasoning": "Key facts preserved but some details lost..."},
    "style":   {"score": "high",   "reasoning": "Reads naturally throughout..."},
}
```

---

## Reading rewrite evaluation results

`display_record()` renders a formatted per-record view that includes the detection validity fraction and all three judge rubrics alongside the rewritten text:

```python
evaluated.display_record(0)
```

For a tabular overview across all records:

```python
evaluated.dataframe[["detection_valid", "judge_evaluation"]]
```

Use `trace_dataframe` for the full internal trace including raw judge outputs.

---

## Model roles (rewrite evaluation)

The rewrite quality judge defaults to `nemotron-30b-thinking`. The detection validity judge shares the `detection_validity_judge` role used by replace evaluation. Defaults are defined in [`evaluate.yaml`](https://github.com/NVIDIA-NeMo/Anonymizer/blob/main/src/anonymizer/config/default_model_configs/evaluate.yaml). Override them via `model_configs`:

```yaml
# my_models.yaml
selected_models:
  evaluate:
    detection_validity_judge: your-model-alias
    rewrite_judge: your-model-alias
```
