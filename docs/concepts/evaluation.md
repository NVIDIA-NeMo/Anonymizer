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

Which judges run depends on the strategy and your `EvaluateConfig`:

| Judge | When it runs |
|-------|--------------|
| **Entity Coverage** | Always. |
| **Detection Validity** | Only when `EvaluateConfig(compute_detection_validity=True)` — off by default. |
| **Type Fidelity, Attribute Fidelity, Relational Consistency** | Substitute mode only. |

---

### Entity Coverage

> "Which sensitive values from the original text survived into the anonymized output?"

Entity coverage is the primary residual-leakage metric and **always runs**. The judge scans the original text independently and reports any in-scope sensitive values that were **not** removed or replaced. A deterministic postprocessing step then drops candidates that are already accounted for by the Anonymizer's final entities (exact, sub-span, or composite matches), so only genuine leaks remain.

The judge is scoped and contextualized by the same signals used during anonymization:

- **`entity_labels`** — the detection taxonomy in scope; the judge only reports values whose type falls within it.
- **`data_summary`** — used purely to interpret literal values and their semantic types, never to invent entities absent from the text.
- **`strict_entity_protection`** — (rewrite only) when enabled, the judge also reports inferable/indirect sensitive values, not just literal identifiers.

| Output column | Type | Description |
|---|---|---|
| `entity_coverage` | `float \| None` | `n_final / (n_final + n_leaked)` — fraction of sensitive values that were removed. `1.0` means nothing leaked; `None` if the judge was unavailable. |
| `leaked_entities` | `list` | Each surviving value with its `value`, `label`, and one-sentence `reasoning`. Empty when nothing leaked. |

**Special values:**

| Scenario | `entity_coverage` | `leaked_entities` |
|---|---|---|
| No sensitive values in the original text | `1.0` | `[]` |
| Judge ran and found no surviving values | `1.0` | `[]` |
| Judge ran and found surviving values | 0–1 fraction | populated |
| Judge call failed or returned a malformed response | `None` | `[]` |

---

### Entity Detection Judge

#### Detection Validity

> "Are the detected entities actually correct (value, label) pairs in context?"

Detection validity is **opt-in** — it runs only when you pass `Anonymizer.evaluate(..., config=EvaluateConfig(compute_detection_validity=True))`. It is disabled by default and is intended for internal model/threshold experiments rather than as a customer-facing metric. When enabled, it looks at each detected span and flags:

- **false_positive** — the span is not actually identifying or sensitive in this context (common word, generic phrase, boilerplate).
- **wrong_label** — the span is sensitive but the label sits in a clearly different domain (e.g. a company name labeled `first_name`). Sibling labels within the same broad domain are treated as valid.
- **not_in_text** — the literal value does not appear in the original text.
- **wrong_boundary** — the span is a clear partial or over-extended capture (omits part of the actual value, or absorbs surrounding function words). Descriptive words in natural prose around a bare entity value are not a boundary error.
- **contextual_mismatch** — the span refers to something other than the labeled entity type in this context (e.g. "Apple" as fruit labeled `company_name`).

| Output column | Type | Description |
|---|---|---|
| `detection_valid` | `bool \| None` | `True` if all detections pass; `None` if the judge was unavailable. |
| `detection_invalid_entities` | `list` | Each flagged detection with value, label, and one-sentence reasoning. |

**Special values:**

| Scenario | `detection_valid` | Display | Log |
|---|---|---|---|
| No entities detected in this record | `True` | Satisfied | `INFO`: "N passthrough row(s) have no detected entities — detection_valid set to True (trivially valid)" |
| Judge ran and all detections passed | `True` | Satisfied | — |
| Judge ran and flagged one or more detections | `False` | Not Satisfied / Partially Satisfied | — |
| Judge call failed or returned a malformed response | `None` | Unavailable | — |

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

### Reading replace evaluation results

`display_record()` renders a formatted per-record view that includes all four judge verdicts alongside the replacement map:

```python
evaluated.display_record(0)
```

For a tabular overview across all records:

```python
evaluated.dataframe[
    [
        "entity_coverage",
        "type_fidelity_valid",
        "attribute_fidelity_valid",
        "relational_consistency_valid",
        # "detection_valid" — present only if compute_detection_validity=True
    ]
]
```

Use `trace_dataframe` for the full internal trace including raw judge outputs.

---

### Model roles

The entity coverage judge defaults to `nemotron-super`; the other replace-evaluation judges default to `gpt-oss-120b`. Defaults are defined in [`evaluate.yaml`](https://github.com/NVIDIA-NeMo/Anonymizer/blob/main/src/anonymizer/config/default_model_configs/evaluate.yaml). Override them by passing a `model_configs` YAML to `Anonymizer(model_configs=...)` — see [Models](models.md) for the full override pattern.

The roles are `entity_coverage_judge`, `detection_validity_judge`, `replace_type_fidelity_judge`, `replace_attribute_fidelity_judge`, and `replace_relational_consistency_judge`.

```yaml
# my_models.yaml
selected_models:
  evaluate:
    entity_coverage_judge: your-model-alias
    detection_validity_judge: your-model-alias
    replace_type_fidelity_judge: your-model-alias
    replace_attribute_fidelity_judge: your-model-alias
    replace_relational_consistency_judge: your-model-alias
```

---

## Rewrite Evaluation

Rewrite evaluation has two layers:

1. **Automatic (always runs)** — leakage mass, utility score, weighted leakage rate, and `needs_human_review` are computed as part of every `run()` / `preview()` call. See [Rewrite](rewrite.md) for the repair loop and output columns.

2. **Post-hoc LLM judges (optional)** — call `Anonymizer.evaluate()` on a completed rewrite result to add the entity coverage judge (always), three holistic quality rubrics (always), and the detection validity judge (only with `EvaluateConfig(compute_detection_validity=True)`).

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

### Entity Coverage

Same judge as in replace mode — see [Entity Coverage](#entity-coverage) above. It **always runs** and emits `entity_coverage` and `leaked_entities`. In rewrite mode the judge additionally honours `strict_entity_protection`: when the rewrite config enables it, inferable/indirect sensitive values are reported as leaks, not just literal identifiers.

---

### Entity Detection Judge

Same judge as in replace mode — see [Entity Detection Judge](#entity-detection-judge) above, and like there it is **opt-in** via `EvaluateConfig(compute_detection_validity=True)` (off by default). In rewrite mode, `detection_valid` is returned as a **0–1 fraction** (the share of detected entities that passed), rather than a boolean. A value of `1.0` means all detections are valid; lower values mean more entities were flagged — the value itself is the fraction that passed.

| Output column | Type | Description |
|---|---|---|
| `detection_valid` | `float \| None` | 1.0 if all detections pass; fraction of valid entities otherwise; `None` if the score is unavailable. |
| `detection_invalid_entities` | `list` | Each flagged detection with value, label, and one-sentence reasoning. |

**Special values:**

| Scenario | `detection_valid` | Display | Log |
|---|---|---|---|
| No entities detected in this record | `1.0` | 1.00 | `INFO`: "N passthrough row(s) have no detected entities — detection_valid set to 1.0 (trivially valid)" |
| Judge ran and all detections passed | `1.0` | 1.00 | — |
| Judge ran and flagged one or more detections | 0–1 fraction | numeric score | — |
| Judge call failed or entity data unreadable | `None` | Unavailable | `WARNING`: "Could not parse entities_by_value to compute detection_valid fraction" |

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

### Reading rewrite evaluation results

`display_record()` renders a formatted per-record view that includes entity coverage, all three judge rubrics, and (when enabled) the detection validity fraction alongside the rewritten text:

```python
evaluated.display_record(0)
```

For a tabular overview across all records:

```python
evaluated.dataframe[["entity_coverage", "judge_evaluation"]]
# add "detection_valid" if you evaluated with compute_detection_validity=True
```

Use `trace_dataframe` for the full internal trace including raw judge outputs.

---

### Model roles

The rewrite quality judge defaults to `nemotron-30b-thinking` and the entity coverage judge to `nemotron-super`. The detection validity judge shares the `detection_validity_judge` role used by replace evaluation. Defaults are defined in [`evaluate.yaml`](https://github.com/NVIDIA-NeMo/Anonymizer/blob/main/src/anonymizer/config/default_model_configs/evaluate.yaml). Override them via `model_configs`:

```yaml
# my_models.yaml
selected_models:
  evaluate:
    entity_coverage_judge: your-model-alias
    detection_validity_judge: your-model-alias
    rewrite_judge: your-model-alias
```
