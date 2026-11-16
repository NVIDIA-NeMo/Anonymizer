<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Evaluation

Anonymizer provides LLM-as-judge evaluation for both modes, replace and rewrite, but they work differently:

| Mode | How evaluation runs |
|------|---------------------|
| **Replace** | Post-hoc, via a separate `Anonymizer.evaluate()` call after `run()` / `preview()`. |
| **Rewrite** | Automatic leakage/utility scoring runs as part of every `run()` / `preview()` call. A separate `Anonymizer.evaluate()` call adds LLM-as-judge quality scoring. |

---

## Combined evaluation report

Both result types support a **NeMo Anonymizer Evaluation Report** from saved results:

```python
evaluated.display_evaluation_report()
evaluated.export_evaluation_report("evaluation_report.html")
```

The strategy and number of unique failed records appear below the title. Failure
counts retain saved pipeline and evaluation failures; repeated failures for the
same record count once. When failures are present, a Failed Record Summary groups
unique records by stage and exact saved reason. Expand Failed record details to
see record IDs, stages, and full reasons. Duplicate entries are omitted from the
display; a record may appear in several groups, so group counts can exceed the
headline total. Generic saved errors are preserved rather than replaced with an
inferred cause; consult the run log for more specific diagnostics.
Unavailable judge results are also shown separately in
the summaries. The strategy is: Rewrite or Replace - Redact, Substitute,
Annotate, or Hash. Older results without strategy metadata display Unknown.
A compact **Models Used** section follows the strategy/failure information.
Anonymization and evaluation tasks appear in two columns, with model names and
providers (aliases are available on hover). Validator pools list all configured
models. Only applicable roles are included; local Redact/Annotate/Hash replacements
have no replacement-model entry. These are saved task assignments, not request
usage logs: conditional tasks such as repair or custom-label classification may
not run for every record. No API keys, endpoints, or inference parameters are saved.

Run metadata is snapshotted on `run()` / `preview()` as `model_metadata`.
`evaluate()` preserves those assignments and snapshots its own requested judge
roles. Repeated evaluation replaces the previous judge assignments, rather than
adding duplicates. Older results without snapshots show **Model information
unavailable**; the report never guesses from current configuration.

**Labels Used** follows Models Used. It says Default label list when
`Detect.entity_labels` was omitted; an explicitly supplied custom list is shown
in full. Any excluded labels are listed separately. This describes the configured
detection scope, rather than the labels actually found in the data.

**Batching Settings** follows Labels Used and displays saved
`privacy_qa_batch_size` and `validation_max_entities_per_call`. Privacy QA batch
size comes from Rewrite's run settings or Replace's evaluation settings; explicit
None means all questions in one call per record. Validation batching is preserved
from the detection run. Missing saved values display Unavailable; privacy QA
that has not run displays Not evaluated. The result's `execution_settings`
dictionary holds these snapshots.

Two evaluation summaries appear side by side on wide screens and stack on narrow
screens and when printing:

- **Privacy & Quality Judge Summary:** record-level pass counts for saved judges,
  pooled entity coverage from candidate counts, unavailable counts, and separate
  high/medium/low distributions for Rewrite's privacy, quality, and style rubrics.
- **Privacy Inference Summary:** the existing inference metrics, plus saved
  Rewrite risk tolerance, repair threshold, repair limit, and iterations taken.

Full-width **Privacy & Quality Judge Issues** and **Privacy Inference Issues**
sections follow. Only problems are listed: missed entities, failed judge checks,
medium/low Rewrite ratings, inference leaks, and unavailable evaluations. Medium
ratings are shown as ratings, not converted to boolean failures. Record indexes
are zero-based and shared across both sections. Original values, replacements,
relations, and saved reasons appear where available. Entity-level issues also
show Source, joined by exact original value and label from saved span annotations:
`detector` (GLiNER in the standard pipeline), `augment`, or `latent`. Multiple
sources are listed when occurrences have different origins. Unknown means no
saved provenance; tagged means an explicit Rewrite entity with no finer source
available. Coverage-judge misses show Not detected when no matching source is
available. Record-level and relation-level issues have no single entity source.

Judges without saved result columns are omitted. Inference that has not run says
**Not evaluated**; missing results from attempted evaluations remain unavailable.
No model calls are made. The HTML export accepts a string or Path, returns a Path,
overwrites existing files, and requires an existing parent directory. It includes
original entity values and can be opened in a browser to print to PDF.
The existing `display_privacy_report()` and `export_privacy_report()` remain
available for inference-only reports.

## Replace Evaluation

### Inspect entity-level privacy findings

Both `AnonymizerResult` and `PreviewResult` expose saved privacy assessments as a
dataframe through `privacy_findings`. After Replace privacy evaluation, or after
Rewrite's automatic evaluation:

```python
evaluated.display_privacy_report()

# Save a standalone HTML file to share or open in a browser and print to PDF.
evaluated.export_privacy_report("report.html")

# Access the same summary and problems programmatically.
summary = evaluated.privacy_summary
findings = evaluated.privacy_findings
```

Rewrite reports also show the saved risk tolerance and its repair threshold in
leakage-mass units, the maximum repair iterations allowed, and the maximum repair
iterations taken across records (initial rewrite/evaluation excluded). Missing
saved iteration counts display as unavailable. These settings are retained by new runs and previews through
subsequent evaluation; older results without saved settings omit these rows.

The report starts with evaluation coverage, records with reported leakage, recoverable
entity counts, high-sensitivity leaks, leakage mass, and the weighted entity recovery
rate. The overall weighted rate divides total leakage mass by total possible sensitivity
weight across complete evaluated records; it does not average record rates.
Unavailable records and Rewrite entities exempted from QA are excluded from rate
denominators. Records not evaluated are counted separately.

The detail table shows **problems only**: reported leaks (`yes`) and unavailable
evaluations (`Unavailable`). It includes record index/ID, entity value/label,
category (shown as Type: direct/quasi/latent), sensitivity (medium shown as med), `Can entity still be inferred?`, confidence, reason, evidence, and
leakage contribution. Negative answers and unevaluated entities are omitted.
Sensitivity rationale, questions, status, and errors remain in
`trace_dataframe`. Indexes are zero-based, matching `display_record()`.
Unavailable contributions are null, never zero. An unavailable record with no
recoverable targets has a row with null entity fields.

`export_privacy_report()` accepts a string or `Path`, writes UTF-8 HTML with
embedded styles, and returns the output `Path`. It overwrites an existing file;
the parent directory must already exist. No notebook dependency or model calls
are needed. The export includes original entity values. For Google Docs, copy
the rendered report from a browser and paste it into a document.

The HTML report wraps full reasons and evidence without truncation. These accessors
read saved final assessments without model calls and preserve the judge's findings;
they do not independently verify their correctness. Editing findings cannot mutate
the saved trace.

### Optional privacy leakage evaluation

Enable recovery checks for the original detected entities with:

```python
from anonymizer import EvaluateConfig

evaluated = anonymizer.evaluate(
    result,
    config=EvaluateConfig(compute_privacy_leakage=True),
)
```

To answer smaller batches of questions per record:

```python
evaluated = anonymizer.evaluate(
    result,
    config=EvaluateConfig(compute_privacy_leakage=True, privacy_qa_batch_size=5),
)
```

`privacy_qa_batch_size` defaults to `5` (up to five questions per call). Set it
to `None` for one call per record. Each batch
receives the full replaced text and preserves the original question IDs. Batches
run sequentially within a record, with independent correction retries, and their
answers are merged back in question order. Classification and sensitivity logic
are unchanged. If any batch fails, the record's privacy answers and leakage score
remain unavailable; partial answers are not scored. Smaller batches increase
request count and repeat the text in each request. The option does not affect Rewrite.

This option is off by default and works with Substitute, Redact, Hash, and Annotate.
It assigns category and sensitivity to each detected
value/label pair, generates privacy questions, and answers them using the replaced
text. All detected entities are evaluated, including low-sensitivity entities.
Sensitivity assessment uses the original text and optional dataset summary;
Replace does not run domain classification or domain supplements.
It shares Rewrite's sensitivity policy: generic quasi-identifiers default to low
unless distinguishing context makes them useful for linkage, while specific
quasi-identifiers default to medium unless they add no meaningful linkage or
recognition risk in the document. A planned edit does not justify a low rating.
Standard labels use the same fixed direct/quasi category mapping as Rewrite;
direct identifiers must be high sensitivity. Like Rewrite, Replace collects the
union of unmapped labels across the dataset and classifies them in one upfront
call, based on general label meaning rather than particular values. No classification
model call runs when every label is covered by the default table. The completed
category map is shared across records and supplied to sensitivity assessment.
The sensitivity model returns only a stable target ID, sensitivity, and
reason. Original entity values, labels, and categories are joined back locally, so Unicode
spelling and label text do not have to be reproduced by the model. Missing,
duplicate, or unknown IDs trigger correction retries.
It does not discover latent entities, require a privacy goal, or rewrite or repair output.

`evaluated.dataframe` includes `leakage_mass`: the sum of sensitivity weight ×
confidence for answers indicating that an original value remains recoverable.
Weights are high = 1.0, medium = 0.6, low = 0.3. `weighted_leakage_rate` is
leakage mass divided by the sum of weights for evaluated entity targets (zero
for an empty successful evaluation). Both scores are unavailable when evaluation fails.

`evaluated.trace_dataframe` also includes:

| Column | Contents |
|---|---|
| `_entity_sensitivity` | Original values and labels, category, sensitivity, and contextual reason |
| `_entity_classification` | Shared taxonomy restricted to labels detected in each record |
| `_privacy_qa` | Questions targeting the original values |
| `_privacy_qa_reanswer` | Yes/no, confidence, reason, and evidence quotes for every question |
| `_privacy_leakage_contributions` | Per-entity contributions, including zero for negative answers |
| `_privacy_qa_error` | Diagnostic when evaluation is unavailable |

The judge assesses both direct mentions and contextual inference. A reported leak
is a model assessment, including its reasoning and confidence, rather than proof
of successful re-identification. Annotate can deliberately retain recoverable values.

Incomplete answers trigger correction retries. Failed evaluation preserves the
record with an unavailable score rather than reporting zero leakage or inventing
a leak. Records with no detected entities receive zero mass without model calls.
Repeated opt-in evaluation recomputes and replaces prior privacy fields.

The `evaluate` model-selection section provides
`privacy_sensitivity_analyzer` (default `gpt-oss-120b`), and
`privacy_qa_judge` (default `nemotron-30b-thinking`). These aliases are validated
only when Replace privacy evaluation is enabled. Rewrite already computes privacy
leakage; this option does not rerun its privacy evaluation.
The optional `privacy_entity_classifier` override selects a separate model for
the upfront custom-label call; when omitted, it reuses `privacy_sensitivity_analyzer`.
If the shared taxonomy cannot be completed, affected privacy scores remain
unavailable rather than falling back to per-record category guesses.

### Standard Replace judges

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

The active judges depend on the anonymization strategy and `EvaluateConfig`:

| Evaluation | Applies to |
|------------|------------|
| Entity Coverage | All anonymization strategies |
| Detection Validity | Optional; enable with EvaluateConfig(compute_detection_validity=True) |
| Type Fidelity, Attribute Fidelity, Relational Consistency | Substitute mode only |

---

### Entity Coverage

> "Which sensitive values in the original text did the anonymizer fail to detect?"

Entity coverage is the primary **judge-anchored recall** metric and **always runs**. The judge scans the original text independently to extract all in-scope PII candidates, without knowledge of what the anonymizer detected. After scope and literal-span filtering, candidates are deduplicated by normalized value. A deterministic postprocessing step then classifies each unique candidate value as covered (also detected by the anonymizer) or missed (not detected). The score is `n_covered / (n_covered + n_missed)` — recall over the judge's unique candidate values.

Note: the judge measures detection recall, not output leakage. A value detected but replaced with a poor substitute still scores as covered. Extra anonymizer detections do not directly enter the numerator or denominator, though a detection that matches a judge candidate under the value-matching rules classifies that candidate as covered. Matching is value-based; detection validity separately evaluates whether detected value-label pairs are correct.

The judge is scoped and contextualized by the same signals used during anonymization:

- **`entity_labels`** — the detection taxonomy in scope; the judge only reports values whose type falls within it.
- **`excluded_entity_labels`** — labels explicitly excluded from detection; the judge ignores entities of these types so excluded labels are never penalised in the coverage score.
- **`data_summary`** — used purely to interpret literal values and their semantic types, never to invent entities absent from the text.

`Detect.entity_label_examples` is detection-only guidance and is not persisted on result objects or passed to evaluation judges. Non-default labels must be declared in `entity_labels`, so their explicit label scope is preserved for post-hoc evaluation even though the configured example values are not.

| Output column | Type | Description |
|---|---|---|
| `entity_coverage` | `float \| None` | `n_covered / (n_covered + n_missed)` — fraction of the judge's unique candidate values that the anonymizer detected. `1.0` means no missed entities; `None` if the judge was unavailable. |
| `missed_entities` | `list` | Each unique candidate value not detected by the anonymizer, with its `value`, judge-assigned `label`, and one-sentence `reasoning`. Empty when no entities were missed. |

**Special values:**

| Scenario | `entity_coverage` | `missed_entities` |
|---|---|---|
| Judge found no PII candidates in the text | `1.0` — No candidates found by the judge | `[]` |
| Judge found candidates and anonymizer caught all | `1.0` | `[]` |
| Judge found candidates and anonymizer missed some | 0–1 fraction | populated |
| Judge call failed or returned a malformed response | `None` | `[]` |

---

### Entity Detection Judge

#### Detection Validity

> "Are the detected entities actually correct (value, label) pairs in context?"

Detection validity is disabled by default. To enable it, set `compute_detection_validity=True` in `EvaluateConfig`. This metric is intended for internal model and threshold evaluation. It examines each detected span and flags:

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

Same judge as in replace mode — see [Entity Coverage](#entity-coverage) above. It **always runs** and emits `entity_coverage` and `missed_entities`. In Rewrite mode, `evaluate()` also sets `needs_human_review=True` on any row whose `missed_entities` is non-empty.

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

### Batching Rewrite privacy answers

Rewrite answers privacy questions during `run()` / `preview()` and after each repair.
Configure batching on `Rewrite`, rather than `EvaluateConfig`:

```python
rewrite=Rewrite(privacy_qa_batch_size=5)
```

The default batch size is `5`. Set `None` to answer all privacy questions for
a record in one call. Each
batch receives the full current rewritten text and preserves original question IDs.
Answers are merged before leakage scoring and repair decisions. This also applies
to the combined graph. A failed batch follows Rewrite's existing evaluation failure
handling; partial answers are not scored. Quality QA is unchanged.
