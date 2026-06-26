# Entity Coverage — Implementation Plan

## Problem

Currently `detection_valid` scores **precision** of the detected entities. Observed scores are predominantly in the lower ranges issue [#176](https://github.com/NVIDIA-NeMo/Anonymizer/issues/176), partly because the judge flags over-detection and mislabeling among other issues. But the actual privacy-sensitive question users care about is the opposite: **did any real PII slip through the anonymizer?**

Precision complaints are already surfaced in the entity dropdown for manual inspection, making `detection_valid` feel redundant as a headline metric. What's missing is a **recall** signal: a score that answers "of all the PII that existed, how much did we successfully catch?"

---

## Decision: Precision vs Recall

`detection_valid` measures **precision** — of the entities we detected, were they correct? `entity_coverage` measures **recall** — of all the PII that existed, how much did we catch?

For an anonymizer, recall is the user-facing priority: a missed PII entity is a privacy failure, while an over-detection is a minor inconvenience already visible in the entity dropdown. We therefore:

- **Keep `detection_valid`** as an internal, opt-in feature (`compute_detection_validity=False` by default) — useful for model/threshold experiments during development, not surfaced to end users.
- **Add `entity_coverage`** as the new customer-facing metric — the subject of this plan.

---

## Proposed Solution: Entity Coverage

Replace `detection_valid` as the headline evaluate metric with `entity_coverage`:

```
entity_coverage = n_entities_detected / (n_entities_detected + n_entities_leaked)
```

- **`n_entities_detected`** — identifiers detected by Anonymizer in the original text (via the detection pipeline)
- **`n_entities_leaked`** — identifiers missed by Anonymizer in the original text, found by the evaluation LLM
- **`leaked_entities`** — the list of missed entities (shown in a dropdown when coverage < 100%)

The evaluation LLM runs on the **original text** and detects all PII that should have been caught. The delta between what it finds and what Anonymizer already detected gives the leaked entities.

**Key shift:** current judge asks "were our detections correct?" (precision). New judge asks "did any PII survive?" (recall).

**Question TBD:** The evaluation and anonymization currently use the same LLM, and we may need to address potential bias by using a different model for evaluation. Potential models: Nemotron Ultra and Nemotron Super (previous [research log](https://nvidia.atlassian.net/wiki/spaces/SDGR/pages/3597435385/2026-06-09+Entity+Validator+Augmentor+Model+Selection+for+NeMo+Anonymizer) around experimenting these 2 models)

---

## Design Decisions

### Coverage display differs by mode

| Mode | Display format | Example |
|---|---|---|
| **Replace** | `Entity Coverage: <Satisfied\|Not Satisfied> — n/d (pct%)` | `Entity Coverage: Not Satisfied — 18/21 (86%)` |
| **Rewrite** | `Entity Coverage: <float> — n/d (pct%)` | `Entity Coverage: 0.86 — 18/21 (86%)` |

Where `n` = `n_entities_detected`, `d` = `n_entities_detected + n_entities_leaked`, and `pct` is derived from the fraction. The fraction gives absolute scale; the percentage makes it immediately readable without mental math. No explanatory subtitle line is shown — entity coverage is self-explanatory.

A value of `1.0` (or Satisfied) means no PII leaked into the output. Any value below `1.0` (or Not Satisfied) means at least one entity was missed, and `leaked_entities` names them.

### Passthrough rows

Rows where Anonymizer detected **zero entities** trivially score `entity_coverage = 1.0` — there was nothing to protect, so nothing could have leaked. The LLM judge is skipped for these rows (same pattern as the current passthrough shortcut in `_BaseJudgeWorkflow`).

### Prompt starting point

The entity judge prompt is based on the [leakage eval script](https://gitlab-master.nvidia.com/sdg-research/anonymizer-experiments/-/blob/lramaswamy/latency-benchmark/experiments/latency-benchmark/scripts/leakage_eval.py?ref_type=heads#L26) from the experiments repo. It will be refined as it is tested against real examples. The core task: given the original text, identify all PII — then report any that Anonymizer missed.

### Reuse `_BaseJudgeWorkflow`

The new `EntityCoverageWorkflow` extends `_BaseJudgeWorkflow`. The partition-LLM-merge skeleton is identical; only the schema, prompt, and post-process step differ (coverage fraction computed from counts, not a boolean verdict field). `postprocess()` is overridden in the subclass — the base class stays unchanged.

---

## Scope

- Adds `entity_coverage` and `leaked_entities` as the new default customer-facing columns in both replace and rewrite `evaluate()` output.
- Retains `detection_valid` and `detection_invalid_entities` as an opt-in internal feature behind `EvaluateConfig(compute_detection_validity=False)` — not customer-facing, off by default.
- New public API: `EvaluateConfig.compute_detection_validity: bool = False`; new column name constants (`COL_ENTITY_COVERAGE`, `COL_LEAKED_ENTITIES`); new model role `entity_coverage_judge` in `EvaluateModelSelection`.
- `COL_DETECTION_VALID` / `COL_DETECTION_INVALID_ENTITIES` constants and `detection_validity_judge` model role are **kept** — they back the opt-in internal feature.
- `DetectionJudgeWorkflow` is **not removed** — it runs only when `compute_detection_validity=True`.

---

## New Column Names

| Constant | Value | Type | Description |
|---|---|---|---|
| `COL_ENTITY_COVERAGE` | `"entity_coverage"` | `float \| None` | Fraction of entities successfully anonymized; `None` if judge unavailable |
| `COL_LEAKED_ENTITIES` | `"leaked_entities"` | `list` | Each missed entity with `value`, `label`, `reasoning` |
| `COL_ENTITY_COVERAGE_JUDGE` | `"_entity_coverage_judge"` | internal | Raw judge output; internal only |

---

## Files Changed

| File | Change |
|---|---|
| `src/anonymizer/engine/constants.py` | Keep `COL_DETECTION_JUDGE`, `COL_DETECTION_VALID`, `COL_DETECTION_INVALID_ENTITIES`; add `COL_ENTITY_COVERAGE_JUDGE`, `COL_ENTITY_COVERAGE`, `COL_LEAKED_ENTITIES` |
| `src/anonymizer/engine/evaluation/entity_coverage_judge.py` | **New file** — `EntityCoverageSchema`, `EntityCoverageWorkflow`, coverage fraction `postprocess()` |
| `src/anonymizer/engine/evaluation/detection_judge.py` | No removal — retained as-is; gated behind `compute_detection_validity` flag in call sites |
| `src/anonymizer/engine/evaluation/judge_base.py` | No changes needed — `postprocess()` is overridden in the subclass |
| `src/anonymizer/engine/rewrite/rewrite_workflow.py` | Swap `DetectionJudgeWorkflow` → `EntityCoverageWorkflow` in `evaluate()`; remove `_detection_valid_fraction()`; update column references |
| `src/anonymizer/engine/replace/replace_runner.py` | Swap `DetectionJudgeWorkflow` → `EntityCoverageWorkflow`; update column references |
| `src/anonymizer/interface/anonymizer.py` | Update `_build_user_dataframe` allowed columns for both modes; update model role references |
| `src/anonymizer/interface/display.py` | Render `entity_coverage` as Satisfied/Not Satisfied (replace) or fraction (rewrite); update leaked_entities dropdown |
| `src/anonymizer/config/anonymizer_config.py` | Add `compute_detection_validity: bool = False` to `EvaluateConfig` |
| `src/anonymizer/config/models.py` | Add `entity_coverage_judge` to `EvaluateModelSelection`; keep `detection_validity_judge` |
| `src/anonymizer/config/default_model_configs/evaluate.yaml` | Add `entity_coverage_judge` key alongside existing `detection_validity_judge` |
| `docs/concepts/evaluation.md` | Replace Entity Detection Judge section with Entity Coverage section in both replace and rewrite docs |
| `skills/anonymizer/SKILL.md` | Update evaluation tips and output template column references |
| `tests/engine/evaluation/test_detection_judge.py` | Keep as-is — covers the retained internal feature |
| `tests/engine/evaluation/test_entity_coverage_judge.py` | **New file** — new schema, prompt, and workflow tests |
| `tests/engine/test_rewrite_workflow.py` | Update column name references; remove `_detection_valid_fraction` tests |
| `tests/engine/test_evaluate.py` | Update column name assertions |
| `tests/interface/test_display.py` | Update rendering assertions for entity coverage |

---

## Step 1 — New constants (`constants.py`)

Keep the existing detection judge constants and add three new ones alongside them:

```python
# Retained (no change — back the opt-in internal feature):
COL_DETECTION_JUDGE             = "_detection_judge"
COL_DETECTION_VALID             = "detection_valid"
COL_DETECTION_INVALID_ENTITIES  = "detection_invalid_entities"

# New:
COL_ENTITY_COVERAGE_JUDGE = "_entity_coverage_judge"   # internal raw output
COL_ENTITY_COVERAGE       = "entity_coverage"          # user-facing float | None
COL_LEAKED_ENTITIES       = "leaked_entities"          # user-facing list
```

---

## Step 2 — New workflow (`entity_coverage_judge.py`)

Add a new file alongside the existing `detection_judge.py`. Key design:

### Output schema

```python
class LeakedEntity(BaseModel):
    value: str
    label: str
    reasoning: str  # one sentence: why this is PII that survived anonymization

class EntityCoverageSchema(BaseModel):
    leaked_entities: list[LeakedEntity]
    # No all_valid field — coverage is computed numerically from counts
```

### Prompt

Runs against the **original text** (`COL_TEXT`), with the Anonymizer-detected entity list injected as context. Starting from the leakage eval script prompt and refined through testing.

Core task phrasing:
- "Given the original text, identify all PII present. Report any identifiers that were not already caught by Anonymizer — these are the missed entities."
- Returns an empty list when Anonymizer caught everything.

#### Strict entity protection (`Rewrite.strict_entity_protection`)

Only applies in rewrite mode. When `strict_entity_protection=True`, the rewrite pipeline disallows `leave_as_is`, bans `combined_risk_level: low`, and overrides the MINIMUM NECESSARY CHANGE and quasi-identifier "not automatically protected" principles. The coverage judge must mirror that posture — otherwise a missed quasi-identifier or low-combined-risk entity would be excused by the judge despite the user having demanded full protection.

The prompt includes a `<strict_entity_protection>` block injected when the flag is set (empty string otherwise):

```
STRICT PROTECTION MODE IS ENABLED.

Flag ALL entities as leaked if they were not caught — including quasi-identifiers
and low-risk entities that would normally be given benefit of the doubt.
Do NOT apply MINIMUM NECESSARY CHANGE reasoning to excuse a missed entity.
Do NOT excuse a missed entity because its combined re-identification risk is low.
Any PII span not caught by Anonymizer is a miss in strict mode.
```

The block mirrors the equivalent block in `sensitivity_disposition.py` so the two prompts enforce the same standard. In replace mode the call site passes `strict_entity_protection=False` (the field does not exist on `Detect`); the block is omitted and the judge uses its default charitable posture.

#### Entity type scope (`Detect.entity_labels`)

When the user configures `Detect(entity_labels=[...])`, detection is intentionally limited to those label types. The coverage judge must be scoped to match — otherwise it would flag missed PII of types the user never asked to detect, artificially deflating the score.

The prompt includes an `<entity_type_scope>` block populated per-run in `prepare()`:

- **`entity_labels` is `None`** (default): `"Evaluate for all PII and sensitive entity types."`
- **`entity_labels` is set**: `"Detection was configured to target ONLY these entity types: <list>. Only report missed entities that belong to one of these types. Do NOT flag PII of other types as leaked — those were intentionally excluded from detection."`

The `<guidance>` section includes a corresponding rule: respect the entity_type_scope and do not penalize Anonymizer for types outside it.

Because `entity_labels` is run-level config (not per-row), the workflow constructor accepts `entity_labels: list[str] | None` and `prepare()` broadcasts it as a constant column (`_entity_type_scope`) across all rows. Call sites (`replace_runner.py`, `rewrite_workflow.py`) pass `config.detect.entity_labels`.

#### Both flags enabled simultaneously

Both can be active at the same time — there is no config constraint preventing it:

```python
AnonymizerConfig(
    detect=Detect(entity_labels=["first_name", "email_address"]),
    rewrite=Rewrite(strict_entity_protection=True),
)
```

The two blocks compose cleanly in the judge prompt: the entity type scope narrows *what* the judge looks for, and the strict protection block raises the bar *within* that scope. No contradiction — scope is applied first, strictness within that scope second. The two blocks are independent injections in `prepare()` and do not interfere with each other.

### Passthrough condition

Rows with no detected entities trivially score `entity_coverage = 1.0` — nothing to miss, LLM skipped.

### Coverage fraction (postprocess override)

```python
def _compute_coverage(self, n_entities_detected: int, n_entities_leaked: int) -> float:
    total = n_entities_detected + n_entities_leaked
    return 1.0 if total == 0 else n_entities_detected / total
```

Passthrough rows get `entity_coverage = 1.0` and `leaked_entities = []` without calling the LLM.

### Class declaration

```python
class EntityCoverageWorkflow(_BaseJudgeWorkflow):
    RAW_COL         = COL_ENTITY_COVERAGE_JUDGE
    VALID_COL       = COL_ENTITY_COVERAGE
    INVALID_COL     = COL_LEAKED_ENTITIES
    SCHEMA          = EntityCoverageSchema
    VERDICT_FIELD   = "leaked_entities"
    DEFAULT_PAYLOAD = {"leaked_entities": []}
    MODEL_ROLE      = "entity_coverage_judge"
    WORKFLOW_NAME   = "entity-coverage-judge"
```

`postprocess()` is overridden to compute the float fraction from `n_entities_detected` (from `COL_ENTITIES_BY_VALUE`) and `len(leaked_entities)` rather than storing a boolean verdict.

> **Note:** `COL_ENTITIES_BY_VALUE` (`_entities_by_value`) is derived from `COL_FINAL_ENTITIES` (`final_entities`) during detection — it groups spans by value with a labels list. The leakage eval script references `final_entities`, which is the same underlying data in a different shape. `COL_ENTITIES_BY_VALUE` is the right source here since it's already what the judge's `prepare()` step and passthrough mask consume.

> **Note on base class / VALID_COL conflict:** `_BaseJudgeWorkflow` does not write to `VALID_COL` before `postprocess()` runs — the base class `evaluate()` calls `postprocess()` as the only step that touches those columns. Overriding `postprocess()` fully in `EntityCoverageWorkflow` is safe. However, `VALID_COL` on the base class carries an implicit `bool | None` semantic (used by other judges). Since `EntityCoverageWorkflow` writes a `float | None` instead, confirm that no shared display or downstream code reads `VALID_COL` generically and assumes boolean before this is merged.

---

## Step 3 — Wire into replace and rewrite evaluate paths

### Replace (`replace_runner.py`)

Add `EntityCoverageWorkflow` — always runs in `evaluate()`. Gate `DetectionJudgeWorkflow` behind `evaluate_config.compute_detection_validity`. Both judges run on `COL_TEXT` (original text). Update column references for the new output columns.

### Rewrite (`rewrite_workflow.py`)

Same structure. Add `EntityCoverageWorkflow` as the default; gate `DetectionJudgeWorkflow` behind the flag. Update `evaluate()`:

- Remove `_detection_valid_fraction()` — no longer needed since `entity_coverage` is already a float from `postprocess()`.
- Update the try/except around the judge call to default `entity_coverage = None` and `leaked_entities = None` on failure.

---

## Step 4 — Model role update (`models.py`, `evaluate.yaml`)

```python
class EvaluateModelSelection(BaseModel):
    entity_coverage_judge: str              # new — always required for evaluate()
    detection_validity_judge: str           # retained — required only when compute_detection_validity=True
    replace_type_fidelity_judge: str
    replace_relational_consistency_judge: str
    replace_attribute_fidelity_judge: str
    rewrite_judge: str
```

Add `entity_coverage_judge` to `evaluate.yaml` alongside the existing `detection_validity_judge`. Update `model_loader.py` validation to check `entity_coverage_judge` unconditionally; `detection_validity_judge` is already present in the YAML default so no user burden is added.

---

## Step 5 — Update `_build_user_dataframe` (`anonymizer.py`)

Add the new columns to the allowed sets for both replace and rewrite mode. Keep the existing detection validity columns — they remain in the allowed set and will appear in output only when `compute_detection_validity=True` produces them:

```python
# Add to both replace and rewrite allowed sets:
COL_ENTITY_COVERAGE,            # new — always present after evaluate()
COL_LEAKED_ENTITIES,            # new — always present after evaluate()
# Already present (retained):
# COL_DETECTION_VALID           — only in output when compute_detection_validity=True
# COL_DETECTION_INVALID_ENTITIES — only in output when compute_detection_validity=True
```

---

## Step 6 — Display (`display.py`)

`display_record()` renders the coverage column as a single line. The format puts the verdict first, then the fraction for scale, then the percentage for readability:

- **Replace:** `Entity Coverage: Satisfied — 21/21 (100%)` / `Entity Coverage: Not Satisfied — 18/21 (86%)`
- **Rewrite:** `Entity Coverage: 0.86 — 18/21 (86%)` (no Satisfied/Not Satisfied badge — rewrite already uses numeric metrics)

Where the fraction is `n_entities_detected / (n_entities_detected + n_entities_leaked)` and the percentage is derived from it. The fraction gives absolute scale (18/20 and 18/200 look very different); the percentage makes it immediately readable.

Drop the explanatory subtitle line that the old Detection Judge displayed — entity coverage is self-explanatory and does not need a definition sentence. Drop the `"LLM alignment score:"` label prefix — that framing belonged to the old judge (LLM-vs-LLM alignment), not to a count ratio.

The `leaked_entities` dropdown is added with the same layout (value, label, reasoning per row). The existing `detection_invalid_entities` dropdown continues to render unchanged when `detection_valid` is present (opt-in enabled).

---

## Step 7 — Docs and skills

### `docs/concepts/evaluation.md`

Replace the **Entity Detection Judge** subsection in both Replace and Rewrite sections with an **Entity Coverage** subsection:

- Explain the recall focus (vs precision of the old judge)
- Document `entity_coverage` type per mode (float in rewrite, Satisfied/Not Satisfied in replace)
- Document `leaked_entities` list shape
- Update the special-values table (passthrough row → `1.0` / Satisfied)
- Update model role name (`entity_coverage_judge`)
- Add a brief **Internal metrics** note: `compute_detection_validity=True` enables `detection_valid` for tag-precision signal during model/threshold experiments — one paragraph, not featured

### `skills/anonymizer/SKILL.md`

- Update the evaluation tips bullet: `entity_coverage_judge` role; `entity_coverage` type per mode; `leaked_entities` dropdown
- Update output template: add `entity_coverage` column references; do not feature `detection_valid` (internal only)

---

## Step 8 — Tests

### New test file (`test_entity_coverage_judge.py`)

```
test_judge_prompt_runs_on_original_text_column
test_judge_prompt_includes_detected_entity_context
test_entity_coverage_schema_accepts_empty_leaked_list
test_entity_coverage_schema_accepts_leaked_entities
test_coverage_fraction_all_caught                     # n_entities_leaked=0 → 1.0
test_coverage_fraction_partial_miss                   # n_entities_leaked=1, n_entities_detected=3 → 0.75
test_coverage_fraction_total_miss                     # n_entities_leaked=n, n_entities_detected=0 → 0.0
test_evaluate_short_circuits_when_no_entities         # passthrough → 1.0, no LLM call
test_evaluate_invokes_adapter_for_rows_with_entities
test_evaluate_merges_entity_and_empty_rows_in_order
test_evaluate_marks_coverage_unavailable_for_malformed_payload
```

### Update existing tests

```
tests/engine/test_rewrite_workflow.py         # rename column refs; remove _detection_valid_fraction tests
tests/engine/test_evaluate.py                 # column name assertions
tests/interface/test_display.py               # entity_coverage rendering
tests/interface/test_anonymizer_interface.py  # allowed column set for both modes
```

All tests use stub adapters — no real LLM calls.

---

## Implementation Order

1. Add new constants to `constants.py`; keep existing `COL_DETECTION_*` constants
2. Add `compute_detection_validity: bool = False` to `EvaluateConfig` (`anonymizer_config.py`)
3. Write `entity_coverage_judge.py` — schema, prompt (v1 from leakage_eval script), workflow class, `postprocess()` override
4. Wire `EntityCoverageWorkflow` into replace evaluate path (`replace_runner.py`); gate `DetectionJudgeWorkflow` behind the flag
5. Wire `EntityCoverageWorkflow` into rewrite evaluate path (`rewrite_workflow.py`); gate `DetectionJudgeWorkflow` behind the flag; remove `_detection_valid_fraction()`
6. Add `entity_coverage_judge` to `EvaluateModelSelection` and `evaluate.yaml`; keep `detection_validity_judge`; update `model_loader.py`
7. Update `_build_user_dataframe` allowed columns (`anonymizer.py`)
8. Update `display.py` rendering
9. Update `docs/concepts/evaluation.md` and `skills/anonymizer/SKILL.md`
10. Add `test_entity_coverage_judge.py`; keep `test_detection_judge.py`; update all affected existing tests
11. Run `make format && make typecheck && make test`

---

## Test Datasets

The following datasets should be used to validate entity coverage results once the feature is shipped.

| Dataset | Rows | Text column | Ground-truth annotations | Notes |
|---|---|---|---|---|
| **usage_logs_seed** (`test-data/usage_logs_seed.parquet`) | 20 | `conversation_trace` | None | LLM conversation traces with embedded PII (names, emails, phone numbers, addresses) across business document scenarios; good for sanity-checking coverage on realistic, high-density PII text |
| **openPII_ai4_part1** (`openPII_ai4_part1.csv`) | 1000 | `source_text` | `privacy_mask` (character-offset spans with labels: TITLE, GIVENNAME, SURNAME, EMAIL, TELEPHONENUM, DATE, etc.) | Synthetic PII benchmark with ground-truth spans; use `privacy_mask` as the reference to evaluate whether the judge surfaces the same entities Anonymizer missed |
| **PII_NEW_TESTDATA_part1** (`PII_NEW_TESTDATA_part1.csv`) | 1000 | `text_us` / `text_intl` | `spans_us` / `spans_intl` (character-offset spans with Anonymizer-native labels: first_name, last_name, email, date_of_birth, street_address, bank_routing_number, credit_debit_card, etc.) | Multi-domain synthetic dataset (Life, Finance, etc.) with both US and international text variants and pre-tagged versions; spans use Anonymizer's own label vocabulary, making it the most direct test for coverage scoring |

Run evaluation in both replace and rewrite modes across all three datasets to confirm `entity_coverage` scores are meaningful and `leaked_entities` lists surface real misses rather than noise. `PII_NEW_TESTDATA_part1` is the highest-signal dataset for this feature because its ground-truth spans use the same label set as Anonymizer's detection pipeline.

---

## Open Questions

- **Prompt v1:** The leakage_eval script is the starting point. The prompt will provide both the original text and the Anonymizer-detected entity list as context — the detected list is what allows the judge to identify the delta efficiently rather than re-detecting everything from scratch. Removing it would mean the judge can't compute a meaningful delta; keeping it is the right call. Once v1 is shipped and results are observed, a follow-up iteration should explore **NER-style prompting** (instructing the judge to perform explicit named-entity recognition passes over the text before computing the delta) as a potential path to improving recall and reducing missed entities.
- **`n_entities_detected` denominator:** Currently planned to count `(value, label)` pairs (flattened, same as current detection judge). If the same value has multiple labels, it counts multiple times. Consider whether to count unique values instead to avoid inflating the denominator.
- **Replace mode threshold:** "Satisfied" is defined as `entity_coverage == 1.0` (strict — even one leaked entity flips to Not Satisfied). Confirm this is the right threshold or whether a small tolerance is acceptable.
