# Entity Coverage — Implementation Plan

## Problem

Currently `detection_valid` scores **precision** of the detection step. Observed scores are predominantly in the lower ranges (issue #176), partly because the judge flags over-detection and mislabeling among other issues. But the actual privacy-sensitive question users care about is the opposite: **did any real PII slip through the anonymizer?**

Precision complaints are already surfaced in the entity dropdown for manual inspection, making `detection_valid` feel redundant as a headline metric. What's missing is a **recall** signal: a score that answers "of all the PII that existed, how much did we successfully catch?"

---

## Proposed Solution: Entity Coverage

Replace `detection_valid` with `entity_coverage`:

```
entity_coverage = n_entities / (n_entities + n_leaked)
```

- **`n_entities`** — entities detected by Anonymizer before anonymization (already available in `COL_ENTITIES_BY_VALUE`)
- **`n_leaked`** — the number of missed identifiers from the original text, detected by the evaluation LLM in the anonymized / rewritten output
- **`leaked_entities`** — the list of missed entities (shown in a dropdown when coverage < 100%)

The evaluation LLM runs on the **original text** and detects all PII that should have been caught. The delta between what it finds and what Anonymizer already detected gives the leaked entities.

**Key shift:** current judge asks "were our detections correct?" (precision). New judge asks "did any PII survive?" (recall).

---

## Design Decisions

### Coverage display differs by mode

| Mode | Display | Rationale |
|---|---|---|
| **Replace** | Satisfied / Not Satisfied | Binary is sufficient — either all PII was caught or it wasn't |
| **Rewrite** | Float fraction (e.g. `0.87`) | Rewrite already exposes numeric metrics (leakage_mass, utility_score); a fraction fits the pattern |

A value of `1.0` (or Satisfied) means no PII leaked into the output. Any value below `1.0` (or Not Satisfied) means at least one entity was missed, and `leaked_entities` names them.

### Passthrough rows

Rows where Anonymizer detected **zero entities** trivially score `entity_coverage = 1.0` — there was nothing to protect, so nothing could have leaked. The LLM judge is skipped for these rows (same pattern as the current passthrough shortcut in `_BaseJudgeWorkflow`).

### Prompt starting point

The entity judge prompt is based on the leakage eval script from the experiments repo. It will be refined as it is tested against real examples. The core task: given the original text, identify all PII — then report any that Anonymizer missed.

### Reuse `_BaseJudgeWorkflow`

The new `EntityCoverageWorkflow` extends `_BaseJudgeWorkflow`. The partition-LLM-merge skeleton is identical; only the schema, prompt, and post-process step differ (coverage fraction computed from counts, not a boolean verdict field). `postprocess()` is overridden in the subclass — the base class stays unchanged.

---

## Scope

- Replaces `detection_valid` and `detection_invalid_entities` in both replace and rewrite `evaluate()` output.
- No new public API symbols beyond new column names (`entity_coverage`, `leaked_entities`) and a renamed model role.
- `COL_DETECTION_VALID` / `COL_DETECTION_INVALID_ENTITIES` constants are retired; new constants added.
- `DetectionJudgeWorkflow` is replaced by `EntityCoverageWorkflow` — the old class is removed, not kept as an alias.

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
| `src/anonymizer/engine/constants.py` | Retire `COL_DETECTION_JUDGE`, `COL_DETECTION_VALID`, `COL_DETECTION_INVALID_ENTITIES`; add `COL_ENTITY_COVERAGE_JUDGE`, `COL_ENTITY_COVERAGE`, `COL_LEAKED_ENTITIES` |
| `src/anonymizer/engine/evaluation/detection_judge.py` | Replace entirely with `entity_coverage_judge.py` — new schema, new prompt, new workflow class |
| `src/anonymizer/engine/evaluation/judge_base.py` | No changes needed — `postprocess()` is overridden in the subclass |
| `src/anonymizer/engine/rewrite/rewrite_workflow.py` | Swap `DetectionJudgeWorkflow` → `EntityCoverageWorkflow` in `evaluate()`; remove `_detection_valid_fraction()`; update column references |
| `src/anonymizer/engine/replace/replace_runner.py` | Swap `DetectionJudgeWorkflow` → `EntityCoverageWorkflow`; update column references |
| `src/anonymizer/interface/anonymizer.py` | Update `_build_user_dataframe` allowed columns for both modes; update model role references |
| `src/anonymizer/interface/display.py` | Render `entity_coverage` as Satisfied/Not Satisfied (replace) or fraction (rewrite); update leaked_entities dropdown |
| `src/anonymizer/config/models.py` | Rename `detection_validity_judge` → `entity_coverage_judge` in `EvaluateModelSelection` |
| `src/anonymizer/config/default_model_configs/evaluate.yaml` | Rename the model role key |
| `docs/concepts/evaluation.md` | Replace Entity Detection Judge section with Entity Coverage section in both replace and rewrite docs |
| `skills/anonymizer/SKILL.md` | Update evaluation tips and output template column references |
| `tests/engine/evaluation/test_detection_judge.py` | Replace with `test_entity_coverage_judge.py` — new schema, prompt, and workflow tests |
| `tests/engine/test_rewrite_workflow.py` | Update column name references; remove `_detection_valid_fraction` tests |
| `tests/engine/test_evaluate.py` | Update column name assertions |
| `tests/interface/test_display.py` | Update rendering assertions for entity coverage |

---

## Step 1 — New constants (`constants.py`)

Retire the three detection judge constants and add three replacements:

```python
# Retired (remove):
# COL_DETECTION_JUDGE = "_detection_judge"
# COL_DETECTION_VALID = "detection_valid"
# COL_DETECTION_INVALID_ENTITIES = "detection_invalid_entities"

# New:
COL_ENTITY_COVERAGE_JUDGE = "_entity_coverage_judge"   # internal raw output
COL_ENTITY_COVERAGE       = "entity_coverage"          # user-facing float | None
COL_LEAKED_ENTITIES       = "leaked_entities"          # user-facing list
```

---

## Step 2 — New workflow (`entity_coverage_judge.py`)

Replace `detection_judge.py` with a new file. Key differences from the old judge:

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

### Passthrough condition

Rows with no detected entities trivially score `entity_coverage = 1.0` — nothing to miss, LLM skipped.

### Coverage fraction (postprocess override)

```python
def _compute_coverage(self, n_entities: int, n_leaked: int) -> float:
    total = n_entities + n_leaked
    return 1.0 if total == 0 else n_entities / total
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

`postprocess()` is overridden to compute the float fraction from `n_entities` (from `COL_ENTITIES_BY_VALUE`) and `len(leaked_entities)` rather than storing a boolean verdict.

> **Note:** `COL_ENTITIES_BY_VALUE` (`_entities_by_value`) is derived from `COL_FINAL_ENTITIES` (`final_entities`) during detection — it groups spans by value with a labels list. The leakage eval script references `final_entities`, which is the same underlying data in a different shape. `COL_ENTITIES_BY_VALUE` is the right source here since it's already what the judge's `prepare()` step and passthrough mask consume.

> **Note on base class / VALID_COL conflict:** `_BaseJudgeWorkflow` does not write to `VALID_COL` before `postprocess()` runs — the base class `evaluate()` calls `postprocess()` as the only step that touches those columns. Overriding `postprocess()` fully in `EntityCoverageWorkflow` is safe. However, `VALID_COL` on the base class carries an implicit `bool | None` semantic (used by other judges). Since `EntityCoverageWorkflow` writes a `float | None` instead, confirm that no shared display or downstream code reads `VALID_COL` generically and assumes boolean before this is merged.

---

## Step 3 — Wire into replace and rewrite evaluate paths

### Replace (`replace_runner.py`)

Swap `DetectionJudgeWorkflow` → `EntityCoverageWorkflow`. The judge runs on `COL_TEXT` (original text) — no change to the input column needed. Update column references for the new output columns.

### Rewrite (`rewrite_workflow.py`)

Same swap. The judge also runs on `COL_TEXT` (original text), same as replace. Update `evaluate()`:

- Remove `_detection_valid_fraction()` — no longer needed since `entity_coverage` is already a float from `postprocess()`.
- Update the try/except around the judge call to default `entity_coverage = None` and `leaked_entities = None` on failure.

---

## Step 4 — Model role rename (`models.py`, `evaluate.yaml`)

```python
class EvaluateModelSelection(BaseModel):
    entity_coverage_judge: str              # replaces detection_validity_judge
    replace_type_fidelity_judge: str
    replace_relational_consistency_judge: str
    replace_attribute_fidelity_judge: str
    rewrite_judge: str
```

Update `evaluate.yaml` default to rename the key. Update `model_loader.py` validation for the new role name.

---

## Step 5 — Update `_build_user_dataframe` (`anonymizer.py`)

Swap old column names for new in the allowed sets for both replace and rewrite mode:

```python
# Replace (was: COL_DETECTION_VALID, COL_DETECTION_INVALID_ENTITIES)
COL_ENTITY_COVERAGE,
COL_LEAKED_ENTITIES,

# Rewrite (same swap)
COL_ENTITY_COVERAGE,
COL_LEAKED_ENTITIES,
```

---

## Step 6 — Display (`display.py`)

`display_record()` renders the coverage column:

- **Replace:** `entity_coverage == 1.0` → "Satisfied"; anything less → "Not Satisfied"
- **Rewrite:** render as numeric fraction (e.g., `0.87`)

The `leaked_entities` dropdown replaces `detection_invalid_entities` with the same layout (value, label, reasoning per row).

---

## Step 7 — Docs and skills

### `docs/concepts/evaluation.md`

Replace the **Entity Detection Judge** subsection in both Replace and Rewrite sections with an **Entity Coverage** subsection:

- Explain the recall focus (vs precision of the old judge)
- Document `entity_coverage` type per mode (float in rewrite, Satisfied/Not Satisfied in replace)
- Document `leaked_entities` list shape
- Update the special-values table (passthrough row → `1.0` / Satisfied)
- Update model role name (`entity_coverage_judge`)

### `skills/anonymizer/SKILL.md`

- Update the evaluation tips bullet: `entity_coverage_judge` role; `entity_coverage` type per mode; `leaked_entities` dropdown
- Update output template: replace `detection_valid` column references with `entity_coverage`

---

## Step 8 — Tests

### New test file (`test_entity_coverage_judge.py`)

```
test_judge_prompt_runs_on_original_text_column
test_judge_prompt_includes_detected_entity_context
test_entity_coverage_schema_accepts_empty_leaked_list
test_entity_coverage_schema_accepts_leaked_entities
test_coverage_fraction_all_caught                     # n_leaked=0 → 1.0
test_coverage_fraction_partial_miss                   # n_leaked=1, n_entities=3 → 0.75
test_coverage_fraction_total_miss                     # n_leaked=n, n_entities=0 → 0.0
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

1. Add new constants to `constants.py`; retire old ones
2. Write `entity_coverage_judge.py` — schema, prompt (v1 from leakage_eval script), workflow class, `postprocess()` override
3. Wire into replace evaluate path (`replace_runner.py`)
4. Wire into rewrite evaluate path (`rewrite_workflow.py`); remove `_detection_valid_fraction()`
5. Rename model role in `models.py`, `evaluate.yaml`, `model_loader.py`
6. Update `_build_user_dataframe` allowed columns (`anonymizer.py`)
7. Update `display.py` rendering
8. Update `docs/concepts/evaluation.md` and `skills/anonymizer/SKILL.md`
9. Replace `test_detection_judge.py` with `test_entity_coverage_judge.py`; update all affected tests
10. Run `make format && make typecheck && make test`

---

## Open Questions

- **Prompt v1:** The leakage_eval script is the starting point. The prompt will provide both the original text and the Anonymizer-detected entity list as context — the detected list is what allows the judge to identify the delta efficiently rather than re-detecting everything from scratch. Removing it would mean the judge can't compute a meaningful delta; keeping it is the right call.
- **`n_entities` denominator:** Currently planned to count `(value, label)` pairs (flattened, same as current detection judge). If the same value has multiple labels, it counts multiple times. Consider whether to count unique values instead to avoid inflating the denominator.
- **Replace mode threshold:** "Satisfied" is defined as `entity_coverage == 1.0` (strict — even one leaked entity flips to Not Satisfied). Confirm this is the right threshold or whether a small tolerance is acceptable.
