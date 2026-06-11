# Rewrite Evaluation Improvements — Implementation Plan

## Problem

The rewrite evaluation has four related issues:

- **Judge scoring is baked into `run()` / `preview()`** — the final holistic judge (privacy / quality / style scores) runs unconditionally as part of the rewrite pipeline. Replace mode separates this into a dedicated `anonymizer.evaluate()` call. Rewrite has no equivalent, forcing users to pay the judge cost on every run even during fast iteration.
- **No detection validity score in rewrite mode** — `anonymizer.evaluate()` produces a `detection_valid` column for replace mode (via `DetectionJudgeWorkflow`). Rewrite mode runs the same GLiNER + LLM detection pipeline but never scores its accuracy.
- **Judge scores are 1–10 integers that saturate** — observed outputs cluster at the extremes (8–10 or 1–2), making the middle of the scale nearly unused. A 3-level categorical (`low` / `medium` / `high`) better matches the distribution, removes false precision, and makes rubric definitions more actionable.
- **"Naturalness" is an ambiguous name** — the dimension measures writing style and readability of the output text, not faithfulness to the original or any other concept the word might suggest. `style` is a clearer term for this and is unambiguous.

---

## Design Decision: categorical scale

**`low / medium / high`** is preferred over a boolean (`pass / fail`):

- Boolean formally acknowledges the binary behaviour but loses the middle bucket entirely. A rewrite that is "mostly fluent but has one awkward sentence" must be called either passing or failing, which is worse for human review triage.
- `low / medium / high` gives reviewers actionable signal: `medium` means "worth a look but not a clear failure", which is exactly the category that surfaces regressions in prompt or model changes.
- The `Score` class already accepts string option keys, so no framework changes are needed.

Rubric anchors for each dimension:

| Score | Privacy | Quality | Style |
|---|---|---|---|
| `high` | Original direct identifiers removed; remaining quasi-identifiers create low linkage risk | Important meaning, facts, and structure fully preserved | Reads as natural, coherent, human-written prose |
| `medium` | No obvious direct identifiers remain, but a distinctive quasi-identifier bundle creates noticeable linkage risk | Most content preserved; minor details lost or slightly distorted | Mostly readable; isolated awkward phrasing or stiff transitions |
| `low` | One or more original direct identifiers or near-equivalents remain, or the record is easily linkable | Material loss of important information, contradictions, or distorted core meaning | Noticeably unnatural; broken grammar, placeholder-like language, or machine-generated feel |

---

## Design Decision: `run()` vs `evaluate()` separation

- `run()` outputs `utility_score`, `leakage_mass`, `weighted_leakage_rate`, `any_high_leaked`, and `needs_human_review` unchanged — the repair loop requires them and they are immediately useful after a run.
- `evaluate()` adds `detection_valid` (0–1) and the holistic judge scores (`privacy` / `quality` / `style`) on top of the existing `run()` output.

This avoids re-running the repair loop in `evaluate()`: if `needs_human_review=True`, that is already the exhausted repair state and `evaluate()` simply reads the metrics already present.

---

## Scope

No new public API symbols beyond extending `EvaluateConfig` and `EvaluateModelSelection`.
All changes are backwards-compatible for replace-mode users.

---

## Files Changed

| File | Change |
|---|---|
| `src/anonymizer/engine/rewrite/final_judge.py` | Rename `NATURALNESS_RUBRIC` → `STYLE_RUBRIC`; change options to `low/medium/high`; update `_judge_prompt` scoring instructions; update `scores=` list; update `FinalJudgeWorkflow.columns()` signature to accept `EvaluateModelSelection` instead of `RewriteModelSelection`; remove `COL_NEEDS_HUMAN_REVIEW` from its column output (see Step 2) |
| `src/anonymizer/engine/rewrite/rewrite_workflow.py` | Remove `_run_final_judge` call from `run()`; add `evaluate()` method that runs detection judge + final judge |
| `src/anonymizer/interface/results.py` | Add `rewrite_config: PrivacyGoal \| None = None` field to `AnonymizerResult` and `PreviewResult`; set it during rewrite `run()` analogous to `replace_method` |
| `src/anonymizer/interface/anonymizer.py` | Extend `evaluate()` to dispatch on `rewrite_config`; add `COL_JUDGE_EVALUATION` + `COL_DETECTION_VALID` to the rewrite allowed-column set in `_build_user_dataframe` |
| `src/anonymizer/config/models.py` | Add `rewrite_judge` alias to `EvaluateModelSelection`; remove `judge` from `RewriteModelSelection` |
| `src/anonymizer/config/anonymizer_config.py` | `EvaluateConfig` is no longer a placeholder — add a docstring clarifying it covers both replace and rewrite evaluation |
| `src/anonymizer/config/default_model_configs/evaluate.yaml` | Add `rewrite_judge: nemotron-30b-thinking` — required to avoid a Pydantic startup crash when the new field lands in `EvaluateModelSelection` |
| `src/anonymizer/config/default_model_configs/rewrite.yaml` | Remove `judge` entry — it moves to `evaluate.yaml` |
| `src/anonymizer/engine/ndd/model_loader.py` | Update `validate_model_alias_references` to check `evaluate.rewrite_judge` when `check_evaluate=True` on a rewrite result |
| `src/anonymizer/interface/display.py` | Update `_render_scores_section` to not append `/10`; fix `_extract_judge_scores` to not cast `value["score"]` through `int()` — for string keys `"low"/"medium"/"high"` this raises `ValueError` which is silently swallowed, causing the judge section to never render; update return type to `list[tuple[str, int \| str]]` |
| `src/anonymizer/engine/schemas/rewrite.py` | Update any schema or docstring that references the 1–10 scale or "naturalness" |
| `docs/concepts/rewrite.md` | Update judge score documentation (rename naturalness → fluency, describe categorical scale, move judge to evaluate step) |
| `skills/anonymizer/SKILL.md` | Update evaluate workflow section with rewrite evaluate example |
| `tests/engine/rewrite/test_final_judge.py` | Update rubric tests; add tests for categorical output parsing |
| `tests/interface/test_anonymizer.py` | Add tests for `evaluate()` on a rewrite result |

`src/anonymizer/engine/evaluation/detection_judge.py` — no changes needed; already works on any trace dataframe that has `COL_TEXT` and `COL_ENTITIES_BY_VALUE`.

---

## Step 1 — Rename and recategorise the judge rubrics (`final_judge.py`)

### Rename

```python
# before
NATURALNESS_RUBRIC = Score(name="naturalness", ...)

# after
STYLE_RUBRIC = Score(name="style", ...)
```

Update `scores=` list in `FinalJudgeWorkflow.columns()`:

```python
scores=[PRIVACY_RUBRIC, QUALITY_RUBRIC, STYLE_RUBRIC],
```

### Change options to `low / medium / high`

Replace the integer-keyed `options` dicts with string keys on all three rubrics:

```python
PRIVACY_RUBRIC = Score(
    name="privacy",
    description="Privacy protection — how well the rewrite removes linkage risk to the original record.",
    options={
        "high":   "Original direct identifiers removed; remaining quasi-identifiers create low linkage risk.",
        "medium": "No obvious direct identifiers remain, but a distinctive quasi-identifier bundle creates noticeable linkage risk.",
        "low" : "The record is easily or near-certainly linkable back to the original: key direct identifiers remain, or enough identifying detail survives that re-identification requires minimal effort regardless of how many entities were successfully transformed.",
    },
)

QUALITY_RUBRIC = Score(
    name="quality",
    description="Content quality — how well important meaning, facts, and structure are preserved.",
    options={
        "high":   "Important meaning, facts, and structure fully preserved.",
        "medium": "Most content preserved; minor details lost or slightly distorted.",
        "low":    "Material loss of important information, contradictions, or distorted core meaning.",
    },
)

STYLE_RUBRIC = Score(
    name="style",
    description="Writing style — does the rewritten text read as natural, grammatically correct, human-written prose?",
    options={
        "high":   "Reads as natural, coherent, human-written prose.",
        "medium": "Mostly readable; isolated awkward phrasing or stiff transitions.",
        "low":    "Noticeably unnatural; broken grammar, placeholder-like language, or machine-generated feel.",
    },
)
```

### Update `_judge_prompt`

Replace the three `<*_scoring_instructions>` blocks to match the new categorical rubric anchors. The core guidance (assess independently, don't penalise necessary changes, etc.) is preserved — only the scale reference changes:

```
<privacy_scoring_instructions>
  ...existing contextual guidance (linkage risk, quasi-identifiers, etc.) preserved verbatim...

  Score as:
  - high   — original direct identifiers removed; remaining details create low linkage risk
  - medium — no obvious direct identifiers, but a distinctive quasi-identifier bundle creates
             noticeable linkage risk
  - low    — one or more direct identifiers or near-equivalents remain, or easily linkable
</privacy_scoring_instructions>

<quality_scoring_instructions>
  ...existing guidance preserved...

  Score as:
  - high   — important meaning, facts, and structure fully preserved
  - medium — most content preserved; minor details lost or slightly distorted
  - low    — material loss of important information, contradictions, or distorted core meaning
</quality_scoring_instructions>

<style_scoring_instructions>
  ...naturalness guidance renamed and preserved...

  Score as:
  - high   — natural, coherent, human-written prose
  - medium — mostly readable; isolated awkward phrasing or stiff transitions
  - low    — noticeably unnatural; broken grammar, placeholder-like language, or machine feel
</style_scoring_instructions>
```

The `<task>` block changes "naturalness of writing" to "style of writing".

---

## Step 2 — Move final judge out of `run()` (`rewrite_workflow.py`)

Remove the `_run_final_judge` call from `RewriteWorkflow.run()` and the `COL_JUDGE_EVALUATION` default from `_PASSTHROUGH_DEFAULTS`.

Add a standalone `evaluate()` method on `RewriteWorkflow`:

```python
def evaluate(
    self,
    df: pd.DataFrame,
    *,
    model_configs: list[ModelConfig],
    selected_models: EvaluateModelSelection,
    privacy_goal: PrivacyGoal,
    preview_num_records: int | None = None,
) -> RewriteResult:
    """Run detection validity judge and final holistic judge on a completed rewrite result.

    Mirrors ReplacementWorkflow.evaluate(): takes the trace dataframe from a
    prior run() / preview() and appends COL_DETECTION_VALID,
    COL_DETECTION_INVALID_ENTITIES, and COL_JUDGE_EVALUATION.
    """
```

Inside `evaluate()`:

1. **Split entity vs passthrough rows** using the same `split_rows` / `_has_entities` pattern as `run()`. Passthrough rows receive `COL_DETECTION_VALID = None` and `COL_JUDGE_EVALUATION = None` as defaults — running either judge on them produces vacuously correct or misleadingly high scores for records that were never anonymized.
2. Run `DetectionJudgeWorkflow` against entity rows only, using `COL_ENTITIES_BY_VALUE` + `COL_TEXT`. The score is surfaced as a **0–1 value** matching the scale of `utility_score` and `leakage_mass`. `COL_DETECTION_VALID` appears alongside the objective scores in the user dataframe and display, **not** grouped with the judge scores (privacy/quality/style). It does **not** influence `COL_NEEDS_HUMAN_REVIEW`.
3. Run `FinalJudgeWorkflow` against entity rows only, for privacy / quality / style scores.
4. Merge entity and passthrough rows and return a new `RewriteResult`.

### `COL_NEEDS_HUMAN_REVIEW` must not be overwritten

`COL_NEEDS_HUMAN_REVIEW` is set correctly during `run()` based on objective metrics (utility/leakage thresholds). `evaluate()` must never touch it — judge scores and detection validity do not influence the human review decision. Remove `COL_NEEDS_HUMAN_REVIEW` from `FinalJudgeWorkflow.columns()` and produce it instead at the end of `_run_evaluate_repair_loop()` in `rewrite_workflow.py`, where the threshold params are already available.

### Telemetry — `rewrite.judge` reference

`anonymizer.py` references `rewrite.judge` in two places that will break when the field is removed from `RewriteModelSelection`:

- `_collect_step_models()` — `"judge": rewrite.judge if has_rewrite else NOT_APPLICABLE`
- `_build_telemetry_event()` — `judge_model=models["judge"]`

Resolution: drop the `judge` key from the `_collect_step_models` rewrite block and from `_build_telemetry_event`. The judge is now an evaluate-time role, not a run-time role, so it doesn't belong in run telemetry. `anonymizer.py` is already in the Files Changed table for the `evaluate()` dispatch; this telemetry fix is part of the same change.

---

## Step 3 — Wire `Anonymizer.evaluate()` for rewrite results (`anonymizer.py`)

The existing `evaluate()` currently raises if `output.replace_method` is `None`:

```python
# before
replace_method = getattr(output, "replace_method", None)
if replace_method is None:
    raise ValueError(...)
```

Extend the dispatch:

```python
rewrite_config = getattr(output, "rewrite_config", None)
replace_method = getattr(output, "replace_method", None)

if rewrite_config is not None:
    # Rewrite evaluate path
    ...call self._rewrite_runner.evaluate(...)
elif replace_method is not None:
    # Replace evaluate path (unchanged)
    ...
else:
    raise ValueError(...)
```

`AnonymizerResult` / `PreviewResult` in `results.py` need a `rewrite_config` field (carrying `PrivacyGoal`) set during `run()` in rewrite mode — analogous to how `replace_method` is set in replace mode.

### Update `_build_user_dataframe`

Add `COL_JUDGE_EVALUATION`, `COL_DETECTION_VALID`, and `COL_DETECTION_INVALID_ENTITIES` to the rewrite allowed set. The objective metrics (`utility_score`, `leakage_mass`, etc.) are already in the allowed set and remain there — they are present after `run()`. The new columns are only present after `evaluate()` and are silently omitted until then.

```python
if f"{text_col}_rewritten" in t.columns:
    allowed = {
        text_col,
        f"{text_col}_rewritten",
        COL_UTILITY_SCORE,
        COL_LEAKAGE_MASS,
        COL_WEIGHTED_LEAKAGE_RATE,
        COL_ANY_HIGH_LEAKED,
        COL_NEEDS_HUMAN_REVIEW,
        COL_DETECTION_VALID,            # ← new, only present after evaluate()
        COL_DETECTION_INVALID_ENTITIES, # ← new, only present after evaluate()
        COL_JUDGE_EVALUATION,           # ← new, only present after evaluate()
    }
```

---

## Step 4 — Update model selection (`models.py`)

Move the `judge` alias out of `RewriteModelSelection` and into `EvaluateModelSelection`:

```python
class EvaluateModelSelection(BaseModel):
    detection_validity_judge: str
    replace_type_fidelity_judge: str
    replace_relational_consistency_judge: str
    replace_attribute_fidelity_judge: str
    rewrite_judge: str  # ← new: holistic privacy/quality/fluency judge for rewrite evaluate
```

`RewriteModelSelection.judge` is removed (or kept with a deprecation note if model YAML defaults need a phased migration).

Update `engine/ndd/model_loader.py` validation to check `evaluate.rewrite_judge` when `check_evaluate=True` and the output is a rewrite result.

Update `FinalJudgeWorkflow.columns()` to accept `EvaluateModelSelection` instead of `RewriteModelSelection`, and resolve the judge alias via `evaluate.rewrite_judge` instead of `rewrite.judge`.

---

## Step 5 — Fix display rendering (`display.py`)

Line 449 currently renders:

```python
score_strs = [f"{name}: {score}/10" for name, score in judge_scores]
```

Change to:

```python
score_strs = [f"{name}: {score}" for name, score in judge_scores]
```

`_extract_judge_scores` returns `list[tuple[str, int]]` — update the return type to `list[tuple[str, int | str]]` since scores are now strings.

### Detection validity placement

`COL_DETECTION_VALID` must be rendered in the **main scores section** (alongside `utility_score` and `leakage_mass`), not inside the judge scores block. Update `_render_scores_section` to include it there when present. The value is already 0–1 so no `/10` suffix is needed and no additional scaling is required.

### "Rewrite Need Review" label

In rewrite mode the `COL_NEEDS_HUMAN_REVIEW` column must be displayed as **"Rewrite Need Review"** (not the generic "Needs Review" used in replace mode). Update the label resolution in `display.py` to emit the rewrite-specific label when rendering a rewrite result.

### Invalid-entities expandable table

When `COL_DETECTION_VALID < 1.0` and `COL_DETECTION_INVALID_ENTITIES` contains entries, append a collapsed `<details>` block below the scores `<div>`. The summary line reads "Show N flagged detection(s)"; expanding it reveals a three-column table (Value / Label / Reason). This is conditional — records with `detection_valid == 1.0` or no invalid-entity entries show no extra element. The `_extract_judge_scores` fix (remove `int()` cast) is required for string scores to appear at all; without it, `ValueError` was silently swallowed for `"low"/"medium"/"high"` values, causing the judge section to never render.

---

## Step 6 — Docs and skills

### `docs/concepts/rewrite.md`

- Output columns table: remove `judge evaluation` from the `run()` output section; add a new **Evaluation** subsection (parallel to the existing replace evaluate docs) showing the `evaluate()` call pattern and what columns it adds.
- Update the judge score description: rename "naturalness" → "style", describe `low/medium/high` scale; note detection validity appears in the main scores section (0–1) not judge scores; note `COL_NEEDS_HUMAN_REVIEW` is labelled "Rewrite Need Review" in the output column table.
- Model roles table: move `judge` from the rewrite pipeline roles to the evaluate roles.

### `skills/anonymizer/SKILL.md`

Add a rewrite evaluate workflow example alongside the existing replace evaluate example:

```python
# after rewrite run / preview:
evaluated = anonymizer.evaluate(result)
evaluated.display_record(0)
# → adds detection_valid (0–1, main scores section), judge evaluation (privacy/quality/style: low/medium/high)
```

---

## Step 7 — Tests

### Update existing tests

- `tests/engine/rewrite/test_final_judge.py` — update rubric option assertions for `low/medium/high`; update any test that checks score parsing for integer values; rename all `naturalness` references to `style`.
- `tests/interface/test_anonymizer.py` — update assertions that check `COL_JUDGE_EVALUATION` is in the `run()` output (it now only appears after `evaluate()`).

### New tests to add

```
# final_judge.py
test_style_rubric_has_low_medium_high_options
test_privacy_rubric_has_low_medium_high_options
test_quality_rubric_has_low_medium_high_options
test_judge_prompt_references_style_not_naturalness
test_judge_prompt_references_categorical_scale

# rewrite_workflow.py
test_run_does_not_produce_judge_evaluation_column
test_evaluate_produces_judge_evaluation_column
test_evaluate_produces_detection_valid_column

# anonymizer.py
test_evaluate_rewrite_result_adds_judge_columns
test_evaluate_rewrite_result_adds_detection_valid
test_evaluate_rewrite_raises_without_rewrite_config
test_run_rewrite_does_not_include_judge_in_user_dataframe

# display.py
test_render_scores_section_categorical_no_slash_10
test_extract_judge_scores_returns_string_scores
test_extract_judge_scores_categorical_not_silently_empty
test_detection_valid_rendered_in_main_scores_section
test_rewrite_needs_human_review_label_is_rewrite_need_review

# rewrite_workflow.py — passthrough + needs_human_review
test_evaluate_skips_passthrough_rows
test_evaluate_passthrough_rows_get_none_judge_defaults
test_run_needs_human_review_not_overwritten_by_evaluate

# anonymizer.py — telemetry
test_run_rewrite_telemetry_has_no_judge_field
```

All new tests construct result objects directly — no real pipeline or LLM calls.

---

## Implementation Order

1. Update rubrics and prompt in `final_judge.py` (rename naturalness → style, 1-10 → low/medium/high)
2. Move `_run_final_judge` out of `RewriteWorkflow.run()`; add `RewriteWorkflow.evaluate()`
3. Add `rewrite_config` field to `AnonymizerResult` / `PreviewResult`; wire `Anonymizer.evaluate()` for rewrite
4. Move `judge` alias from `RewriteModelSelection` to `EvaluateModelSelection` (as `rewrite_judge`); update model loader validation
5. Update `_build_user_dataframe` allowed columns for rewrite
6. Fix `display.py` score rendering
7. Update `docs/concepts/rewrite.md` and `skills/anonymizer/SKILL.md`
8. Update existing tests; add new tests
9. Run `make format && make typecheck && make test`
