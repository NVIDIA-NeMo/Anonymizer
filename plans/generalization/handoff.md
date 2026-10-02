<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Generalization and rewrite handoff — 2026-10-02

Read this file first when resuming. It describes the current state; implementation.md includes historical decisions that were superseded.

## Branch and scope

Current branch: asteier2026/feature/generalization, tracking origin.
Stack: entity-direct-quasi-classification (3383957) → sensitivity-disposition-plan (ed31d7e) → generalization (previous pushed checkpoint ca16f61, followed by today's checkpoint).
User requested saving and pushing today's changes. Do not merge or open a PR without further direction.
No public model configuration was added; generator and reviewer both use the configured rewriter model. Repair uses repairer.

## Current behavior

- Synthetic replacement generation receives all explicit detected entities per document in one request, before sensitivity filtering. Explicit input comes from final_entities → _entities_by_value, not the direct/quasi classifier. Latent entities are stored separately.
- Filter the generated map to replace dispositions before generalization/rewrite. Discarding other synthetic values can break jointly generated age/date consistency.
- Generalization: targets → candidate LLM call → stripped review input (IDs and proposed values only) → independent reviewer LLM call → deterministic validation. Both calls skip rows without targets.
- Reviewer reports defects with evidence and conflicting IDs before a complete corrected suggestion set. Candidate reasons/statuses are withheld to reduce endorsement. Validation checks coverage and IDs, not semantic privacy.
- Trace columns: _raw_generalization_suggestions, _generalization_review_input, _reviewed_generalization_suggestions (includes defects), _generalization_suggestions (canonical), _generalization_needs_review.
- Rewrite receives pre-replaced tagged text plus _rewrite_actions: generalize, remove, suppress_latent_inferences. No replacement map or full sensitivity disposition is passed to the writer.
- Action builder excludes replace/leave_as_is, maps no_effective_generalization to remove, joins original latent evidence/rationale through stable numbered IDs. Explicit suppress_inference is also supported.
- Generalize/remove entries omit superseded sensitivity reasons and reviewer justifications. Removals use fixed omission wording, never free-form reviewer replacement suggestions. Overridden guidance is retained in _rewrite_action_diagnostics; this records all overrides, not proven contradictions.
- Protected latent actions retain evidence, rationale, and protection reason. Evidence may contain original identifiers; prompt forbids restoring them.
- Repair now receives ONLY current rewrite, privacy goal, and leaked-item feedback. No original/baseline document, full disposition, generalization/actions, or aggregate scores in its prompt. Scores still drive pipeline decisions.
- User explicitly preferred leaked-item-focused repair. Do not reintroduce planning metadata or baseline text without discussing it.
- Repair feedback is a plain-text list for privacy answers marked yes: sensitivity, entity label, original protected value, question, confidence, evaluator reason, optional quoted evidence. Those values remain a possible source of disclosure.
- Repair prompt requires generalization/removal only, no invented facts or increased specificity. It does not assume knowledge of synthetic values or earlier omissions.

## Evaluation findings

Evaluation CSVs are local in ~/Documents/rewrite_results_new/; they have not been committed or uploaded.

- Run 9: 85 suggestions, no values changed by reviewer; 14 statuses changed for irrelevant repeated gender/spouse concerns.
- Run 10: 87 suggestions, six wording changes, six defects, two no_effective_generalization results. Useful state-disclosure correction, but overly broad removal reasoning and missed conflicts. Rewrite ignored an explicit age omission.
- Run 11: explicit action lists helped remove Jodi's degree and Nancy's political view. Bobby's removal entries still contained contradictory reviewer generalization wording; this prompted canonical omission instructions. Repair restored family details and changed a synthetic name.
- Run 12: Nancy's initial rewrite obeyed age omission; repair restored 21-year-old, English, and female pronouns from the baseline/feedback. Jodi's children were restored. Triggered removal of baseline text from repair.
- Run 13: ONLY FOUR records, not five. Missing Idilio Bell, record ID a280a834a8e6573eb1df009fd30503e1. CSV alone gives no failure reason. Inspect failed-record output/logs before drawing dataset-wide conclusions.
- Run 13 current-text-only repair still invents or increases specificity: Bobby early forties → thirties; Jodi lives on Oakridge → lives in Oakridge, a town; publishing industry becomes explicit; Nancy centrist → moderate. James's English omission was ignored already in initial rewrite. Gender suppression and empty phrases still fail.
- Evaluator weaknesses: mid-30s treated as evidence of exact age 36; gender sometimes missed despite she; publishing industry missed despite publishing company in earlier runs; citizenship inferred from residence/work. Do not treat zero leakage as proof of successful protection.
- Full runs change detection/disposition/latent inventories, so score differences do not isolate prompt effects. Match rows by original text/record ID, not row position (run 13 omits the second biography).

## Next discussion / proposed work

No further architecture change has been approved. Proposed:
1. Account for missing run 13 record using failed-record output or logs.
2. Freeze current rewrite and leaked-item feedback for controlled repair-only comparisons.
3. Use regression cases: no invented age, no street-to-town conversion, no explicit protected industry, no synonym-only repair.
4. Evaluate initial rewrite separately from repair; max_repair_iterations=0 can help, but freezing upstream inputs is better.
Avoid adding more broad prompt instructions until failures are isolated. Additional validators, model changes, replay harnesses, and evaluator calibration were discussed but not implemented.

## Files and verification

Core: engine/rewrite/generalization.py, rewrite_generation.py, repair.py; engine/schemas/generalization.py; engine/constants.py.
Tests: tests/engine/test_generalization.py, test_rewrite_generation.py, test_repair.py, test_prompt_templates.py, combined/legacy workflow tests.
Run: .venv/bin/python -m pytest tests/engine tests/interface/test_display.py -q
Run: .venv/bin/ty check --error-on-warning
Commit hooks run repository-wide checks; do not bypass them.
Restart notebook kernel before live runs to load edits.

Unrelated untracked local artifacts remain intentionally outside commits: anonymizer_claude_session.md, docs/notebooks/01_replace.ipynb, 02_detection_debug.ipynb, 03_rewrite.ipynb, models0630.yaml. They remain on disk. Do not delete or blindly commit them.

## Local result manifest

Hashes identify the exact evaluated files without placing data in Git.

- rewrite_synth_bio5_sensdisp_branch_eval_run6.csv: 5 records; SHA-256 c2f17dc493e908775dc9f912854be715fa76c70e15ffcc9ed3b61b5ffd924272

- rewrite_synth_bio5_sensdisp_branch_eval_run8.csv: 5 records; SHA-256 62eff2ca139a143fd9376d8b418da58b0e9128449ab81e10f36fb657897a6f68

- rewrite_synth_bio5_sensdisp_branch_eval_run9.csv: 5 records; SHA-256 8a5bda9490fdde341aba658f7e64cc84c1c278f974f52d5d0a2848025457014b

- rewrite_synth_bio5_sensdisp_branch_eval_run10.csv: 5 records; SHA-256 e26b9abd202bd78434c7836925ec09c99f995b484224c914c8aa00dafeeca95c

- rewrite_synth_bio5_sensdisp_branch_eval_run11.csv: 5 records; SHA-256 cbdcb912ff81e77d9fb353f62906518719a63bd3be2a316bfe18503c9d1d5b1e

- rewrite_synth_bio5_sensdisp_branch_eval_run12.csv: 5 records; SHA-256 b60f060765a9819c046e8bad4aacc6defec7931314769a7ed8d725a60b64083f

- rewrite_synth_bio5_sensdisp_branch_eval_run13.csv: 4 records; SHA-256 50469f742ac628999d95193954c6620e5e12e7165898d3a0312e4bd17594f089
