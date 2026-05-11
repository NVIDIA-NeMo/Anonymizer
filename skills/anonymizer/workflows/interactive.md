<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Interactive Workflow

Iterative design with the user. Do not disengage from the loop until the user says they're satisfied.

1. **Verify install** — Run `python -c "import anonymizer; print(anonymizer.__version__)"`.
   - If the import fails, STOP and follow the Troubleshooting section in `SKILL.md`. Do not continue.

2. **Inspect the data** — Read the first few rows of the source file with pandas. You need to know:
   - Path, format, encoding.
   - The column that holds the text to anonymize.
   - The shape of the text (one sentence vs paragraphs vs multi-page; structured vs prose).
   - Any obvious entities you can see (names, IDs, dates, etc.) — useful as a sanity check later.

3. **Clarify** — Ask the user the questions you need to write a config. Keep the set short; batch related questions; provide concrete options where possible. Common things to make precise:
   - **Goal**: scrub PII for retention/sharing? produce training data? demos? compliance?
   - **Mode**: Replace (entities only) vs Rewrite (full text transformation that also removes inferable identifiers like "ringing the bell" → cancer)? Default per the rule in `SKILL.md`.
   - **For Replace**: which strategy? (`Substitute` for realistic-looking, `Redact` for explicit `[REDACTED_…]` markers, `Hash` for stable cross-row identifiers, `Annotate` for inspection only).
   - **For Rewrite**: what must be protected? what must be preserved? how strict (`risk_tolerance`)? Read [`docs/concepts/choosing-a-strategy.md`](../../../docs/concepts/choosing-a-strategy.md) sections 5–6 with the user's answers in mind.
   - **Domain-specific entity labels** the defaults won't cover (e.g. `"medical record number"`, `"case number"`, `"internal project codename"`). If yes, read [`docs/concepts/choosing-a-strategy.md`](../../../docs/concepts/choosing-a-strategy.md) section 2.
   - **Cross-record consistency requirement** (does the same person/ID need the same replacement everywhere)? If yes, use `Hash`; do not promise this with `Substitute`.

4. **Plan** — Briefly state the config you intend to write (mode, strategy, key fields, any non-default detection knobs). Confirm with the user before writing the script.

5. **Build** — Write the Python script to the current directory using the Output Template in `SKILL.md`. Name the file after the dataset (e.g. `anonymize_clinical_notes.py`).

6. **Preview** — Run `python <script>.py` (no flags = preview 5 rows).
   - **First check `failed_records`.** If non-empty, STOP iterating on strategy. The script's failure-first guard already exits non-zero. Read [`docs/troubleshooting.md`](../../../docs/troubleshooting.md) "Did the run actually complete cleanly?" and "Rate limits / dropped rows". Fix the infra issue, then re-preview.
   - For Rewrite, read the leakage / utility summary the script prints. For Replace, eyeball the output rows directly.
   - For deeper inspection, open a Python session and `from <script> import build_config; ... result = anonymizer.preview(...)` to get back the live `result.trace_dataframe` for filtering and analysis.

7. **Iterate**
   - Ask the user how the preview looks.
   - If quality is off, consult [`docs/troubleshooting.md`](../../../docs/troubleshooting.md) at the relevant symptom (`leakage_mass too high`, `utility_score too low`, `Most rows have needs_human_review=True`, etc.). Apply the suggested knob change to the script, re-preview.
   - Loop until the user is satisfied.

8. **Finalize** — Tell the user they can run the full pipeline with:
   - `python <script>.py --full`
   - Caution that wall-clock and cost depend heavily on dataset size, model choice, and (for Rewrite) `max_repair_iterations`. Suggest they start by running on a larger sample (e.g. `--num-records 100`) before committing to the full dataset.
   - Remind them: Anonymizer is best-effort. Rewrite outputs with `needs_human_review=True` should be reviewed.
   - Do not run `--full` yourself unless the user explicitly asks. The user controls when expensive work runs.
