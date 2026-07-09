<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Interactive Workflow

Iterative design with the user. Do not disengage from the loop until the user says they're satisfied.

1. **Verify environment**
   - **Install**: run `python -c "import anonymizer; print(anonymizer.__version__)"`. If the import fails, STOP and follow the Troubleshooting section in `SKILL.md`.
   - **Model providers**: plain `Anonymizer()` loads bundled providers from `src/anonymizer/config/default_model_configs/providers.yaml`. Before going further, confirm the API key for those defaults is set (`NVIDIA_API_KEY` for build.nvidia.com). Only ask for a custom `providers.yaml` when the user targets a non-default endpoint. If the key is missing, STOP and walk the user through [`docs/concepts/models.md`](../../../docs/concepts/models.md) or the [published models guide](https://nvidia-nemo.github.io/Anonymizer/concepts/models/).

2. **Inspect the data** — Read the first few rows of the source file with pandas. You need to know:
   - Path, format, encoding.
   - The column that holds the text to anonymize.
   - The shape of the text (one sentence vs paragraphs vs multi-page; structured vs prose).
   - Any obvious entities you can see (names, IDs, dates, etc.) — useful as a sanity check later.

3. **Clarify** — Ask the user the questions you need to write a config. Keep the set short; batch related questions; provide concrete options where possible. Common things to make precise:
   - **Goal**: scrub PII for retention/sharing? produce training data? demos? compliance?
   - **Mode**: ask the user, describing both options:
     - **Replace** — detect entities and replace each in place. Faster, cheaper, keeps document structure. Best for structured records, logs, or single-cell PII.
     - **Rewrite** — transform the full text to also remove inferable identifiers (e.g. "during her third round of chemo" implies cancer treatment). More expensive, may shorten or restructure. Best for free-text with implicit identifiers (clinical notes, biographies, depositions).
   - **For Replace**: which strategy? Default to `Substitute` if the user hasn't specified. (`Substitute` for realistic-looking, `Redact` for explicit `[REDACTED_…]` markers, `Hash` for stable cross-row identifiers, `Annotate` for inspection only).
   - **For Rewrite**: what must be protected? what must be preserved? how strict (`risk_tolerance`)? Read sections 5–6 of [`docs/concepts/choosing-a-strategy.md`](../../../docs/concepts/choosing-a-strategy.md), or the [published strategy guide](https://nvidia-nemo.github.io/Anonymizer/concepts/choosing-a-strategy/), with the user's answers in mind.
   - **Domain-specific entity labels** the defaults won't cover (e.g. `"clinical_facility"`, `"case_number"`, `"internal_project_codename"`). If yes, read section 2 of [`docs/concepts/choosing-a-strategy.md`](../../../docs/concepts/choosing-a-strategy.md) or the [published strategy guide](https://nvidia-nemo.github.io/Anonymizer/concepts/choosing-a-strategy/).
   - **Cross-record consistency requirement** (does the same person/ID need the same replacement everywhere)? If yes, use `Hash`; do not promise this with `Substitute`.
   - **Model providers**: use shipped defaults (`Anonymizer()` — calls `build.nvidia.com` via `NVIDIA_API_KEY`) or a custom `providers.yaml`? Defaults are right for most cases; only ask for a path if the user has a non-NVIDIA endpoint or a specific deployment to target. If custom, capture the path now so the script can pass it via `Anonymizer(model_providers="path/to/providers.yaml")`.

4. **Plan** — Briefly state the config you intend to write (mode, strategy, key fields, any non-default detection knobs). Confirm with the user before writing the script.

5. **Build** — Write the Python script to the current directory using the Output Template in `SKILL.md`. Name the file after the dataset (e.g. `anonymize_clinical_notes.py`).

6. **Preview** — Run `python <script>.py` (no flags = preview 5 rows).
   - **First check `failed_records`.** If non-empty, STOP iterating on strategy. The script's failure-first guard already exits non-zero. Read "Did the run actually complete cleanly?" and "Rate limits / dropped rows" in [`docs/troubleshooting.md`](../../../docs/troubleshooting.md) or the [published troubleshooting guide](https://nvidia-nemo.github.io/Anonymizer/troubleshooting/). Fix the infra issue, then re-preview.
   - For Rewrite, read the leakage / utility summary the script prints. For Replace, eyeball the output rows directly.
   - For deeper inspection: the script saves `preview.parquet` (trace dataframe — superset of the user-facing columns). Load with `pd.read_parquet("preview.parquet")` to inspect per-entity validation decisions, sensitivity dispositions, and other trace columns.

7. **Iterate**
   - Ask the user how the preview looks.
   - If quality is off, consult the relevant symptom in [`docs/troubleshooting.md`](../../../docs/troubleshooting.md) or the [published troubleshooting guide](https://nvidia-nemo.github.io/Anonymizer/troubleshooting/). Apply the suggested setting change to the script, then re-preview.
   - Loop until the user is satisfied.

8. **Finalize** — Tell the user they can run the full pipeline with:
   - `python <script>.py --full`
   - Caution that wall-clock and cost depend heavily on dataset size, model choice, and (for Rewrite) `max_repair_iterations`. Suggest they start by running on a larger sample (e.g. `--num-records 100`) before committing to the full dataset.
   - Remind them: Anonymizer is best-effort. Rewrite outputs with `needs_human_review=True` should be reviewed.
   - Do not run `--full` yourself unless the user explicitly asks. The user controls when expensive work runs.
