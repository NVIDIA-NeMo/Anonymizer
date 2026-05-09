# AGENTS.md

This file is for agents **developing** NeMo Anonymizer — the codebase you are working in.
If you are an agent helping a user **anonymize data**, use the [product documentation](https://nvidia-nemo.github.io/Anonymizer/) instead.

**NeMo Anonymizer** detects and protects PII through context-aware entity replacement and LLM-powered rewriting. Users supply a text dataset and a strategy; Anonymizer detects entities and transforms the text.

## Module Map

`nemo-anonymizer` is a single package with three modules:

- **`anonymizer.config`** — user-facing configuration: `AnonymizerConfig`, `AnonymizerInput`, replace strategies (`Substitute`, `Redact`, `Annotate`, `Hash`), and rewrite config (`Rewrite`, `EvaluationCriteria`, `RiskTolerance`). New user-facing knobs go here.
- **`anonymizer.engine`** — internal pipeline implementation: detection, replacement, and rewrite sub-workflows, the NDD adapter, prompt utilities, and all `COL_*` column constants. Never imported directly by users.
- **`anonymizer.interface`** — user-facing entry points: the `Anonymizer` class, CLI, `AnonymizerResult`, `PreviewResult`, and canonical error types. Thin layer that wires config → engine and exposes results.

NeMo Anonymizer wraps [DataDesigner](https://github.com/NVIDIA-NeMo/DataDesigner) (NDD) for LLM column generation. `NddAdapter` is the only place this dependency crosses — engine sub-workflows declare NDD column configs and hand them to the adapter, which manages DataDesigner internally.

## Core Concepts

- **Entity** — a detected span of text with a label (e.g. `"Alice"` → `first_name`) and character offsets
- **Latent entity** — an entity detected in rewrite mode that is sensitive but not directly named; used to guide rewriting without explicit replacement
- **Replacement map** — a per-record dict mapping entity text → substitute value, built by `LlmReplaceWorkflow` and injected into rewrite prompts
- **Leakage mass** — a weighted score measuring how much sensitive information survives in a rewritten record; drives the repair loop
- **Utility score** — a 0–1 score measuring how much semantic content the rewritten record preserves
- **RiskTolerance** — a preset (`minimal` / `low` / `moderate` / `high`) that bundles the leakage threshold, repair behaviour, and human-review flags into a single user-facing knob
- **Repair loop** — the evaluate → repair → re-evaluate cycle in `RewriteWorkflow`; runs up to `max_repair_iterations` times on failing rows
- **FailedRecord** — a record that was dropped by an NDD workflow; surfaced explicitly rather than silently lost

## Pipelines

### Replace mode — `AnonymizerConfig(replace=...)`

```
input_df
  → EntityDetectionWorkflow.run()              # engine/detection/detection_workflow.py
        GLiNER detection
        → parse + tag
        → LLM augmentation  (add entities GLiNER missed)
        → LLM validation    (keep / drop candidates)
        → merge + finalize  → COL_DETECTED_ENTITIES, COL_FINAL_ENTITIES
  → ReplacementWorkflow.run()                  # engine/replace/replace_runner.py
        Redact / Annotate / Hash  → applied locally, no LLM
        Substitute                → LlmReplaceWorkflow → NddAdapter
  → output: {text_col}_replaced, {text_col}_with_spans, final_entities
```

### Rewrite mode — `AnonymizerConfig(rewrite=...)`

```
input_df
  → EntityDetectionWorkflow.run()              # same as above, plus latent entity tagging
  → RewriteWorkflow.run()                      # engine/rewrite/rewrite_workflow.py
        LlmReplaceWorkflow.generate_map_only() # build replacement map for prompt
        → single NDD adapter call (pipeline_columns):
              DomainClassificationWorkflow    → _domain, _domain_supplement
              SensitivityDispositionWorkflow  → _sensitivity_disposition
              QAGenerationWorkflow            → _quality_qa, _privacy_qa
              RewriteGenerationWorkflow       → _rewritten_text
        → evaluate-repair loop (up to max_repair_iterations):
              EvaluateWorkflow                → leakage_mass, utility_score, _needs_repair
              RepairWorkflow                  → _rewritten_text (failing rows only)
        → FinalJudgeWorkflow (non-critical)   → _judge_evaluation, needs_human_review
  → output: {text_col}_rewritten, utility_score, leakage_mass, needs_human_review, …
```

Records with no detected entities skip all LLM sub-workflows and pass through with default metrics (utility=1.0, leakage=0.0).

## Config Pattern

`AnonymizerConfig.rewrite` is the user-facing `Rewrite` model. The engine never receives `Rewrite` directly — it receives `EvaluationCriteria` via the `Rewrite.evaluation` property.

`Rewrite` and `EvaluationCriteria` both hold `max_repair_iterations`. They must stay in sync:

- `Rewrite.max_repair_iterations` is the user-facing field (default 3)
- `Rewrite.evaluation` constructs `EvaluationCriteria(risk_tolerance=..., max_repair_iterations=self.max_repair_iterations)`
- **Never construct `EvaluationCriteria` with hardcoded values** — always go through `Rewrite.evaluation`

Leakage thresholds and repair parameters are derived from `RiskTolerance` via `_RiskToleranceBundle` in `config/rewrite.py`. Don't hardcode them elsewhere.

## NDD Adapter

`NddAdapter.run_workflow()` (`engine/ndd/adapter.py`) wraps a DataFrame slice + NDD column configs into a DataDesigner run and returns `WorkflowRunResult(dataframe, failed_records)`. Records missing from the output surface as `FailedRecord` objects rather than silently disappearing. Never access DataDesigner directly from engine workflows — always go through `NddAdapter`.

## Prompt Conventions

All column references in NDD prompt templates go through `_jinja()` (`engine/constants.py`) — never format column names directly into strings. Dynamic prompt values use `substitute_placeholders()` (`engine/prompt_utils.py`) with `<<PLACEHOLDER>>` markers; see its docstring for the substitution contract. Prompts are inline triple-quoted strings in the workflow file that uses them; there is no separate registry.

## Structural Invariants

- `from __future__ import annotations` in every Python file
- Absolute imports only (enforced by ruff `TID`)
- Type annotations on all functions, methods, and class attributes
- SPDX license header on every file
- All column names defined in `engine/constants.py` — never use string literals for column names
- `COL_TEXT` is the internal name for the input text column; renamed to the user's original column name in final output

## What NOT To Do

- **Don't bypass `Rewrite.evaluation`** — don't construct `EvaluationCriteria` with hardcoded thresholds
- **Don't call DataDesigner directly** — always go through `NddAdapter.run_workflow()`
- **Don't use string literals for column names** — use `COL_*` constants from `engine/constants.py`
- **Don't add a domain to only one supplement map** — see `engine/rewrite/domain_classification.py` for the sync invariant
- **Don't hardcode `gliner_threshold`** — it belongs in `Detect` config (default 0.3)

## Development

```bash
make test          # run all tests
make bootstrap     # install dev dependencies
make format        # ruff format + sort imports
make format-check  # read-only lint check (used in CI)
make typecheck     # ty type check (advisory)
make docs-serve    # local MkDocs server at http://127.0.0.1:8000
```

For contributor workflow and branch naming see [CONTRIBUTING.md](CONTRIBUTING.md).
For code style and naming conventions see [STYLEGUIDE.md](STYLEGUIDE.md).
