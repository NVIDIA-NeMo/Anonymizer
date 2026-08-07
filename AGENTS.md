<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# AGENTS.md

This file is for agents **developing** NeMo Anonymizer — the codebase you are working in.
If you are an agent helping a user **anonymize data**, use the [product documentation](https://nvidia-nemo.github.io/Anonymizer/) or the bundled agent skill at [`skills/anonymizer/`](skills/anonymizer/SKILL.md) instead.

**NeMo Anonymizer** detects and protects PII through context-aware entity replacement and LLM-powered rewriting. Users supply a text dataset and a strategy; Anonymizer detects entities and transforms the text.

## Agent compatibility

`AGENTS.md` is the canonical instruction file for coding agents working in this repository. Keep it tool-neutral:

- Use plain Markdown and repository-relative links.
- Do not rely on vendor-specific include syntax, slash commands, MCP names, or IDE-only behavior.
- Put tool-specific adapter instructions in thin wrapper files such as `CLAUDE.md`.

## Module Map

`nemo-anonymizer` is a single package with three primary subpackages plus top-level public utilities:

- **`anonymizer.config`** — user-facing configuration: `AnonymizerConfig`, `AnonymizerInput`, replace strategies (`Substitute`, `Redact`, `Annotate`, `Hash`), and rewrite config (`Rewrite`, `EvaluationCriteria`, `RiskTolerance`). New user-facing knobs go here.
- **`anonymizer.engine`** — internal pipeline implementation: detection, replacement, and rewrite sub-workflows, the NDD adapter, prompt utilities, and all `COL_*` column constants. Never imported directly by users.
- **`anonymizer.interface`** — user-facing entry points: the `Anonymizer` class, CLI, `AnonymizerResult`, `PreviewResult`, and canonical error types. Thin layer that wires config → engine and exposes results.
- **`anonymizer.logging`** — public logging configuration (`LoggingConfig`, `configure_logging`) used by the API, CLI, and examples.

NeMo Anonymizer wraps [DataDesigner](https://github.com/NVIDIA-NeMo/DataDesigner) (NDD) for LLM column generation. `NddAdapter.run_workflow()` is the engine boundary for *executing* DataDesigner workflows — engine sub-workflows may declare DataDesigner column configs (e.g. `LLMStructuredColumnConfig`), but they do not call `DataDesigner.create()` or `preview()` directly.

Public-API changes (re-exports, signatures, kwargs, default values) may require a matching update to the bundled agent skill at [`skills/anonymizer/SKILL.md`](skills/anonymizer/SKILL.md), whose output template imports and instantiates these symbols. Check the template before shipping a change to anything in `anonymizer/__init__.py` or the public surface of `Detect`, `Substitute`, `Redact`, `Annotate`, `Hash`, `Rewrite`, `PrivacyGoal`, or `AnonymizerInput`.

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
  → output: {text_col}_rewritten, utility_score, leakage_mass, needs_human_review, …

Later, `Anonymizer.evaluate()` runs the non-critical judge steps:
  → EntityCoverageWorkflow                  → entity_coverage, leaked_entities   (always)
  → FinalJudgeWorkflow                      → judge_evaluation                    (always)
  → DetectionJudgeWorkflow                  → detection_valid, detection_invalid_entities
                                              (only when EvaluateConfig.compute_detection_validity=True)
```

Records with no detected entities skip all LLM sub-workflows and pass through with default metrics (utility=1.0, leakage=0.0).

## Config Pattern

`AnonymizerConfig.rewrite` is the user-facing `Rewrite` model. The engine never receives `Rewrite` directly — it receives `EvaluationCriteria` via the `Rewrite.evaluation` property. See that property's docstring for the sync contract (how `risk_tolerance` and `max_repair_iterations` flow into the engine, why production code should not duplicate the mapping).

## NDD Adapter

`NddAdapter.run_workflow()` (`engine/ndd/adapter.py`) is the engine boundary for *executing* DataDesigner workflows. See its docstring for the contract (input/output shapes, `FailedRecord` semantics).

## Prompt Conventions

NDD prompts are inline triple-quoted strings in the workflow file that uses them; there is no separate registry. For DataFrame column references inside templates, use `_jinja()`; for dynamic prompt values, use `substitute_placeholders()`. See each function's docstring for details.

## Structural Invariants

Code conventions enforced in review (future-annotations import, absolute imports, type annotations, SPDX headers, column-name constants) live in [STYLEGUIDE.md](STYLEGUIDE.md).

One pipeline-specific fact worth knowing: `COL_TEXT` is the internal name for the input text column; it's renamed to the user's original column name in final output.

## What NOT To Do

- **Don't duplicate the `Rewrite` → `EvaluationCriteria` mapping** when production code starts from a `Rewrite`; route it through `Rewrite.evaluation`.
- **Don't execute DataDesigner workflows directly** — call `DataDesigner.create()` / `.preview()` only via `NddAdapter.run_workflow()`. Declaring column configs (`LLMStructuredColumnConfig`, etc.) is fine.
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

For contributor workflow and branch naming see [CONTRIBUTING.md](CONTRIBUTING.md). For local setup, tests, docs, and day-to-day development tasks see [DEVELOPMENT.md](DEVELOPMENT.md). For code style and naming conventions see [STYLEGUIDE.md](STYLEGUIDE.md).

Reference GLiNER server for self-hosted detection: [`tools/serve_gliner.py`](tools/serve_gliner.py) — see [`docs/concepts/self-hosting-gliner.md`](docs/concepts/self-hosting-gliner.md).
