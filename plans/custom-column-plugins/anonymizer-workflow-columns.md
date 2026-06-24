<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Anonymizer Workflow Columns

## Context

Anonymizer currently uses DataDesigner `CustomColumnConfig` for private pipeline steps. This works in-process, but custom functions serialize poorly and complicate distributed execution. The goal is to make Anonymizer workflows portable when `nemo-anonymizer` is installed, without exposing implementation details as user-facing API.

## Goal

Remove `CustomColumnConfig` from production Anonymizer workflow builders and replace it with Anonymizer-owned plugin column configs that DataDesigner can discover through entry points.

## Naming

Use `anonymizer.engine.workflow_columns` for the implementation package.

Rationale:

- It describes the Anonymizer concept, not the host engine.
- It avoids names like `dd_plugins`.
- It keeps these classes internal to the Anonymizer engine while still allowing DataDesigner to load them by entry point.

Entry point names should use the Anonymizer domain:

```toml
[project.entry-points."data_designer.plugins"]
anonymizer-detection-transform = "anonymizer.engine.workflow_columns.detection.plugins:detection_transform_plugin"
anonymizer-chunked-validation = "anonymizer.engine.workflow_columns.detection.plugins:chunked_validation_plugin"
```

## Design

Use one installable package, `nemo-anonymizer`, with multiple DataDesigner plugin entry points. Each entry point exposes one `Plugin` object. This gives us one package with multiple submodules, rather than one distribution per plugin.

Do not create a 1:1 plugin for every current helper function unless it is clearer. Prefer a small set of role-based column types:

- Detection transforms for parse, validation prep, merge, and finalize operations.
- Chunked validation for model-backed validation.
- Rewrite transforms for prompt prep, extraction, metrics, and postprocess operations.
- Rewrite model-call columns for evaluator and repair calls that need parser context or dynamic model aliases.

Keep the existing pure helper functions as implementation cores. The plugin generators should adapt DataDesigner row input to those helpers, not rewrite the business logic.

## Native NER Transport

Moving custom columns into plugins does not by itself remove the need for a chat-completions-compatible NER endpoint. The current detector remains an `LLMTextColumnConfig`, so DataDesigner routes it through chat completion generation.

To remove the extra chat completions head on NER models, the detector call itself must become an Anonymizer workflow column that talks to a native NER endpoint or client. That is a separate design slice because it needs a stable endpoint contract, provider configuration, and response normalization.

## Stage 1: Detection Plugins

Move the detection workflow off custom columns first.

Tasks:

1. Add `src/anonymizer/engine/workflow_columns/`.
2. Add detection plugin configs and generators:
   - `DetectionTransformConfig`
   - `ChunkedValidationConfig`
3. Add DataDesigner entry points in `pyproject.toml`.
4. Replace detection `CustomColumnConfig` usage in `EntityDetectionWorkflow`.
5. Keep `chunked_validate_row(...)` as the core validator implementation.
6. Add a serialization test that builds the detection workflow and confirms config JSON contains no serialized function names.

Validation:

```bash
.venv/bin/ruff check src/anonymizer/engine/detection src/anonymizer/engine/workflow_columns tests/engine/test_detection_workflow.py tests/engine/test_chunked_validation.py
.venv/bin/pytest tests/engine/test_detection_postprocess.py tests/engine/test_chunked_validation.py tests/engine/test_detection_workflow.py
```

Exit criteria:

- Detection behavior is unchanged.
- Detection builder configs serialize without custom generator functions.
- No production detection workflow path instantiates `CustomColumnConfig`.

## Stage 2: Rewrite Plugins

Move rewrite, evaluation, repair, and final judge postprocess columns.

Tasks:

1. Add rewrite plugin configs and generators:
   - `RewriteTransformConfig`
   - `RewriteEvaluatorCallConfig`
   - `RewriteRepairCallConfig`
   - `FinalJudgeTransformConfig`
2. Replace custom columns in:
   - `domain_classification.py`
   - `qa_generation.py`
   - `rewrite_generation.py`
   - `evaluate.py`
   - `repair.py`
   - `final_judge.py`
3. Preserve current parser-context behavior in evaluator columns.
4. Add tests that workflow column lists contain plugin configs, not `CustomColumnConfig`.

Validation:

```bash
.venv/bin/ruff check src/anonymizer/engine/rewrite src/anonymizer/engine/workflow_columns tests/engine/test_rewrite_workflow.py
.venv/bin/pytest tests/engine/test_domain_classification.py tests/engine/test_qa_generation.py tests/engine/test_rewrite_generation.py tests/engine/test_evaluate.py tests/engine/test_repair.py tests/engine/test_final_judge.py tests/engine/test_rewrite_workflow.py
```

Exit criteria:

- Rewrite behavior is unchanged.
- Repair loop still runs only on failing rows.
- Evaluator calls still enforce expected QA IDs.
- Rewrite builder configs serialize without custom generator functions.

## Stage 3: Distribution Guardrails

Make the new path hard to regress.

Tasks:

1. Add a test that production code under `src/anonymizer/engine` no longer creates `CustomColumnConfig`.
2. Add export tests for distributed builder factories.
3. Update any docs or comments that describe custom columns as the distributed limitation.
4. Bump the DataDesigner dependency only when the needed plugin and serialization behavior is available in the supported release.
5. Consider a native detector workflow column if we want NER models to avoid chat-completions-compatible serving.

Validation:

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pytest tests/engine tests/interface
```

Exit criteria:

- Runtime Anonymizer workflows use Anonymizer plugin columns only.
- Serialized configs are portable across machines where `nemo-anonymizer` is installed.
- Tests prevent reintroducing custom columns into production workflow builders.

## Open Questions

1. Should plugin configs live under `anonymizer.engine.workflow_columns` or a shorter `anonymizer.workflow_columns` namespace?
2. Should `DetectionTransformConfig` use an operation enum, or should parse, prep, merge, and finalize each get a distinct config class?
3. Which DataDesigner release should become the minimum supported version for this migration?
4. Is native NER transport in scope for this migration, or should it be a separate provider/client project?
