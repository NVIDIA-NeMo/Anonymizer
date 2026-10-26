<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# GLiNER2 notebook word windows and source-safe spans

## Goal and scope

Fix #329: GLiNER2 appends terminal punctuation to character chunks, and predictions can include that synthetic suffix. Blindly remapping these spans causes the source integrity check to reject entire records. Keep that check. Change only the notebook/development GLiNER2 backend and document its chunk units; keep the checkpoint, threshold, external request field names, and existing overlap policies.

## Implementation and trade-offs

- Build overlapping windows using the loaded model processor's native word splitter and its original character offsets. `chunk_length` and `overlap` count native word tokens in this bundled server, with the existing defaults 384/128. Do not reinterpret these fields for independently hosted services.
- Retain batched raw chunk inference and the existing document overlap resolver. Do not call the native long-document merger: its source-slice rewrite can hide invalid raw span boundaries. Window construction remains small; tokenization comes from the model rather than a second tokenizer.
- Before document remapping, validate each span against its original chunk. Derive admitted text from source offsets. If a prediction demonstrably includes only a trailing period suffix added outside the chunk, trim only that synthetic suffix and clamp its end to the original chunk. Ignore predictions wholly inside that synthetic suffix because they contain no source text. Reject unexplained mismatches or invalid bounds with value-free errors; never fuzzy-match or shift coordinates.
- Preserve overlapping occurrences and resolve duplicate predictions after normalization with the existing longest/disallow policies.
- Keep the bundled server's helpers importable without the isolated GLiNER2 dependency; obtain the word splitter from the loaded model processor.

## Validation

Use fabricated fixtures for regression tests: complete words at window boundaries, preserved whitespace and Unicode offsets, overlapping duplicate predictions, final-source entities with synthetic punctuation, synthetic-only predictions, empty sources, multiple documents, and malformed spans that must still fail. Mock only the external model/splitter boundary. Test the server response contract. Run notebook unit tests, full tests, and `make check`; build docs in strict mode. Replay the approved demonstration fixtures with real inference, checking every admitted span against the source. Detection recall requires separate evaluation; span validity alone does not establish coverage. Keep fixture outputs and model caches out of commits.

## Rollout

Commit with a conventional message and DCO signoff after repository hooks pass. Review the plan and implementation together. Ship the server correction before rerunning release-candidate notebook validation. Legal rewrite-provider timeouts remain a separate issue. No release tag or package publication is part of this fix.


## Verified results

- `make check`: passed formatting, lint, type checking, lock consistency, and SPDX checks.
- `make test`: 1,571 passed after the review follow-up, with three dependency deprecation warnings.
- Strict MkDocs build: passed, writing generated output outside the checkout.
- Live endpoint replay using the model's actual native word splitter: all 11 affected fixtures (two biographies and nine public legal records) returned source-valid spans; one synthetic suffix prediction was corrected. Only aggregate results are recorded here.
- [Notebook CI run 40271396191](https://github.com/NVIDIA-NeMo/Anonymizer/actions/runs/40271396191) on commit `b27e39e3` completed CPU runtime installation and notebooks 01–03. Notebook 04 detection completed across all 25 records with zero detection failures; provider timeouts then failed all eight repair records and triggered the missing-Parquet crash tracked in [#331](https://github.com/NVIDIA-NeMo/Anonymizer/issues/331). Notebook 05 did not execute, and artifact upload was skipped. The full five-notebook run has not passed; detection recall remains unevaluated.
- Local legal verification on commit `b27e39e3`: all 25 records passed through the actual notebook-local server on CPU, using the pinned checkpoint, notebook 05 labels, threshold 0.3, and word windows 384/128. All 1,717 returned spans had valid bounds and exact source text; the engine parser passed and retained 1,578 spans after overlap resolution. This includes all nine previously failing legal records. LLM validation, augmentation, and rewriting were not part of this local check.


## CI setup follow-up

The first branch notebook run failed before inference with a disk quota error. Compared with the successful baseline, the runner changed from Ubuntu 24.04 to 26.04 and restored an old isolated runtime that could not be reused because its internal identity contains platform details. Keep `ubuntu-latest`, but cache only Hugging Face model files under a new model-only cache key. Rebuild the isolated package environment on each runner, then reuse it across tutorials within the job.

Constrain pip installs in this CPU-only CI job to `torch==2.14.1+cpu`, matching the previously resolved PyTorch release. Discover that wheel through the official PyTorch CPU torch listing only; continue resolving other packages from the normal package index. Disable pip's download cache to avoid retaining a second copy of large wheels. Apply these settings to the notebook execution step so its nested isolated-runtime pip install inherits them. Local notebook installation and GPU selection remain unchanged.

Validate workflow structure, resolve the Linux Python 3.11 dependency graph and confirm it contains CPU PyTorch without CUDA packages, run repository checks and strict docs, then dispatch a new notebook run on the updated PR. The full notebook run remains the Linux execution check.

Follow-up validation: Linux x86_64/Python 3.11 resolution passed with 43 packages, `torch==2.14.1+cpu`, and no NVIDIA CUDA or Triton packages. Workflow YAML validation, `make check`, and the strict documentation build passed. The subsequent notebook run confirmed CPU runtime installation succeeded without the earlier disk-quota failure, before encountering the separate rewrite-repair failure described above.


## Review follow-up

Reject empty or whitespace-only model entity text before source-span comparison, even when its offsets point to whitespace. Add synthetic regressions for both cases; both reproduced the missing rejection before the guard was added. This is defensive validation of malformed responses, separate from the real-model span validation above.
