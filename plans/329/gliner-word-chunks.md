<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# GLiNER2 notebook word windows and source-safe spans

## Goal and scope

Fix #329: GLiNER2 appends terminal punctuation to character chunks, and predictions can include that synthetic suffix. Blindly remapping these spans causes the source integrity check to reject entire records. Keep that check. Change only the notebook/development GLiNER2 backend and document its chunk units; keep the checkpoint, threshold, external request field names, and existing overlap policies.

## Implementation and trade-offs

- Build overlapping windows using the loaded model processor's native word splitter and its original character offsets. `chunk_length` and `overlap` count native word tokens in this bundled server, with the existing defaults 384/128. Do not reinterpret these fields for independently hosted services.
- Retain batched raw chunk inference and the existing document overlap resolver. Do not call the native long-document merger: its source-slice rewrite can hide invalid raw span boundaries. Window construction remains small; tokenization comes from the model rather than a second tokenizer.
- Before document remapping, validate each span against its original chunk. Derive admitted text from source offsets. If a prediction demonstrably includes only a single trailing period added outside the chunk, trim only that synthetic suffix and clamp its end to the original chunk. Ignore predictions wholly inside that synthetic suffix because they contain no source text. Reject unexplained mismatches or invalid bounds with value-free errors; never fuzzy-match or shift coordinates.
- Preserve overlapping occurrences and resolve duplicate predictions after normalization with the existing longest/disallow policies.
- Keep the bundled server's helpers importable without the isolated GLiNER2 dependency; obtain the word splitter from the loaded model processor.

## Validation

Use fabricated fixtures for regression tests: complete words at window boundaries, preserved whitespace and Unicode offsets, overlapping duplicate predictions, final-source entities with synthetic punctuation, synthetic-only predictions, empty sources, multiple documents, and malformed spans that must still fail. Mock only the external model/splitter boundary. Test the server response contract. Run notebook unit tests, full tests, and `make check`; build docs in strict mode. Replay the approved demonstration fixtures with real inference, checking every admitted span against the source. Detection recall requires separate evaluation; span validity alone does not establish coverage. Keep fixture outputs and model caches out of commits.

## Rollout

Commit with a conventional message and DCO signoff after repository hooks pass. Review the plan and implementation together. Ship the server correction before rerunning release-candidate notebook validation. Legal rewrite-provider timeouts remain a separate issue. No release tag or package publication is part of this fix.


## Verified results

- `make check`: passed formatting, lint, type checking, lock consistency, and SPDX checks.
- `make test`: 1,575 passed after the exact-contract review follow-up, with three dependency deprecation warnings.
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


## Exact-contract review follow-up

The pinned GLiNER2 2.0 implementation returns case-preserving slices at word offsets and appends exactly one period when input lacks terminal punctuation. Tighten chunk grounding to exact equality and that single-period case; reject case changes, whitespace-normalized values, extra synthetic periods, and invalid bounds. Retain blank-text rejection and synthetic-only skipping. Replace the accepted double-period test with a URL-at-window-end regression and add explicit rejection cases. The earlier external endpoint replay is historical evidence of span repair, not evidence that extra periods belong to the pinned library contract.

Document that native word counts do not bound subword length, particularly for unspaced Chinese/Japanese runs. Compare original character windows and revised word windows with the same local pinned model, legal labels, threshold, and overlap policy on the 16 legal records that previously passed. Save only per-record aggregate counts and source-validity measures; compare word-window sizes if a large drop appears. Counts are a regression screen, not labeled precision or recall. Rerun the full 25-record source-span check after tightening, along with unit tests, repository checks, and strict docs.


### Paired legal-window comparison

Same pinned local model on CPU, legal notebook labels, threshold 0.3, `flat_ner=false`, and the existing overlap resolver. Character windows use the unmodified backend at `1872309`; word windows use the tightened implementation. All 16 records below were previously successful in CI, and their character-window predictions were source-valid in this local replay. These are proposal counts after document overlap resolution, before LLM validation or augmentation; they are not ground-truth recall.

| Record index (zero-based) | Character 384/128 | Word 384/128 | Word 128/32 | Word 64/16 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 78 | 48 | 70 | 77 |
| 1 | 97 | 68 | 84 | 90 |
| 2 | 102 | 82 | 97 | 111 |
| 4 | 70 | 51 | 62 | 65 |
| 5 | 63 | 48 | 62 | 66 |
| 6 | 91 | 64 | 68 | 80 |
| 7 | 94 | 79 | 85 | 93 |
| 8 | 89 | 60 | 80 | 81 |
| 13 | 81 | 59 | 76 | 85 |
| 14 | 102 | 80 | 97 | 111 |
| 15 | 131 | 112 | 120 | 132 |
| 16 | 62 | 45 | 58 | 64 |
| 18 | 66 | 49 | 61 | 64 |
| 21 | 134 | 89 | 128 | 136 |
| 23 | 98 | 67 | 85 | 94 |
| 24 | 102 | 82 | 97 | 111 |
| **Total** | 1460 | 1083 | 1330 | 1460 |

The 384-word setting produced 25.8% fewer proposals than the character baseline (1,083 vs 1,460); 960 exact offset/label proposals were shared, 500 were character-only, and 123 were word-only. Shorter word windows recovered proposal counts, but counts alone cannot distinguish recovered identifiers from false positives. The default remains unchanged pending a detection-quality decision; do not interpret the source-validity checks as clearance of this quality concern. The experiment does not establish whether the encoder context length caused the differences.

All 25 legal records also passed the tightened 384-word grounding implementation with zero source-span mismatches (1,717 proposals in total). Shorter settings had no span mismatches on the 16-record comparison. Full tests, repository checks, and strict docs passed after the tightening change. No source text, entity values, or model outputs beyond aggregate counts are committed.
