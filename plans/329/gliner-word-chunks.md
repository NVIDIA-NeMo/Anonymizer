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
- `make test`: 1,569 passed, with three dependency deprecation warnings.
- Strict MkDocs build: passed, writing generated output outside the checkout.
- Live endpoint replay using the model's actual native word splitter: all 11 affected fixtures (two biographies and nine public legal records) returned source-valid spans; one synthetic suffix prediction was corrected. Only aggregate results are recorded here.
- Full tutorial execution and detection-recall evaluation have not been repeated for this branch. The separate LLM repair-provider timeouts are not addressed by this change.
