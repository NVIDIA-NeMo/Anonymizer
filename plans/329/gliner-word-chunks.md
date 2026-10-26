<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# GLiNER2 notebook word windows and source-safe spans

## Goal and scope

Fix #329: GLiNER2 appends terminal punctuation to character chunks, and predictions can include that synthetic suffix. Blindly remapping these spans causes the source integrity check to reject entire records. Keep that check. Update the notebook/development GLiNER2 backend, notebook model settings, and engine chunk-parameter pass-through, and document the chunk units. Keep the checkpoint, threshold, external request field names, and existing overlap policies. Leave chunk fields unset in the shipped localhost model entry so it uses the bundled server's measured defaults.

## Implementation and trade-offs

- Build overlapping windows using the loaded model processor's native word splitter and its original character offsets. `chunk_length` and `overlap` count native word tokens in this bundled server, initially with 384/128; the labeled evaluation below changes the bundled defaults to 128/32. Do not reinterpret these fields for independently hosted services.
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
- `make test`: 1,579 passed after the measured-default and engine pass-through update, with three dependency deprecation warnings.
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

The 384-word setting produced 25.8% fewer proposals than the character baseline (1,083 vs 1,460); 960 exact offset/label proposals were shared, 500 were character-only, and 123 were word-only. Shorter word windows recovered proposal counts, but counts alone cannot distinguish recovered identifiers from false positives. This initial count screen prompted the labeled evaluation and default update documented below; source-validity checks alone did not clear the quality concern. The experiment does not establish whether the encoder context length caused the differences.

All 25 legal records also passed the tightened 384-word grounding implementation with zero source-span mismatches (1,717 proposals in total). Shorter settings had no span mismatches on the 16-record comparison. Full tests, repository checks, and strict docs passed after the tightening change. No source text, entity values, or model outputs beyond aggregate counts are committed.


## Measured default update

Use native word windows 128/32 for the bundled notebook/development server. Share these defaults between server parsing and the notebook-owned detector model configuration. Remove the engine's hardcoded chunk assignments so configured `extra_body.chunk_length` and `extra_body.overlap` pass through and omitted settings defer to the server. Continue injecting effective labels and threshold from detection config. Omit chunk settings in the shipped `models.yaml` detector entry, which targets the bundled localhost server. Plain `Anonymizer()` therefore receives the measured 128/32 server defaults, as does the notebook helper through its explicit parameters. Custom model pools without chunk fields defer to their server's defaults and units. No new `Detect` fields or top-level public symbols are needed; the bundled skill template does not change.

Verify notebook configuration through engine parameter injection and through the server request contract, retain regressions for the shipped model pool omitting chunk fields and explicit custom values passing through, and rerun all unit tests, repository checks, and strict docs. Run the actual notebook-local server with the newly injected request parameters on all 25 legal CSV records. Post the measured tradeoff in the review thread after the change is pushed.

### Labeled evaluation supporting 128/32

All 25 CSV records matched original [TAB annotations](https://github.com/NorskRegnesentral/text-anonymization-benchmark) exactly; one record is duplicated, leaving 24 unique training documents. A separate cohort of 50 development documents was selected before inference from 69 eligible documents of 2,000–6,000 characters, excluding fixture text, sorted by SHA-256 of text. The same pinned checkpoint, CPU, notebook 05 labels, threshold 0.3, batch size 8, and overlap resolver were used throughout. No LLM stages were run.

The evaluation uses untyped letter/digit position coverage so split first/last-name predictions can jointly cover a full PERSON annotation. Precision compares predicted positions with all annotated identifiers, including NO_MASK; sensitive-text coverage compares with DIRECT/QUASI positions. Scores micro-average annotation layers. These are diagnostic coverage metrics, not official TAB weighted/entity scores. Invalid character-baseline predictions are excluded from quality scoring and counted separately; the production parser can fail their entire record.

| Setting | Sensitive-text coverage (50 dev docs) | Annotation precision | Complete sensitive-mention coverage | Records with invalid spans |
| --- | ---: | ---: | ---: | ---: |
| Character 384/128 | 91.73% | 55.67% | 82.07% | 25/50 |
| Word 384/128 | 87.65% | 65.34% | 80.64% | 0/50 |
| Word 128/32 | 91.12% | 58.07% | 81.66% | 0/50 |
| Word 64/16 | 91.06% | 55.41% | 81.73% | 0/50 |

128/32 improves sensitive-text coverage by 3.47 percentage points over 384-word windows (paired document bootstrap, 2,000 samples, seed 330: 95% interval +2.19 to +5.00 points), with 7.27 points lower annotation precision. Compared with the previous character implementation, it trades 0.61 points of coverage for 2.40 points of precision and avoids the observed source-span failures. PERSON character coverage is 79.99% at 128/32 versus 63.47% at 384 words and 82.68% at 384 characters. At 64 words, coverage is essentially the same as 128 words, but precision is lower and more candidates require validation. This supports 128/32 as a compromise for the English notebook dataset, not a universal optimum.

All three word settings had zero inference errors and zero invalid source spans across 74 unique documents. On the 24 unique notebook fixtures, sensitive-text coverage/precision was 93.11%/61.81% at 128/32, versus 93.74%/59.56% for character windows. Source data, gold annotations, and entity values remain outside the repository; only aggregate results are recorded here.


Final default validation: 1,579 unit tests, `make check`, and strict documentation build passed. The actual notebook-local server was exercised on all 25 legal CSV records using parameters obtained from the notebook model builder through engine injection: 128/32, threshold 0.3. All 2,097 returned spans had exact source text and valid bounds; the engine parser passed (1,928 spans after overlap resolution), with zero record failures. The server was stopped after the check. Tests cover the notebook route, server defaults and explicit request overrides, omission of unconfigured engine chunk fields, preservation of configured chunk fields, and the notebook defaults. The later localhost-default correction below removes the shipped pool's 384/128 override.


## Localhost-default review correction

The reviewer correctly identified that the shipped `local-gliner2` provider points at `127.0.0.1:8001`, where the docs start the bundled server. Retaining numeric 384/128 in that model entry preserved request numbers but changed their meaning to words and defeated the measured 128/32 default. Remove that override. Both plain `Anonymizer()` and the notebook helper now obtain 128/32 for the bundled server; explicit custom detector settings still pass through. Include a compatibility note in the imminent release notes explaining that omitted chunk fields now defer to server defaults instead of being overwritten by the engine. The PR description carries that note for release preparation; a standalone changelog is unnecessary because release automation does not consume it.

Update the default-pool regression to assert the outgoing request omits both chunk fields, retain server parsing coverage, and verify the plain default-model-pool request against the real local server. No new model-quality experiment is needed because inference still uses the already-evaluated 128/32 settings.

Validation of the localhost correction: 1,579 tests, `make check`, and strict docs passed. The real local server processed all 25 legal records with the plain shipped model pool passed through engine injection; both chunk fields were absent from the request. All 2,097 returned spans were source-valid, and the engine parser passed (1,928 after overlap resolution). The server was stopped after validation.
