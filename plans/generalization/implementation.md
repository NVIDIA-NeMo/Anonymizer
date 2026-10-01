<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Generalization suggestions

Add a distinct document-level generalization workflow after normalized sensitivity dispositions and filtered synthetic replacements. Produce suggestions only for entities assigned generalize, validate exact ID coverage and dependencies, and pass concrete wording and contextual instructions to rewrite and repair. Reuse the rewriter model selection to preserve existing configurations.

Keep raw and validated outputs separate. Skip generation for empty target lists. Flag unresolved statuses for human review; no-effective-generalization entries use conservative omission guidance rather than a fabricated suggestion. Context changes remain instructions rather than automatic substitutions. Generic information, synonyms, cross-entity disclosure, grammar, and temporal consistency are addressed by the prompt.

Validate schemas, target extraction, filtered-map dependencies, empty-target execution, and real-scheduler propagation without remote calls. Exercise both rewrite orchestrators and prompt sandbox validation. Run engine tests, lint, formatting, and type checking. Live quality evaluation remains a separate dataset run. No issue number has been supplied for this local branch.

## Validation

Implemented the suggestion schema, filtered-map dependency, empty-target skip, rewrite/repair guidance, and review flags in both orchestrators. Real-scheduler coverage also exposed nested replacement-map arrays being parsed as empty; normalize them before parsing. Engine/display tests: 810 passed. Type checks passed. Live evaluation findings are recorded below.

## Resume notes

The prompt was shortened by about 27% around three acceptance checks: information reduction, joint protection, and usable wording. The output schema and pipeline remain unchanged. Focused generalization and prompt-template tests passed (12 tests).

Five synthetic biography records were evaluated in runs 6 and 8. Run 6 produced 89 ready suggestions and no contextual instructions. Run 8 produced 87 ready suggestions, one needs_context_change, and no no_effective_generalization results. Its 37 nonempty instructions mostly repeated substitutions. Target coverage was complete, but synonyms, protected geographic anchors, latent-inference conflicts, and grammar problems persisted. The age/birth-date conflict was detected once but resolved by changing the original age rather than reporting an incompatible replacement. Repair still bears much of the burden.

Current implementation is an experimental checkpoint, not a demonstrated quality improvement. Dispositions changed in all five records across runs, preventing a controlled prompt comparison. Proposed next work, not implemented or approved yet: freeze documents/dispositions/replacement maps for evaluation; consider an independent review of the complete suggestion set; clarify conflicts between faithful abstraction and synthetic replacements. A separate evaluator issue can miss semantic disclosure (for example, a publishing company still reveals a publishing/media industry).

Local evaluation files remain outside the repository under ~/Documents/rewrite_results_new/, named rewrite_synth_bio5_sensdisp_branch_eval_run6.csv and rewrite_synth_bio5_sensdisp_branch_eval_run8.csv.
