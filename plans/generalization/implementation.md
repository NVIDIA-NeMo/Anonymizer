<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Generalization suggestions

For the current state and next steps, read [handoff.md](handoff.md). Sections below record implementation history; later decisions supersede earlier ones.

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

## Independent review

Add one fresh rewriter-model call between candidates and canonical validation. Preserve raw candidates and raw reviewed output separately; rewrite and repair consume only validated reviewed suggestions. Skip both model calls for empty targets. Validate coverage after review and derive human-review flags from reviewed statuses. Test actual scheduler propagation with a reviewer that rejects a ready candidate. No automatic review loop or public model configuration changes.

## Review input and evidence revision

Run 9 changed none of 85 suggested values; 14 statuses changed for unrelated gender/spouse concerns. Strip candidates to IDs and wording before review. Require structured defects with quoted evidence and conflicting IDs before corrected suggestions. Validate defect references and retain diagnostics in the raw reviewed output, while canonical suggestions keep the existing downstream schema. Reuse fixed run 9 candidates for the next live comparison rather than attributing full-pipeline score changes to the reviewer alone. Live replay has not been executed.

## Preserve pre-applied replacements

Remove the replacement map from the rewrite prompt and instruct preservation of synthetic values already in its tagged input. Required privacy edits may remove a containing clause, but must not invent another replacement or restore the original. Generalization and review retain the filtered map for coordination.

## Rewrite action inputs

Replace the sensitivity-disposition prompt section with deterministic generalize, remove, and suppress-latent-inference lists in _rewrite_actions. Exclude replace/leave_as_is entries; route no_effective_generalization to explicit removal. Join protected latent entries to numbered evidence and rationale by stable ID, verifying label/value fidelity. Preserve original dispositions internally for filtering/scoring. Rewrite now prioritizes omissions and inference suppression, natural grammar, and already-applied synthetic values. Repair remains unchanged at user request.

## Unambiguous rewrite actions

Remove sensitivity reasons and reviewer justifications from generalization/removal prompt entries. Keep actionable reviewed generalizations and contextual instructions. All removal entries use a deterministic omission instruction; free-form reviewer removal guidance is retained only in _rewrite_action_diagnostics. Every such override is recorded, without claiming deterministic semantic contradiction detection. Latent evidence, rationale, and protection outcomes remain. Repair is unchanged. Tests cover clean and contradictory reviewer removal wording.

## Restore leaked-item-focused repair

At the user’s request, restore repair.py to local main: remove the disposition/generalization prompt block and dependencies. Repair receives the baseline, previous rewrite, privacy goal, and evaluator leakage feedback as on main. Rewrite retains its action lists. This supersedes earlier plans to feed reviewed actions into repair.

## Current-text-only repair

Replace main’s repair prompt with the approved focused editing prompt. Remove baseline/original text and aggregate metric dependencies from the repair generator. Preserve evaluation/threshold logic in the pipeline; the model sees only privacy goal, current rewrite, and leaked-item diagnostics. Prompt explicitly prohibits adding facts from feedback. Validate rendered input isolation and both orchestration paths.
