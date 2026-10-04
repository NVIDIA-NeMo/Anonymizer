<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Remove disposition protection reasons

The user approved removing `protection_reason` from sensitivity dispositions. Generated reasons have introduced wording suggestions and conflicting guidance despite prompt instructions. Remove the field from the internal schema, disposition output instructions, normalization, and downstream meaning-unit/rewrite representations. Keep structured sensitivity/method decisions, coverage checks, and latent detection evidence/rationale unchanged. There is no public configuration change.

The trade-off is losing generated per-decision explanations, including justifications for low-sensitivity exceptions. Keep the policy for selecting those exceptions in the prompt; downstream consumers follow the structured method and privacy goal rather than a reason. Existing saved payloads containing the obsolete field remain parseable under Pydantic's existing extra-field behavior, but newly serialized dispositions omit it.

Update fixtures and verify absence in generated schemas and prompt inputs, legacy payload parsing, strict schema conversion, protection normalization, and workflow execution. Run engine/interface tests, lint/format, and blocking type checks. No live quality evaluation, commit, push, or stacked-branch migration is included in this change.

## Generic-value policy clarification

The user approved replacing the low-sensitivity prompt section with the exact wording reviewed in conversation. Add positive generic examples and contrasting contextual references; clarify that entity labels and imagined identification paths do not establish specificity. Keep context-dependent assessment, the medium uncertainty default, direct-identifier protection, and strict-mode behavior. No deterministic proper-noun/number heuristic is introduced. Run disposition and prompt-template tests plus formatting checks; controlled model evaluation remains needed to establish quality effects.

## Contextually safe information

The user subsequently approved broadening low eligibility beyond generic wording. Replace the low section with the reviewed contextual linkage/recognition criteria, separately honoring goals that require concealing an attribute itself. Concrete information may be low when it contributes no meaningful re-identification risk in the complete supplied document. Keep combination assessment, the medium uncertainty default, and direct-identifier protection. Clarify that strict mode protects every entity regardless of its contribution to re-identification risk. This supersedes the generic-only eligibility above. The trade-off is greater model judgment and possible variability; live controlled evaluation remains outstanding.

## Diagnostic low-sensitivity explanations

The user approved `low_sensitivity_reason`, required and nonblank for retained low entries, null for medium/high entries. The disposition prompt asks for contextual linkage/recognition analysis, consistency with attribute-concealment goals and other protected entities, and prohibits replacement/generalization wording or edit instructions. Keep explanations only in disposition diagnostics; existing explicit-field serializers exclude them from meaning-unit and writer inputs. Protection normalization clears obsolete low reasons when promoting to medium/high. Historical low payloads without explanations now fail validation rather than receiving invented justifications. Schema validation checks presence, not semantic correctness; live evaluation must assess explanation quality.

## Consolidated sensitivity prompt

The user approved the complete consolidated template: direct identifiers remain high, generic quasi-identifiers start low, specific quasi-identifiers and latent inferences start medium, and context/privacy-goal requirements can override these presumptions. Replace repeated guidance with the ordered decision hierarchy and retain the low explanation output contract. Strict mode overrides low eligibility and protects every entity. This supersedes the earlier medium default for all quasi-identifiers. Validate prompt rendering and strict-mode isolation with existing tests; model quality remains a separate controlled evaluation.

## Explicit generic/specific decision rules

The user approved replacing the sensitivity policy with separate direct, generic quasi-identifier, specific quasi-identifier, and latent-inference rules. Generic quasi-identifiers are low unless document-supported distinguishing context establishes meaningful linkage/recognition contribution; specific values and latent inferences default medium unless context supports low. Define specific values using named references, concrete attributes, and precise facts, with proper nouns as an indicator rather than a hard rule. Shorten low explanations to generic/specific contextual reasoning and consistency checks. Retain the scope, methods, strict override, and schema unchanged. This supersedes the numbered consolidated policy.
