<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Compact sensitivity-disposition output

## Goal

Reduce long disposition responses by asking the model for decisions only. Reconstruct
the existing full disposition from stable IDs and the authoritative entity inventory.
This experiment is confined to the integration branch; no public configuration changes.

## Implementation

- Add compact raw schemas containing ID, sensitivity, method, and low-sensitivity reason,
  including a strict variant that excludes low sensitivity and leave-as-is.
- Update the prompt to request only those fields, retaining the full document context,
  inventory, classification table, and existing protection policy.
- Validate sequential IDs and complete coverage before reconstructing source, category,
  label, and value. Run existing consistency checks and normalization on the full result.
- Keep the raw trace compact and the normalized disposition unchanged for privacy QA,
  rewriting, repair, and both workflow execution paths.

## Tradeoffs and rollout

This reduces repeated identity text without changing protection decisions. It does not
retry incomplete responses, batch inventories, or guarantee that omissions disappear.
Incomplete coverage intentionally fails the record rather than continuing with partial
protection decisions or returning a rewrite with `needs_human_review`. The adapter
surfaces dropped rows as failed records; frequent failures can reduce batch output or
trigger DataDesigner's early shutdown. This change adds no retry or flagged fallback.

Sequential IDs and matching counts establish structural coverage, not semantic alignment.
A model can attach a decision intended for another entity to a valid ID; reconstructing
identity cannot detect that mistake. Removing echoed labels and values also removes a
diagnostic signal that could expose duplicated or mismatched identities in full output.

Old full raw payloads are no longer the model-response contract; normalized output stays
compatible. Re-evaluate the same agent traces before porting to the disposition branch.

## Validation

Test compact and strict response schemas, ID and count failures, missing categories,
exact Unicode identity reconstruction, sensitivity/method normalization, and low-reason
validation. Exercise reconstruction and downstream consumers through the real scheduler,
run both workflow suites, compare output size on the supplied trace export, and run the
repository checks.

## Local results

- Full suite: 1,481 tests passed; repository checks passed.
- Replaying all 608 decisions from the eight exported agent traces reconstructed
  exactly the existing normalized dispositions.
- With the same JSON serialization, raw payload size fell from 144,748 to 75,289
  characters (48% smaller). This measures output size, not model token usage.
- A fresh provider-backed run is still needed to measure omission rates and decision quality.

## Review evidence and follow-up

The reviewer reports omissions in 2 of 8 runs at `max_tokens=16384` and short
responses in 9 of 10 recorded dispositions at `max_tokens=4096`, including responses
well below the output ceiling. These observations motivate coverage enforcement;
they do not establish omission rates for fresh compact-format generations.

Synthetic regression tests cover incomplete counts, acceptance and reconstruction of
valid equal-count decisions, and failed-record propagation through the real scheduler
and adapter. The reviewer's exact inventory/decision fixtures have not been supplied
locally; add sanitized fixtures when available, without document text or real PII.
A fresh provider-backed comparison remains outstanding.
