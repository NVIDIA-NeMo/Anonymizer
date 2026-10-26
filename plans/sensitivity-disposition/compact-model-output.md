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
