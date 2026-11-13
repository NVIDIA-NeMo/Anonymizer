<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Shared JSON storage handoffs

Upgrade the minimum supported DataDesigner version to 0.10.0. At the shared
adapter boundary, carry DataDesigner's JSON storage contract for explicitly
registered nested engine columns, preserving decoded Python values and existing
field-level contracts. Use the same preparation for local execution and exported
configs. Do not infer JSON from strings or change arbitrary user columns.

Exercise real Parquet write/read/write with empty and populated nested values,
including Arrow-backed batch inputs, before testing Substitute and latent
detection handoffs. Verify no diagnostic columns are lost.

DataDesigner 0.10.0 cannot independently declare storage encoding for a custom
generator's side-effect outputs. Test the incoming seed handoff separately from
initial generation storage; a PR305-specific declaration/refactor may remain.

Document through a deterministic no-inference test whether an all-failed batch
still raises rather than returning FailedRecord entries (#331). Do not change
all-failed semantics as part of this storage prerequisite.

## Verification and scope

- The supported range is `data-designer>=0.10.0,<0.11`; resolved config and engine are
  both 0.10.0. Notebook Pillow is updated to its compatible 13.x range.
- Normalize registered and explicitly planned seed columns to plain Python
  values, including object-dtype NumPy arrays and scalars, before JSON encoding.
  Preserve existing field-level storage paths and unrelated user columns.
- Cover empty and populated row groups with `buffer_size=2`, carrying the
  resulting side-effect diagnostics into the production Substitute map and
  latent-detection workflows. Mock only model generation and remote readiness;
  use real scheduling, prompt rendering, response parsing, and Parquet I/O.
- Check malformed storage metadata and invalid JSON values through run, preview,
  and exported-config paths. Seed preparation and writes must raise
  `AnonymizerWorkflowError` with the original exception chained.
- Run adapter/storage regressions, the full test suite, and `make check`.
  Keep revision-specific validation results in the PR description.
- Verify a non-editable wheel install over the locked dependencies from outside
  the checkout, including dependency compatibility and adapter/storage tests.
- A deterministic custom generator that fails every row still produces
  `AnonymizerWorkflowError`, caused by `DataDesignerGenerationError`, under
  DataDesigner 0.10.0. This upgrade does not resolve #331.
- PR305's `_validation_request_inventories` constant is not on main. After
  rebasing, add its constant to `JSON_STORAGE_COLUMNS` to cover incoming
  Arrow-backed inventories at every adapter handoff. The registry does not
  automatically register plugin side effects for initial batch writes; DD0.10
  has no corresponding public side-effect storage flag. If the initial write
  must be JSON-encoded too, promote that inventory to a named output generator
  with `storage_encoding="json"` or add upstream side-effect storage support.
