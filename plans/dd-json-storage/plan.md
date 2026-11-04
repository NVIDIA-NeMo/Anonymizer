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

- The supported range is `data-designer>=0.10.0,<2`; resolved config and engine are
  both 0.10.0. Notebook Pillow is updated to its compatible 13.x range.
- Real synthetic DataDesigner batch generation and handoffs into the production
  Substitute map and latent-detection workflows pass. Only the model facade and
  remote readiness check are mocked for those two workflow tests.
- Engine, interface, and config tests: 1,153 passed. Full repository typecheck,
  changed-file Ruff, and offline lock freshness check passed.
- Full `tests/` run: 1,613 passed and five environment-blocked tests. All five
  passed when rerun with local socket permissions and a shorter temporary path
  (1,618 distinct tests passed across the main run and bounded rerun).
- Fresh-install check: a new isolated virtualenv installed the built wheel
  non-editably over 113 hash-locked runtime/dev dependencies exported from this
  branch's lockfile. From outside the checkout, Python isolated mode and a
  separate pytest config verified site-packages imports and passed all 28
  adapter/storage regressions. DD, DD config, and DD engine were each 0.10.0;
  package dependency compatibility passed. No existing environment was reused.
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
