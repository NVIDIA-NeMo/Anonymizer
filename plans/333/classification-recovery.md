<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Classification validation and recovery

Validate coverage of requested custom labels and reject contradictory categories
before disposition. Accept identical duplicates and preserve authoritative local
categories. Move generation to a custom column with a Pydantic response parser
and request-aware validation, because structured columns use JSON Schema only.
Allow one correction with validation feedback and zero conversation restarts;
a second invalid response fails the record through the existing adapter boundary.
Keep defensive validation in the merge. No public configuration changes.

Tests exercise real ModelFacade correction with mocked provider completions,
both conflict orders, missing labels, exhaustion, identical duplicates, local
categories, and scheduler skip behavior. Run rewrite workflow tests and repository
checks. Ship with PR #333; no migration is required. Provider transport retries
retain their existing behavior; the two-attempt limit concerns invalid responses.
