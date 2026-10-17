# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact retained root, final output, and membership discriminators."""

from __future__ import annotations

import asyncio

from tests.graph_sdk.evidence_fixtures import _execute_assessment
from tests.graph_sdk.qualification_fixtures import _inputs


def test_execution_only_open_root_membership_is_pending_not_assessed() -> None:
    execution, result = asyncio.run(_execute_assessment(execution_only=True))
    admitted, current, submissions = _inputs(execution, result)
    (root,) = result.record.memberships
    assert root.parent is None and root.closed
