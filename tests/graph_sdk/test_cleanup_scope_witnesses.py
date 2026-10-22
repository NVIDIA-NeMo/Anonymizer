# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution cleanup distinguishes attribution failure from transport-only failure."""

from __future__ import annotations

import asyncio

import pytest

from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.resources import ResourceLease
from tests.graph_sdk.evidence_fixtures import _execute_assessment
from tests.graph_sdk.qualification_fixtures import _inputs, _RecoveringTransport


@pytest.mark.parametrize("fail_close", [False, True])
def test_execution_cleanup_scope_of_a_two_target_run(fail_close: bool) -> None:
    transport = _RecoveringTransport(fail_close=fail_close)
    lease = ResourceLease.create(owner="sdk", safe_detachment="forbidden", handle=transport)
    execution, result = asyncio.run(_execute_assessment(target_count=2, external=(transport, lease)))
    admitted, current, submissions = _inputs(execution, result)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(baseline.qualified) == (0 if fail_close else 2)
    assert transport.calls == 2 and transport.closes == 1
