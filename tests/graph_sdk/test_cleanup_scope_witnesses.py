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


@pytest.mark.parametrize("case", ["empty-accounting", "empty-verification", "missing-association", "transport-failed"])
def test_execution_cleanup_scope_controls_qualification(case: str) -> None:
    transport = _RecoveringTransport(fail_close=case == "transport-failed")
    lease = ResourceLease.create(owner="sdk", safe_detachment="forbidden", handle=transport)
    execution, result = asyncio.run(_execute_assessment(target_count=2, external=(transport, lease)))
    admitted, current, submissions = _inputs(execution, result)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(baseline.qualified) == (0 if case == "transport-failed" else 2)
    (association,) = result.cleanup_associations
    targets, purpose = association.targets, association.purpose
    associations = result.cleanup_associations
    if case == "missing-association":
        object.__setattr__(result, "cleanup_associations", ())
    elif case == "transport-failed":
        assert result.cleanup[0].disposition == "close_failed"
        object.__setattr__(association, "purpose", "transport_only")
    else:
        object.__setattr__(association, "targets", frozenset())
        object.__setattr__(association, "purpose", "verification" if case == "empty-verification" else "accounting")
    try:
        output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
        if case == "transport-failed":
            assert len(output.qualified) == 2
            assert all(not item.withholding for item in output.targets)
        else:
            assert not output.qualified
            assert all(item.withholding == frozenset({"inconsistent_attribution"}) for item in output.targets)
            assert all(item.qualification == "unknown" for item in output.record.statuses)
        assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == output
    finally:
        object.__setattr__(association, "targets", targets)
        object.__setattr__(association, "purpose", purpose)
        object.__setattr__(result, "cleanup_associations", associations)
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == baseline
    assert transport.calls == 2 and transport.closes == 1
