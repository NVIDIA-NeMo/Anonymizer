# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retained cleanup corruption has invocation-wide attribution consequences."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from anonymizer.engine.graph_sdk.context import ContextSourceRef
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.resources import ResourceLease
from anonymizer.graph._values import DatumId
from tests.graph_sdk.evidence_fixtures import _execute_assessment
from tests.graph_sdk.qualification_fixtures import (
    _BindingProvider,
    _initial_resource,
    _inputs,
    _RecoveringTransport,
)


@pytest.mark.parametrize("binding", [False, True])
@pytest.mark.parametrize("corruption", ["duplicate-fact", "duplicate-association", "sdk-left-open", "foreign-target"])
def test_cleanup_attribution_corruption_withholds_successful_peer(binding: bool, corruption: str) -> None:
    """CA1/CA2 apply to both ledgers, including an otherwise valid peer."""
    if binding:
        providers = tuple(_BindingProvider(ContextSourceRef(name=name, revision=1)) for name in ("A", "B"))
        execution, result = asyncio.run(
            _execute_assessment(target_count=2, initial_resources=tuple(_initial_resource(p) for p in providers))
        )
        bound = execution.context.bound_context
        assert bound is not None
        ledger = bound.receipt
        assert all(p.calls == p.closes == 1 for p in providers)
    else:
        transport = _RecoveringTransport()
        lease = ResourceLease.create(owner="sdk", safe_detachment="forbidden", handle=transport)
        execution, result = asyncio.run(_execute_assessment(target_count=2, external=(transport, lease)))
        ledger = result
        assert transport.calls == 2 and transport.closes == 1
    admitted, current, submissions = _inputs(execution, result)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(baseline.qualified) == 2
    cleanup, associations = ledger.cleanup, ledger.cleanup_associations
    original_targets = associations[0].targets
    if corruption == "duplicate-fact":
        object.__setattr__(ledger, "cleanup", (*cleanup, cleanup[0]))
    elif corruption == "duplicate-association":
        object.__setattr__(ledger, "cleanup_associations", (*associations, associations[0]))
    elif corruption == "sdk-left-open":
        object.__setattr__(ledger, "cleanup", (replace(cleanup[0], disposition="left_open"), *cleanup[1:]))
    else:
        object.__setattr__(associations[0], "targets", frozenset({DatumId.new(graph=result.record.graph)}))
    try:
        output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
        assert not output.qualified
        assert all(item.withholding == frozenset({"inconsistent_attribution"}) for item in output.targets)
        assert all(item.qualification == "unknown" for item in output.record.statuses)
        assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == output
    finally:
        object.__setattr__(associations[0], "targets", original_targets)
        object.__setattr__(ledger, "cleanup", cleanup)
        object.__setattr__(ledger, "cleanup_associations", associations)
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == baseline
    if binding:
        assert all(p.calls == p.closes == 1 for p in providers)
    else:
        assert transport.calls == 2 and transport.closes == 1
