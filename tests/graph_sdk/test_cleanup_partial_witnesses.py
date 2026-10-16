# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retained cleanup ledger discriminators after real two-target execution."""

from __future__ import annotations

import asyncio

import pytest

from anonymizer.engine.graph_sdk.context import ContextSourceRef
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.resources import ResourceLease
from tests.graph_sdk.evidence_fixtures import _execute_assessment
from tests.graph_sdk.qualification_fixtures import _BindingProvider, _initial_resource, _inputs, _RecoveringTransport


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_binding_attribution_failure_withholds_successful_peer_without_effects(mutation: str) -> None:
    providers = tuple(_BindingProvider(ContextSourceRef(name=name, revision=1)) for name in ("A", "B"))
    execution, result = asyncio.run(
        _execute_assessment(
            target_count=2, initial_resources=tuple(_initial_resource(provider) for provider in providers)
        )
    )
    bound = execution.context.bound_context
    assert bound is not None
    admitted, current, submissions = _inputs(execution, result)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(baseline.qualified) == 2
    associations, cleanup = bound.receipt.cleanup_associations, bound.receipt.cleanup
    assert len(associations) == len(cleanup) == 2
    assert len({next(iter(item.targets)) for item in associations}) == 2
    if mutation == "missing":
        object.__setattr__(bound.receipt, "cleanup_associations", associations[1:])
    elif mutation == "extra":
        object.__setattr__(bound.receipt, "cleanup", cleanup[1:])
    else:
        object.__setattr__(bound.receipt, "cleanup_associations", (*associations, associations[0]))
    try:
        output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
        assert not output.qualified
        assert len(output.targets) == 2
        assert len(output.verified) == 2
        assert all(item.candidate is not None for item in output.targets)
        assert all(item.completion == "closed" for item in output.record.statuses)
        assert all(item.withholding == frozenset({"inconsistent_attribution"}) for item in output.targets)
        assert all(item.qualification == "unknown" for item in output.record.statuses)
        assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == output
    finally:
        object.__setattr__(bound.receipt, "cleanup_associations", associations)
        object.__setattr__(bound.receipt, "cleanup", cleanup)
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == baseline
    assert [(provider.calls, provider.closes) for provider in providers] == [(1, 1), (1, 1)]


@pytest.mark.parametrize(
    "purpose,disposition,expected",
    [
        ("accounting", "close_unknown", "cleanup_accounting"),
        ("verification", "close_failed", "cleanup_verification"),
        ("accounting", "left_open", "inconsistent_attribution"),
    ],
)
def test_execution_cleanup_purpose_and_disposition_have_exact_scope(
    purpose: str, disposition: str, expected: str
) -> None:
    transport = _RecoveringTransport(fail_close=True)
    lease = ResourceLease.create(owner="sdk", safe_detachment="forbidden", handle=transport)
    execution, result = asyncio.run(_execute_assessment(target_count=2, external=(transport, lease)))
    admitted, current, submissions = _inputs(execution, result)
    (fact,), (association,) = result.cleanup, result.cleanup_associations
    assert fact.disposition == "close_failed" and fact.owner == "sdk"
    assert association.targets == execution.context.prepared.data.targets
    original = fact.disposition, association.purpose, association.targets
    selected = next(iter(association.targets))
    object.__setattr__(fact, "disposition", disposition)
    object.__setattr__(association, "purpose", purpose)
    object.__setattr__(association, "targets", frozenset({selected}))
    try:
        output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
        assert len(output.verified) == 2
        assert all(item.candidate is not None for item in output.targets)
        assert all(item.completion == "closed" for item in output.record.statuses)
        expected_targets = (
            execution.context.prepared.data.targets if expected == "inconsistent_attribution" else {selected}
        )
        assert {item.target for item in output.targets if item.withholding} == expected_targets
        assert all(item.withholding in (frozenset(), frozenset({expected})) for item in output.targets)
        assert len(output.qualified) == (0 if expected == "inconsistent_attribution" else 1)
        withheld_status = "unknown" if expected == "inconsistent_attribution" else "unmet"
        assert all(
            item.qualification == (withheld_status if item.target in expected_targets else "met")
            for item in output.record.statuses
        )
        assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == output
    finally:
        object.__setattr__(fact, "disposition", original[0])
        object.__setattr__(association, "purpose", original[1])
        object.__setattr__(association, "targets", original[2])
    assert transport.calls == 2 and transport.closes == 1
