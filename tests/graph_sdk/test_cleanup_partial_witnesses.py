# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retained cleanup ledger discriminators after real two-target execution."""

from __future__ import annotations

import asyncio

from anonymizer.engine.graph_sdk.context import ContextSourceRef
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.resources import ResourceLease
from tests.graph_sdk.evidence_fixtures import _execute_assessment
from tests.graph_sdk.qualification_fixtures import _BindingProvider, _initial_resource, _inputs, _RecoveringTransport


def test_binding_cleanup_associates_each_target_once() -> None:
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
    assert [(provider.calls, provider.closes) for provider in providers] == [(1, 1), (1, 1)]


def test_failed_close_is_one_sdk_cleanup_fact_for_all_targets() -> None:
    transport = _RecoveringTransport(fail_close=True)
    lease = ResourceLease.create(owner="sdk", safe_detachment="forbidden", handle=transport)
    execution, result = asyncio.run(_execute_assessment(target_count=2, external=(transport, lease)))
    admitted, current, submissions = _inputs(execution, result)
    (fact,), (association,) = result.cleanup, result.cleanup_associations
    assert fact.disposition == "close_failed" and fact.owner == "sdk"
    assert association.targets == execution.context.prepared.data.targets
    assert transport.calls == 2 and transport.closes == 1
