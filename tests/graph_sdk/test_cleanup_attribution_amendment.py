# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retained cleanup corruption has invocation-wide attribution consequences."""

from __future__ import annotations

import asyncio

import pytest

from anonymizer.engine.graph_sdk.context import ContextSourceRef
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.resources import ResourceLease
from tests.graph_sdk.evidence_fixtures import _execute_assessment
from tests.graph_sdk.qualification_fixtures import (
    _BindingProvider,
    _initial_resource,
    _inputs,
    _RecoveringTransport,
)


@pytest.mark.parametrize("binding", [False, True])
def test_cleanup_ledgers_of_a_successful_two_target_run(binding: bool) -> None:
    if binding:
        providers = tuple(_BindingProvider(ContextSourceRef(name=name, revision=1)) for name in ("A", "B"))
        execution, result = asyncio.run(
            _execute_assessment(target_count=2, initial_resources=tuple(_initial_resource(p) for p in providers))
        )
        bound = execution.context.bound_context
        assert bound is not None
        assert all(p.calls == p.closes == 1 for p in providers)
    else:
        transport = _RecoveringTransport()
        lease = ResourceLease.create(owner="sdk", safe_detachment="forbidden", handle=transport)
        execution, result = asyncio.run(_execute_assessment(target_count=2, external=(transport, lease)))
        assert transport.calls == 2 and transport.closes == 1
    admitted, current, submissions = _inputs(execution, result)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(baseline.qualified) == 2
    if binding:
        assert all(p.calls == p.closes == 1 for p in providers)
    else:
        assert transport.calls == 2 and transport.closes == 1
