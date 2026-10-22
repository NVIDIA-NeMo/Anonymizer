# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check retained membership and terminal integrity after real nested execution."""

from __future__ import annotations

import asyncio

from anonymizer.engine.graph_sdk.qualification import qualify
from tests.graph_sdk.evidence_fixtures import _execute_assessment
from tests.graph_sdk.qualification_fixtures import _inputs


def test_nested_execution_retains_container_and_operation_terminals() -> None:
    execution, result = asyncio.run(_execute_assessment(nested=True))
    admitted, current, submissions = _inputs(execution, result)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(baseline.qualified) == 1
    assert any(t.structural for t in result.record.terminals)
    assert any(not t.structural for t in result.record.terminals)
