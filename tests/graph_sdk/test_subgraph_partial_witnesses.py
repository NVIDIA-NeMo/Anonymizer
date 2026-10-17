# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact captured producer and body projection corruption after real execution."""

from __future__ import annotations

import asyncio

from anonymizer.engine.graph_sdk.executor import OperationOutputKey, RootInputKey
from anonymizer.engine.graph_sdk.qualification import qualify
from tests.graph_sdk.evidence_fixtures import _execute_assessment
from tests.graph_sdk.qualification_fixtures import _inputs


def test_subgraph_body_projection_keeps_exact_parents() -> None:
    execution, result = asyncio.run(_execute_assessment(nested=True, candidate_input=True))
    admitted, current, submissions = _inputs(execution, result)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(baseline.qualified) == 1
    (root,) = tuple(item for item in result.provenance if isinstance(item.key, RootInputKey))
    (body,) = tuple(
        item
        for item in result.provenance
        if isinstance(item.key, OperationOutputKey) and item.key.activation.parent is not None
    )
    (container,) = tuple(
        item
        for item in result.provenance
        if isinstance(item.key, OperationOutputKey) and item.key.activation.parent is None
    )
    assert root.artifact == body.artifact == container.artifact
    assert container.parents == frozenset({body.key})
