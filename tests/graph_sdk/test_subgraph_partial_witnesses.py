# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact captured producer and body projection corruption after real execution."""

from __future__ import annotations

import asyncio

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk.executor import OperationOutputKey, RootInputKey
from anonymizer.engine.graph_sdk.qualification import qualify
from tests.graph_sdk.evidence_fixtures import _execute_assessment
from tests.graph_sdk.qualification_fixtures import _inputs


@pytest.mark.parametrize("mutation", ["body-projection", "body-input-capture", "passthrough-capture"])
def test_subgraph_exact_projection_or_capture_rejects_equal_artifact_substitute(mutation: str) -> None:
    passthrough = mutation == "passthrough-capture"
    execution, result = asyncio.run(
        _execute_assessment(nested=True, candidate_input=True, root_passthrough=passthrough)
    )
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
    if mutation == "body-projection":
        assert container.parents == frozenset({body.key})
        owner, field, original = container, "parents", container.parents
        replacement = frozenset({root.key})
    else:
        field = "_passthrough_parents" if passthrough else "_input_parents"
        owner, original = result, getattr(result, field)
        assert len(original) == 1 and original[0][3] == root.key
        replacement = ((*original[0][:3], body.key),)
    object.__setattr__(owner, field, replacement)
    try:
        with pytest.raises(EffectRejected) as error:
            qualify(admitted=admitted, result=result, current=current, submissions=submissions)
        assert error.value.code is EffectCode.CONTRADICTORY
    finally:
        object.__setattr__(owner, field, original)
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == baseline
