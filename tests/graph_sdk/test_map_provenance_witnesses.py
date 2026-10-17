# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retained map coordinates and collection parents after actual execution."""

from __future__ import annotations

import asyncio

import pytest

from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.graph import EffectRejected
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.graph_sdk.qualification_map_execution import _execute_reference_map


@pytest.mark.parametrize(
    "corruption",
    ["membership-port"],
)
def test_map_collection_output_identity_and_declaration(corruption: str) -> None:
    execution, result, nodes = asyncio.run(_execute_reference_map(1))
    admitted, current, submissions = _inputs(execution, result)
    assert (
        len(
            qualify(
                admitted=admitted,
                result=result,
                current=current,
                submissions=submissions,
            ).qualified
        )
        == 1
    )
    (publication,) = execution.map_expansions
    object.__setattr__(publication, "membership_port", "not_members")
    with pytest.raises(EffectRejected) as error:
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert error.value.code.value == "contradictory"
