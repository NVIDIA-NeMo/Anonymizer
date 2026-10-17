# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Latest binding receipt integrity is checked again at qualification."""

from __future__ import annotations

import asyncio

import pytest

from anonymizer.engine.graph_sdk.context import ContextSourceRef
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.graph import EffectRejected
from tests.graph_sdk.evidence_fixtures import _execute_assessment
from tests.graph_sdk.qualification_fixtures import (
    _BindingProvider,
    _initial_resource,
    _inputs,
)


def test_latest_revision_view_cannot_select_two_versions_of_one_key() -> None:
    from anonymizer.engine.graph_sdk.evidence import evidence_revision_view

    provider = _BindingProvider(source=ContextSourceRef(name="source", revision=1), versions=(1, 2))
    execution, result = asyncio.run(
        _execute_assessment(
            initial_resources=(_initial_resource(provider),),
            initial_item_limit=2,
            initial_version_selection="latest",
        )
    )
    admitted, current, submissions = _inputs(execution, result, latest=True)
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
    with pytest.raises(EffectRejected) as error:
        evidence_revision_view(
            admitted=admitted,
            result=result,
            artifacts=tuple(ref for ref, _ in result.artifacts),
            configurations=current.configurations,
        )
    assert error.value.code.value == "duplicate"
