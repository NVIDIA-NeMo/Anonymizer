# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retained binding corruption cannot bypass qualification or repeat effects."""

from __future__ import annotations

import asyncio

from anonymizer.engine.graph_sdk.context import ContextSourceRef
from tests.graph_sdk.evidence_fixtures import _execute_assessment
from tests.graph_sdk.qualification_fixtures import _BindingProvider, _initial_resource


def test_source_join_refactor_preserves_existing_context_admission_diagnostic() -> None:
    provider = _BindingProvider(source=ContextSourceRef(name="source", revision=1), versions=(1, 2))
    execution, _ = asyncio.run(
        _execute_assessment(
            initial_resources=(_initial_resource(provider),),
            initial_item_limit=2,
            initial_version_selection="latest",
        )
    )
    context = execution.context
    bound = context.bound_context
    assert bound is not None
