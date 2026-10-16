# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise structural ownership through actual retained execution fields."""

from __future__ import annotations

import asyncio

import pytest

from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.graph import EffectCode, EffectRejected
from tests.graph_sdk.test_qualification import _inputs
from tests.graph_sdk.test_qualification_subject_context import _execute_separate_subject_context


def test_duplicate_retained_membership_rows_preserve_observable_multiplicity() -> None:
    execution, result = asyncio.run(_execute_separate_subject_context())
    admitted, current, submissions = _inputs(execution, result)
    assert len(qualify(admitted=admitted, result=result, current=current, submissions=submissions).qualified) == 1
    (root,) = result.record.memberships
    assert root.parent is None and len(root.members) == 1
    object.__setattr__(result.record, "memberships", (root, root))
    with pytest.raises(EffectRejected) as rejected:
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert rejected.value.code is EffectCode.DUPLICATE
