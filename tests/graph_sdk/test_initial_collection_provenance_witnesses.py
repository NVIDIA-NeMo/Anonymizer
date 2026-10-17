# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Authenticate initial collection parents at an actual qualification boundary."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from anonymizer.engine.graph_sdk.executor import BoundInputKey, InitialCollectionKey
from anonymizer.engine.graph_sdk.qualification import QualificationResult, qualify
from tests.graph_sdk import context_source_assertions as fixture


def test_initial_collection_retains_both_bound_parents(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def inspect(**kwargs: Any) -> QualificationResult:
        nonlocal calls
        calls += 1
        baseline = qualify(**kwargs)
        assert all(status.qualification == "not_assessed" for status in baseline.record.statuses)
        result = kwargs["result"]
        collection = next(fact for fact in result.provenance if isinstance(fact.key, InitialCollectionKey))
        assert len(collection.parents) == 2 and all(isinstance(key, BoundInputKey) for key in collection.parents)
        return baseline

    monkeypatch.setattr(fixture, "qualify", inspect)
    asyncio.run(fixture._assert_context_execution("collection", same_key_versions=True))
    assert calls == 1
