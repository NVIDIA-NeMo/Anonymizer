# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Authenticate initial collection parents at an actual qualification boundary."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectRejected
from anonymizer.engine.graph_sdk.executor import BoundInputKey, InitialCollectionKey
from anonymizer.engine.graph_sdk.qualification import QualificationResult, qualify
from tests.graph_sdk import context_source_assertions as fixture


@pytest.mark.parametrize(
    "corruption,expected",
    [
        ("missing-parent", "contradictory"),
        ("unretained-parent", "missing"),
        ("wrong-declaration", "missing"),
    ],
)
def test_initial_collection_exact_retained_parent_set(
    corruption: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def inspect(**kwargs: Any) -> QualificationResult:
        nonlocal calls
        calls += 1
        baseline = qualify(**kwargs)
        assert all(status.qualification == "not_assessed" for status in baseline.record.statuses)
        result = kwargs["result"]
        collection = next(fact for fact in result.provenance if isinstance(fact.key, InitialCollectionKey))
        assert len(collection.parents) == 2 and all(isinstance(key, BoundInputKey) for key in collection.parents)
        parents = collection.parents
        if corruption == "missing-parent":
            object.__setattr__(collection, "parents", frozenset(list(parents)[1:]))
        elif corruption == "unretained-parent":
            parent = next(iter(parents))
            other = replace(parent, binding_artifact=replace(parent.binding_artifact, key=999))
            object.__setattr__(collection, "parents", (parents - {parent}) | {other})
        else:
            old = collection.key
            new = replace(
                old,
                declaration=type(old.declaration).new(binding=old.declaration.binding, ordinal=999),
            )
            for fact in result.provenance:
                if fact.key == old:
                    object.__setattr__(fact, "key", new)
                object.__setattr__(
                    fact,
                    "parents",
                    frozenset(new if key == old else key for key in fact.parents),
                )
            object.__setattr__(
                result,
                "_input_parents",
                tuple((t, a, p, new if key == old else key) for t, a, p, key in result._input_parents),
            )
        with pytest.raises(EffectRejected) as error:
            qualify(**kwargs)
        assert error.value.code.value == expected
        return baseline

    monkeypatch.setattr(fixture, "qualify", inspect)
    asyncio.run(fixture._assert_context_execution("collection", same_key_versions=True))
    assert calls == 1
