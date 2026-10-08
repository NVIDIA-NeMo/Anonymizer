# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare retained-record corruption after authentic map execution."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectRejected
from anonymizer.engine.graph_sdk.executor import MapItemKey, RootInputKey
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.graph._values import ActivationKey, DatumId, InvocationId
from tests.graph_sdk import test_qualification_map_conformance as fixture

CASE_IDS = {
    f"map_item_evidence/{suffix}"
    for suffix in (
        "wrong_item_version",
        "wrong_item_key",
        "wrong_item_owner",
        "wrong_expander",
        "wrong_member",
        "wrong_target",
        "wrong_invocation",
        "typed_consumed_wrong_owner",
    )
}


@pytest.mark.parametrize(
    "case", [case for case in fixture.CORPUS if case["case_id"] in CASE_IDS], ids=lambda case: case["case_id"]
)
def test_retained_map_corruption_matches_reference(case: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    baseline_id = (
        "map_item_evidence/typed_consumed_endpoint"
        if case["case_id"].endswith("typed_consumed_wrong_owner")
        else "map_item_evidence/direct_1"
    )
    baseline = next(item for item in fixture.CORPUS if item["case_id"] == baseline_id)
    assert case["declaration"] == baseline["declaration"]
    assert len(case["events"]) == len(baseline["events"])
    changes = [
        (before, after) for before, after in zip(baseline["events"], case["events"], strict=True) if before != after
    ]
    assert len(changes) == 1
    before, after = changes[0]
    changed_fields = {key for key in before if before[key] != after[key]}
    assert len(changed_fields) == 1
    field = changed_fields.pop()
    calls = 0

    def corrupt_then_qualify(**kwargs: Any):
        nonlocal calls
        calls += 1
        result = kwargs["result"]
        provenance = next(item for item in result.provenance if isinstance(item.key, MapItemKey))
        key = provenance.key
        assert isinstance(key, MapItemKey)
        if before["kind"] == "provenance":
            assert before["source"] == "map_item"
            if field in {"item_key", "item_version"}:
                assert getattr(key, field) == before[field]
                replacement = replace(key, **{field: after[field]})
            elif field == "target":
                assert (before[field], after[field]) == ("A", "B")
                replacement = replace(key, target=DatumId.new(graph=key.target.graph))
            else:
                assert field in {"member", "expander"} and after[field] == "OTHER"
                original = getattr(key, field)
                other = ActivationKey(
                    invocation=original.invocation, occurrence=999, parent=original.parent, iteration=original.iteration
                )
                # Corrupt the retained key, not its constructor: an invalid
                # expander/member relationship cannot be publicly constructed.
                replacement = replace(key)
                object.__setattr__(replacement, field, other)
            object.__setattr__(provenance, "key", replacement)
            # The neutral producer ID remains MAPITEM:0. Translate every
            # reference to that ID to the changed structural key as well.
            for item in result.provenance:
                object.__setattr__(
                    item, "parents", frozenset(replacement if parent == key else parent for parent in item.parents)
                )
            object.__setattr__(
                result,
                "_input_parents",
                tuple(
                    (target, activation, port, replacement if producer == key else producer)
                    for target, activation, port, producer in result._input_parents
                ),
            )
        elif before["kind"] == "input_producer":
            assert field == "producer" and (before[field], after[field]) == ("MAPITEM:0", "ROOT:A:subject")
            root = next(
                item.key
                for item in result.provenance
                if isinstance(item.key, RootInputKey) and item.key.port == "subject"
            )
            object.__setattr__(
                result,
                "_input_parents",
                tuple(
                    (target, activation, port, root if activation == key.member and port == key.port else producer)
                    for target, activation, port, producer in result._input_parents
                ),
            )
        else:
            assert before["kind"] == "artifact" and field == "invocation"
            assert (before[field], after[field]) == ("I0", "I1")
            foreign = InvocationId.new(plan=provenance.artifact.invocation.plan)
            object.__setattr__(
                result,
                "artifacts",
                tuple(
                    (replace(ref, invocation=foreign) if ref == provenance.artifact else ref, value)
                    for ref, value in result.artifacts
                ),
            )
        return qualify(**kwargs)

    monkeypatch.setattr(fixture, "qualify", corrupt_then_qualify)
    try:
        actual = asyncio.run(fixture._run_reference_map_case(baseline))
    except EffectRejected as exc:
        actual = {"status": "rejected", "code": exc.code.value}
    assert calls == 1
    assert actual == case["expected"]
