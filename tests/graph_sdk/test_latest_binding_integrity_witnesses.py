# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Latest binding receipt integrity is checked again at qualification."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from anonymizer.engine.graph_sdk.context import ContextSourceRef
from anonymizer.engine.graph_sdk.executor import BoundInputKey
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.requests import BindingAssociation
from anonymizer.graph import EffectRejected
from anonymizer.graph._values import DatumId
from tests.graph_sdk.evidence_fixtures import _execute_assessment
from tests.graph_sdk.qualification_fixtures import (
    _BindingProvider,
    _initial_resource,
    _inputs,
)


@pytest.mark.parametrize(
    "corruption,expected",
    [
        ("missing-dispatch", "missing"),
        ("missing-settlement", "missing"),
        ("missing-cleanup", "missing"),
        ("foreign-association", "foreign_owner"),
        ("foreign-cleanup", "foreign_owner"),
        ("foreign-source", "foreign_owner"),
        ("foreign-artifact-source", "foreign_owner"),
    ],
)
def test_latest_binding_corruption_cannot_qualify(corruption: str, expected: str) -> None:
    provider = _BindingProvider(source=ContextSourceRef(name="source", revision=1), versions=(1, 2))
    target_count = 2 if corruption == "foreign-cleanup" else 1
    execution, result = asyncio.run(
        _execute_assessment(
            target_count=target_count,
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
        == target_count
    )
    bound = execution.context.bound_context
    assert bound is not None
    assert len([fact for fact in result.provenance if isinstance(fact.key, BoundInputKey)]) == 2 * target_count
    receipt = bound.receipt
    if corruption == "missing-dispatch":
        object.__setattr__(receipt.requests, "dispatches", ())
    elif corruption == "missing-settlement":
        object.__setattr__(receipt.requests, "settlements", ())
    elif corruption == "missing-cleanup":
        object.__setattr__(receipt, "cleanup", ())
    elif corruption == "foreign-association":
        dispatch = receipt.requests.dispatches[0]
        (association,) = dispatch.associations
        assert isinstance(association, BindingAssociation)
        object.__setattr__(
            dispatch,
            "associations",
            frozenset(
                {
                    replace(
                        association,
                        declaration=type(association.declaration).new(
                            binding=association.declaration.binding, ordinal=999
                        ),
                    )
                }
            ),
        )
    elif corruption == "foreign-cleanup":
        association = receipt.cleanup_associations[0]
        object.__setattr__(association, "targets", frozenset({DatumId.new(graph=result.record.graph)}))
    elif corruption == "foreign-artifact-source":
        artifact = max(bound.artifacts, key=lambda item: item.reference.version)
        selected = next(
            fact
            for fact in result.provenance
            if isinstance(fact.key, BoundInputKey) and fact.key.binding_artifact == artifact.reference
        )
        assert selected.artifact in {port.artifact for port in result.ports if port.port == "input"}
        assert artifact.source == receipt.sources[0].declaration.source
        assert artifact.reference.declaration == receipt.sources[0].identity
        assert artifact.reference in {item.reference for item in receipt.artifacts}
        object.__setattr__(artifact, "source", ContextSourceRef(name="foreign-source", revision=1))
    else:
        source = receipt.sources[0]
        object.__setattr__(
            source.declaration,
            "source",
            ContextSourceRef(name="foreign-source", revision=1),
        )
    assert provider.calls == target_count and provider.closes == 1
    if corruption in {
        "missing-dispatch",
        "missing-settlement",
        "missing-cleanup",
        "foreign-association",
        "foreign-cleanup",
    }:
        output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
        assert not output.qualified
        assert len(output.targets) == target_count
        assert all(status.qualification == "unknown" for status in output.record.statuses)
        expected_reason = "request_accounting" if corruption == "missing-settlement" else "inconsistent_attribution"
        assert all(target.withholding == frozenset({expected_reason}) for target in output.targets)
        assert provider.calls == target_count and provider.closes == 1
        return
    with pytest.raises(EffectRejected) as error:
        output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
        pytest.fail(
            f"Expected rejection; actual qualification={output.record.statuses[0].qualification}, withholding={output.targets[0].withholding}, released={len(output.qualified)}"
        )
    assert error.value.code.value == expected
    assert provider.calls == provider.closes == 1


@pytest.mark.parametrize("coordinate", ["target", "declaration", "unknown-declaration"])
def test_latest_bound_producer_cannot_cross_actual_target_declarations(
    coordinate: str,
) -> None:
    provider = _BindingProvider(source=ContextSourceRef(name="source", revision=1), versions=(1, 2))
    execution, result = asyncio.run(
        _execute_assessment(
            target_count=2,
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
        == 2
    )
    facts = [fact for fact in result.provenance if isinstance(fact.key, BoundInputKey)]
    selected = next(fact for fact in facts if fact.artifact.version == 2)
    old = selected.key
    assert isinstance(old, BoundInputKey)
    other = next(
        fact.key
        for fact in facts
        if isinstance(fact.key, BoundInputKey) and fact.key.target != old.target and fact.artifact.version == 2
    )
    assert isinstance(other, BoundInputKey)
    new = (
        replace(old, target=other.target)
        if coordinate == "target"
        else replace(old, binding_artifact=other.binding_artifact)
    )
    if coordinate == "unknown-declaration":
        new = replace(
            old,
            binding_artifact=replace(
                old.binding_artifact,
                declaration=type(old.binding_artifact.declaration).new(
                    binding=old.binding_artifact.declaration.binding, ordinal=999
                ),
            ),
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
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert error.value.code.value == ("missing" if coordinate == "unknown-declaration" else "foreign_owner")
    assert provider.calls == 2 and provider.closes == 1


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


def test_duplicate_retained_source_pair_is_not_silently_coalesced() -> None:
    provider = _BindingProvider(source=ContextSourceRef(name="source", revision=1))
    execution, result = asyncio.run(_execute_assessment(initial_resources=(_initial_resource(provider),)))
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
    bound = execution.context.bound_context
    assert bound is not None and len(bound.artifacts) == 1
    pair = (bound.artifacts[0], bound.artifacts[0])
    object.__setattr__(bound, "artifacts", pair)
    object.__setattr__(bound.receipt, "artifacts", pair)
    with pytest.raises(EffectRejected) as error:
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert error.value.code.value == "duplicate"
