# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
from typing import cast
from unittest.mock import patch

import pytest

from anonymizer.engine.graph_sdk.continuity import StateNamespace, commit_qualified, create_continuity_state
from anonymizer.engine.graph_sdk.extensions import (
    ArtifactSelection,
    ContentKind,
    RemoteContext,
    bind_projection_source,
    project_content,
    project_protected,
)
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.requests import TextArtifactValue, TextCollectionValue
from anonymizer.graph import EffectCode, EffectRejected
from tests.graph_sdk.context_source_assertions import _assert_context_execution
from tests.graph_sdk.context_source_fixtures import _ContextProvider
from tests.graph_sdk.continuity_fixtures import limits, prepare_case
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.graph_sdk.test_qualification_subject_context import _execute_separate_subject_context


def test_all_projection_kinds_keep_one_record_and_do_no_work() -> None:
    async def run() -> None:
        state = create_continuity_state(namespace=StateNamespace(name="history"), limits=limits())
        graph, admitted = await prepare_case("independent", state)
        with pytest.raises(EffectRejected) as missing:
            bind_projection_source(execution=graph.result, qualification=graph.qualified, continuity=None)
        assert missing.value.code is EffectCode.MISSING
        receipt = commit_qualified(admitted=admitted, execution=graph.result, qualification=graph.qualified)
        source = bind_projection_source(execution=graph.result, qualification=graph.qualified, continuity=receipt)
        selections = tuple(
            ArtifactSelection(target=o.target, artifact=o.candidate.artifact) for o in graph.qualified.qualified
        )
        before = state.view(limits=limits())
        with (
            patch(
                "anonymizer.engine.graph_sdk.qualification.qualify", side_effect=AssertionError("no requalification")
            ),
            patch("anonymizer.engine.graph_sdk.continuity.commit_qualified", side_effect=AssertionError("no commit")),
        ):
            protected = project_protected(source=source)
            assert protected.record is graph.qualified.record
            for kind in ["annotation", "research", "execution_only", "custom"]:
                content = project_content(source=source, kind=kind, artifacts=selections)
                assert content.record is protected.record
                assert {item.content.text for item in content.items if isinstance(item.content, TextArtifactValue)} == {
                    item.content for item in protected.items
                }
        assert state.view(limits=limits()) == before
        with pytest.raises(EffectRejected) as relabel:
            project_content(source=source, kind=cast(ContentKind, "protected"), artifacts=selections)
        assert relabel.value.code is EffectCode.UNSUPPORTED
        with pytest.raises(EffectRejected) as duplicate:
            project_content(source=source, kind="custom", artifacts=(selections[0], selections[0]))
        assert duplicate.value.code is EffectCode.DUPLICATE
        other, other_admitted = await prepare_case("independent", state, changes={"x": "reuse", "y": "reuse"})
        other_receipt = commit_qualified(admitted=other_admitted, execution=other.result, qualification=other.qualified)
        with pytest.raises(EffectRejected) as foreign:
            bind_projection_source(execution=graph.result, qualification=graph.qualified, continuity=other_receipt)
        assert foreign.value.code is EffectCode.FOREIGN_OWNER
        # A copied structurally equal record cannot replace the owner's exact object.
        original = source.record
        object.__setattr__(source, "record", copy.copy(original))
        # CanonicalRecord itself copies by identity; manufacture the independent value through its public constructor.
        from dataclasses import fields

        object.__setattr__(
            source, "record", type(original)(**{f.name: getattr(original, f.name) for f in fields(original)})
        )
        with pytest.raises(EffectRejected) as copied:
            project_protected(source=source)
        assert copied.value.code is EffectCode.FOREIGN_OWNER

    asyncio.run(run())


def test_real_collection_content_keeps_versions_without_flattening() -> None:
    execution, result = asyncio.run(
        _execute_separate_subject_context(
            versioned_context_dependencies=frozenset({"a"}), versioned_context_ports=("a",), initial_collection=True
        )
    )
    admitted, current, submissions = _inputs(execution, result, latest=True)
    qualified = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    source = bind_projection_source(execution=result, qualification=qualified, continuity=None)
    port = next(port for port in result.ports if port.port == "a")
    content = project_content(
        source=source, kind="research", artifacts=(ArtifactSelection(target=port.target, artifact=port.artifact),)
    )
    value = content.items[0].content
    assert isinstance(value, TextCollectionValue)
    assert len(value.items) == 2 and {item.version for item in value.items} == {1, 2}
    assert content.record is qualified.record and content.items[0].status is qualified.record.statuses[0]
    assert project_protected(source=source).record is content.record


def test_nominal_remote_context_uses_existing_binding_and_ownership() -> None:
    seen = []

    @dataclass
    class Nominal(_ContextProvider, RemoteContext):
        def __post_init__(self) -> None:
            seen.append(self)

    # Inherit the dataclass constructor explicitly; no registry/client or alternate result protocol.
    with patch("tests.graph_sdk.context_source_assertions._ContextProvider", Nominal):
        asyncio.run(_assert_context_execution("scalar"))
        asyncio.run(_assert_context_execution("collection", same_key_versions=True))
    assert len(seen) == 2 and all(provider.calls == 1 for provider in seen)
    assert issubclass(Nominal, RemoteContext)
    with pytest.raises(TypeError):
        RemoteContext()


@pytest.mark.parametrize("latest", [False, True])
def test_nominal_remote_malformed_oversize_and_cancellation(latest: bool) -> None:
    from tests.graph_sdk import test_binding as binding_fixture

    class Boundary(binding_fixture._BoundaryProvider, RemoteContext):
        pass

    class Late(binding_fixture._LateProvider, RemoteContext):
        pass

    with patch.object(binding_fixture, "_BoundaryProvider", Boundary):
        asyncio.run(binding_fixture._assert_initial_binding_boundary_settlements(latest=latest))
    with patch.object(binding_fixture, "_LateProvider", Late):
        for acknowledged in (False, True):
            asyncio.run(binding_fixture._assert_late_initial_binding("failure", acknowledged, latest=latest))


def test_protected_projection_rejects_collection_instead_of_flattening() -> None:
    from anonymizer.engine.graph_sdk.requests import TextArtifactValue, TextCollectionItem

    async def run() -> None:
        state = create_continuity_state(namespace=StateNamespace(name="history"), limits=limits())
        graph, admitted = await prepare_case("independent", state)
        receipt = commit_qualified(admitted=admitted, execution=graph.result, qualification=graph.qualified)
        source = bind_projection_source(execution=graph.result, qualification=graph.qualified, continuity=receipt)
        candidate = graph.qualified.qualified[0].candidate.artifact
        previous = graph.result.artifacts
        # Boundary fault injection into an existing factory result tests only the scalar guard;
        # real collection production/content preservation is exercised separately above.
        value = TextCollectionValue(
            items=(TextCollectionItem(key=0, version=1, value=TextArtifactValue(text="secret")),)
        )
        object.__setattr__(
            graph.result, "artifacts", tuple((ref, value if ref == candidate else old) for ref, old in previous)
        )
        try:
            with pytest.raises(EffectRejected) as failure:
                project_protected(source=source)
            assert failure.value.code is EffectCode.UNSUPPORTED
        finally:
            object.__setattr__(graph.result, "artifacts", previous)

    asyncio.run(run())
