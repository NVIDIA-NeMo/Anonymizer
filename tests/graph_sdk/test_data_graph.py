# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Product contract tests for immutable data graph validation."""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import pytest

from anonymizer.engine.graph_sdk.data import (
    AtomicGroup,
    CoherenceScope,
    ContextView,
    DataGraph,
    DataLimits,
    DatumDependency,
    OutputRegion,
    OwnershipRange,
    SourceRelation,
    SourceView,
    ValidatedDataGraph,
)
from anonymizer.graph._values import ContractViolation, DatumId, GraphId, ValidationCode


def _limits(graph: DataGraph, *, declarations: int = 0, group_members: int = 0) -> DataLimits:
    return DataLimits(
        max_datums=len(graph.datums),
        max_targets=len(graph.datums),
        max_text_bytes=sum(len(datum.text.encode("utf-8")) for datum in graph.datums),
        max_declarations=declarations,
        max_group_members=group_members,
    )


def _validate(
    graph: DataGraph,
    targets: tuple[DatumId, ...],
    *,
    source_relations: tuple[SourceRelation, ...] = (),
    contexts: tuple[ContextView, ...] = (),
    dependencies: tuple[DatumDependency, ...] = (),
    coherence: tuple[CoherenceScope, ...] = (),
    atomic: tuple[AtomicGroup, ...] = (),
    output_regions: tuple[OutputRegion, ...] = (),
) -> ValidatedDataGraph:
    declarations = sum(
        len(values) for values in (source_relations, contexts, dependencies, coherence, atomic, output_regions)
    )
    group_members = sum(len(group.members) for group in (*coherence, *atomic))
    return graph.validate(
        targets=targets,
        source_relations=source_relations,
        contexts=contexts,
        dependencies=dependencies,
        coherence=coherence,
        atomic=atomic,
        output_regions=output_regions,
        limits=_limits(graph, declarations=declarations, group_members=group_members),
    )


def _graph_with_texts(*texts: str) -> tuple[DataGraph, tuple[DatumId, ...]]:
    graph = DataGraph.new()
    identifiers: list[DatumId] = []
    for text in texts:
        graph, identifier = graph.add_text(text)
        identifiers.append(identifier)
    return graph, tuple(identifiers)


def test_graph_versions_and_equal_text_datums_preserve_identity() -> None:
    empty = DataGraph.new()
    first_graph, first = empty.add_text("same")
    second_graph, second = first_graph.add_text("same")
    assert empty.datums == ()
    assert first_graph.graph is empty.graph is second_graph.graph
    assert first != second
    assert second not in {datum.id for datum in first_graph.datums}
    with pytest.raises(ContractViolation) as missing:
        _validate(first_graph, (second,))
    assert missing.value.code is ValidationCode.MISSING


def test_context_cycles_and_overlapping_reads_are_legal_but_dependency_cycles_are_not() -> None:
    graph, (left, right) = _graph_with_texts("ab", "ab")
    contexts = (
        ContextView(target=left, view=SourceView(source=right, start=0, end=2)),
        ContextView(target=right, view=SourceView(source=left, start=0, end=2)),
        ContextView(target=left, view=SourceView(source=left, start=0, end=2)),
    )
    validated = _validate(graph, (left, right), contexts=contexts)
    assert validated.contexts == frozenset(contexts)

    dependencies = (
        DatumDependency(prerequisite=left, dependent=right),
        DatumDependency(prerequisite=right, dependent=left),
    )
    with pytest.raises(ContractViolation) as cycle:
        _validate(graph, (left, right), dependencies=dependencies)
    assert cycle.value.code is ValidationCode.CYCLE


def test_source_chain_resolves_exact_root_range_and_detects_output_overlap() -> None:
    graph, (root, middle, leaf, adjacent) = _graph_with_texts("abcdef", "bcde", "cd", "f")
    sources = (
        SourceRelation(derived=middle, view=SourceView(source=root, start=1, end=5)),
        SourceRelation(derived=leaf, view=SourceView(source=middle, start=1, end=3)),
        SourceRelation(derived=adjacent, view=SourceView(source=root, start=5, end=6)),
    )
    validated = _validate(graph, (leaf, adjacent), source_relations=sources)
    assert validated.effective_ownership == frozenset(
        {
            OwnershipRange(target=leaf, source=root, start=2, end=4),
            OwnershipRange(target=adjacent, source=root, start=5, end=6),
        }
    )

    with pytest.raises(ContractViolation) as overlap:
        _validate(graph, (root, leaf), source_relations=sources)
    assert overlap.value.code is ValidationCode.OVERLAP


def test_empty_target_has_legal_empty_ownership_but_explicit_regions_are_positive_width() -> None:
    graph, (empty,) = _graph_with_texts("")
    validated = _validate(graph, (empty,))
    assert validated.effective_ownership == frozenset({OwnershipRange(target=empty, source=empty, start=0, end=0)})
    with pytest.raises(ContractViolation) as invalid_range:
        _validate(
            graph,
            (empty,),
            output_regions=(OutputRegion(target=empty, source=empty, start=0, end=0),),
        )
    assert invalid_range.value.code is ValidationCode.INVALID_RANGE


def test_group_kinds_normalize_independently_and_input_permutations_are_equal() -> None:
    graph, (first, second, third) = _graph_with_texts("a", "b", "c")
    coherence = (CoherenceScope(members=(first, second)),)
    atomic = (AtomicGroup(members=(second, third)),)
    forward = _validate(graph, (first, second, third), coherence=coherence, atomic=atomic)
    reverse = _validate(
        graph,
        (third, second, first),
        coherence=(CoherenceScope(members=(second, first)),),
        atomic=(AtomicGroup(members=(third, second)),),
    )
    assert forward == reverse
    assert forward.coherence == frozenset({frozenset({first, second}), frozenset({third})})
    assert forward.atomic == frozenset({frozenset({first}), frozenset({second, third})})


def test_bijective_identity_renaming_preserves_normalized_structure() -> None:
    left_graph, (left_root, left_first, left_second) = _graph_with_texts("ab", "a", "b")
    right_graph, (right_second, right_root, right_first) = _graph_with_texts("b", "ab", "a")
    left = _validate(
        left_graph,
        (left_first, left_second),
        source_relations=(
            SourceRelation(derived=left_first, view=SourceView(source=left_root, start=0, end=1)),
            SourceRelation(derived=left_second, view=SourceView(source=left_root, start=1, end=2)),
        ),
        coherence=(CoherenceScope(members=(left_first, left_second)),),
    )
    right = _validate(
        right_graph,
        (right_second, right_first),
        source_relations=(
            SourceRelation(derived=right_second, view=SourceView(source=right_root, start=1, end=2)),
            SourceRelation(derived=right_first, view=SourceView(source=right_root, start=0, end=1)),
        ),
        coherence=(CoherenceScope(members=(right_second, right_first)),),
    )
    rename = {left_root: right_root, left_first: right_first, left_second: right_second}
    assert {rename[target] for target in left.targets} == right.targets
    assert {
        (rename[relation.derived], rename[relation.view.source], relation.view.start, relation.view.end)
        for relation in left.source_relations
    } == {
        (relation.derived, relation.view.source, relation.view.start, relation.view.end)
        for relation in right.source_relations
    }
    assert {frozenset(rename[member] for member in group) for group in left.coherence} == right.coherence


def test_limits_use_raw_counts_and_strict_utf8_bytes() -> None:
    graph, (target,) = _graph_with_texts("é")
    context = ContextView(target=target, view=SourceView(source=target, start=0, end=1))
    exact = DataLimits(
        max_datums=1,
        max_targets=1,
        max_text_bytes=2,
        max_declarations=2,
        max_group_members=0,
    )
    graph.validate(
        targets=(target,),
        source_relations=(),
        contexts=(context, context),
        dependencies=(),
        coherence=(),
        atomic=(),
        output_regions=(),
        limits=exact,
    )
    for field in (
        "max_datums",
        "max_targets",
        "max_text_bytes",
        "max_declarations",
    ):
        values = {
            "max_datums": 1,
            "max_targets": 1,
            "max_text_bytes": 2,
            "max_declarations": 2,
            "max_group_members": 0,
        }
        values[field] -= 1
        with pytest.raises(ContractViolation) as exceeded:
            graph.validate(
                targets=(target,),
                source_relations=(),
                contexts=(context, context),
                dependencies=(),
                coherence=(),
                atomic=(),
                output_regions=(),
                limits=DataLimits(**values),
            )
        assert exceeded.value.code is ValidationCode.LIMIT_EXCEEDED

    surrogate_graph, _ = _graph_with_texts("\ud800")
    with pytest.raises(ContractViolation) as unencodable:
        surrogate_graph.validate(
            targets=(),
            source_relations=(),
            contexts=(),
            dependencies=(),
            coherence=(),
            atomic=(),
            output_regions=(),
            limits=DataLimits(
                max_datums=1,
                max_targets=0,
                max_text_bytes=10,
                max_declarations=0,
                max_group_members=0,
            ),
        )
    assert unencodable.value.code is ValidationCode.INVALID_VALUE


def test_validated_boundary_is_immutable_copy_stable_and_not_replaceable() -> None:
    graph, (target,) = _graph_with_texts("private-payload")
    validated = _validate(graph, (target,))
    assert copy.copy(validated) is validated
    assert copy.deepcopy(validated) is validated
    assert "private-payload" not in repr(validated)
    with pytest.raises(FrozenInstanceError):
        setattr(validated, "targets", frozenset())
    with pytest.raises(TypeError, match="created by DataGraph.validate"):
        ValidatedDataGraph(
            object(),
            graph=validated.graph,
            datums=validated.datums,
            targets=validated.targets,
            source_relations=validated.source_relations,
            contexts=validated.contexts,
            dependencies=validated.dependencies,
            coherence=validated.coherence,
            atomic=validated.atomic,
            output_regions=validated.output_regions,
            effective_ownership=validated.effective_ownership,
            limits=validated.limits,
        )
    with pytest.raises(TypeError):
        replace(validated)


def test_replacement_revalidates_graph_and_ownership_values() -> None:
    graph, (target,) = _graph_with_texts("x")
    with pytest.raises(ContractViolation) as duplicate:
        replace(graph, datums=(*graph.datums, graph.datums[0]))
    assert duplicate.value.code is ValidationCode.DUPLICATE
    ownership = OwnershipRange(target=target, source=target, start=0, end=1)
    with pytest.raises(ContractViolation) as invalid_range:
        replace(ownership, start=2)
    assert invalid_range.value.code is ValidationCode.INVALID_RANGE
    with pytest.raises(ContractViolation) as invalid_type:
        replace(ownership, end=cast(Any, True))
    assert invalid_type.value.code is ValidationCode.INVALID_TYPE


def test_legal_deep_source_chains_are_iterative() -> None:
    graph, identifiers = _graph_with_texts(*("x" for _ in range(2500)))
    sources = tuple(
        SourceRelation(derived=identifiers[index], view=SourceView(source=identifiers[index - 1], start=0, end=1))
        for index in range(1, len(identifiers))
    )
    validated = _validate(graph, (identifiers[-1],), source_relations=sources)
    assert validated.effective_ownership == frozenset(
        {OwnershipRange(target=identifiers[-1], source=identifiers[0], start=0, end=1)}
    )


def test_foreign_equal_text_identity_is_not_local_membership() -> None:
    graph, (local,) = _graph_with_texts("same")
    foreign = DatumId.new(graph=GraphId.new())
    with pytest.raises(ContractViolation) as rejected:
        _validate(graph, (foreign,))
    assert rejected.value.code is ValidationCode.FOREIGN_OWNER
    assert local != foreign
