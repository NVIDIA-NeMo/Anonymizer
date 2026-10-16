# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable data graph declarations and graph-owned validation."""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, NoReturn

from anonymizer.graph._values import ContractViolation, DatumId, GraphId, ValidationCode


class _PrivateRepr:
    __slots__ = ()

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"


def _reject(code: ValidationCode) -> NoReturn:
    raise ContractViolation(code)


def _require_offset(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        _reject(ValidationCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class Datum(_PrivateRepr):
    """One text value with an opaque graph-owned identity."""

    id: DatumId
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, DatumId) or not isinstance(self.text, str):
            _reject(ValidationCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class SourceView(_PrivateRepr):
    """A Unicode code-point view onto a source datum."""

    source: DatumId
    start: int
    end: int

    def __post_init__(self) -> None:
        if not isinstance(self.source, DatumId):
            _reject(ValidationCode.INVALID_TYPE)
        _require_offset(self.start)
        _require_offset(self.end)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class SourceRelation(_PrivateRepr):
    """Declare that a datum is an exact view of another datum."""

    derived: DatumId
    view: SourceView

    def __post_init__(self) -> None:
        if not isinstance(self.derived, DatumId) or not isinstance(self.view, SourceView):
            _reject(ValidationCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ContextView(_PrivateRepr):
    """Declare a read-only source view used as context for a target."""

    target: DatumId
    view: SourceView

    def __post_init__(self) -> None:
        if not isinstance(self.target, DatumId) or not isinstance(self.view, SourceView):
            _reject(ValidationCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class DatumDependency(_PrivateRepr):
    """Declare an ordering dependency between selected targets."""

    prerequisite: DatumId
    dependent: DatumId

    def __post_init__(self) -> None:
        if not isinstance(self.prerequisite, DatumId) or not isinstance(self.dependent, DatumId):
            _reject(ValidationCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class CoherenceScope(_PrivateRepr):
    """Declare targets that must remain mutually coherent."""

    members: tuple[DatumId, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.members, tuple) or any(not isinstance(member, DatumId) for member in self.members):
            _reject(ValidationCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class AtomicGroup(_PrivateRepr):
    """Declare targets whose release is atomic."""

    members: tuple[DatumId, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.members, tuple) or any(not isinstance(member, DatumId) for member in self.members):
            _reject(ValidationCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class OutputRegion(_PrivateRepr):
    """Declare the positive-width source region owned by one target."""

    target: DatumId
    source: DatumId
    start: int
    end: int

    def __post_init__(self) -> None:
        if not isinstance(self.target, DatumId) or not isinstance(self.source, DatumId):
            _reject(ValidationCode.INVALID_TYPE)
        _require_offset(self.start)
        _require_offset(self.end)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class OwnershipRange(_PrivateRepr):
    """A normalized target range on its authoritative root source."""

    target: DatumId
    source: DatumId
    start: int
    end: int

    def __post_init__(self) -> None:
        if not isinstance(self.target, DatumId) or not isinstance(self.source, DatumId):
            _reject(ValidationCode.INVALID_TYPE)
        _require_offset(self.start)
        _require_offset(self.end)
        if self.target.graph != self.source.graph:
            _reject(ValidationCode.FOREIGN_OWNER)
        if not 0 <= self.start <= self.end:
            _reject(ValidationCode.INVALID_RANGE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class DataLimits(_PrivateRepr):
    """Explicit finite admission limits for a data graph."""

    max_datums: int
    max_targets: int
    max_text_bytes: int
    max_declarations: int
    max_group_members: int

    def __post_init__(self) -> None:
        values = (
            self.max_datums,
            self.max_targets,
            self.max_text_bytes,
            self.max_declarations,
            self.max_group_members,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            _reject(ValidationCode.INVALID_TYPE)
        if any(value < 0 for value in values):
            _reject(ValidationCode.INVALID_VALUE)


_VALIDATED_KEY = object()


@dataclass(frozen=True, slots=True, init=False, repr=False)
class ValidatedDataGraph(_PrivateRepr):
    """A graph whose complete declarations passed graph-owned validation."""

    graph: GraphId
    datums: frozenset[Datum]
    targets: frozenset[DatumId]
    source_relations: frozenset[SourceRelation]
    contexts: frozenset[ContextView]
    dependencies: frozenset[DatumDependency]
    coherence: frozenset[frozenset[DatumId]]
    atomic: frozenset[frozenset[DatumId]]
    output_regions: frozenset[OutputRegion]
    effective_ownership: frozenset[OwnershipRange]
    limits: DataLimits

    def __init__(
        self,
        validation_key: object,
        /,
        *,
        graph: GraphId,
        datums: frozenset[Datum],
        targets: frozenset[DatumId],
        source_relations: frozenset[SourceRelation],
        contexts: frozenset[ContextView],
        dependencies: frozenset[DatumDependency],
        coherence: frozenset[frozenset[DatumId]],
        atomic: frozenset[frozenset[DatumId]],
        output_regions: frozenset[OutputRegion],
        effective_ownership: frozenset[OwnershipRange],
        limits: DataLimits,
    ) -> None:
        if validation_key is not _VALIDATED_KEY:
            raise TypeError("validated data graphs are created by DataGraph.validate")
        object.__setattr__(self, "graph", graph)
        object.__setattr__(self, "datums", datums)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "source_relations", source_relations)
        object.__setattr__(self, "contexts", contexts)
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "coherence", coherence)
        object.__setattr__(self, "atomic", atomic)
        object.__setattr__(self, "output_regions", output_regions)
        object.__setattr__(self, "effective_ownership", effective_ownership)
        object.__setattr__(self, "limits", limits)

    def __copy__(self) -> ValidatedDataGraph:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> ValidatedDataGraph:
        del memo
        return self


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class DataGraph(_PrivateRepr):
    """An immutable registry of text datums sharing one graph owner."""

    graph: GraphId
    datums: tuple[Datum, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.graph, GraphId) or not isinstance(self.datums, tuple):
            _reject(ValidationCode.INVALID_TYPE)
        if any(not isinstance(datum, Datum) for datum in self.datums):
            _reject(ValidationCode.INVALID_TYPE)
        if any(datum.id.graph != self.graph for datum in self.datums):
            _reject(ValidationCode.FOREIGN_OWNER)
        identifiers = [datum.id for datum in self.datums]
        if len(identifiers) != len(set(identifiers)):
            _reject(ValidationCode.DUPLICATE)

    @classmethod
    def new(cls) -> DataGraph:
        """Create an empty graph with a fresh owner."""
        return cls(graph=GraphId.new(), datums=())

    def add_text(self, text: str) -> tuple[DataGraph, DatumId]:
        """Return a new version containing a fresh datum for ``text``."""
        identifier = DatumId.new(graph=self.graph)
        datum = Datum(id=identifier, text=text)
        return DataGraph(graph=self.graph, datums=(*self.datums, datum)), identifier

    def validate(
        self,
        *,
        targets: tuple[DatumId, ...],
        source_relations: tuple[SourceRelation, ...],
        contexts: tuple[ContextView, ...],
        dependencies: tuple[DatumDependency, ...],
        coherence: tuple[CoherenceScope, ...],
        atomic: tuple[AtomicGroup, ...],
        output_regions: tuple[OutputRegion, ...],
        limits: DataLimits,
    ) -> ValidatedDataGraph:
        """Validate and normalize declarations without allocating identities."""
        _check_input_types(
            targets,
            source_relations,
            contexts,
            dependencies,
            coherence,
            atomic,
            output_regions,
            limits,
        )
        texts = {datum.id: datum.text for datum in self.datums}
        target_set = frozenset(targets)
        required_targets = _required_targets(contexts, dependencies, coherence, atomic, output_regions)
        text_bytes = _text_bytes(self.datums)
        if any(not group.members for group in (*coherence, *atomic)) or any(
            identifier in texts and identifier not in target_set for identifier in required_targets
        ):
            _reject(ValidationCode.INVALID_VALUE)
        _check_limits(
            self.datums,
            targets,
            source_relations,
            contexts,
            dependencies,
            coherence,
            atomic,
            output_regions,
            limits,
            text_bytes,
        )
        relation_ids = _relation_ids(source_relations, contexts, dependencies, coherence, atomic, output_regions)
        all_ids = (*targets, *relation_ids)
        if any(identifier.graph != self.graph for identifier in all_ids):
            _reject(ValidationCode.FOREIGN_OWNER)
        if len(targets) != len(target_set):
            _reject(ValidationCode.DUPLICATE)
        _check_declaration_duplicates(source_relations, coherence, atomic, output_regions)
        if any(identifier not in texts for identifier in all_ids):
            _reject(ValidationCode.MISSING)
        _check_ranges(texts, source_relations, contexts, output_regions)

        source_by_derived = {relation.derived: relation.view for relation in source_relations}
        region_targets = {region.target for region in output_regions}
        if any(
            relation.derived in target_set
            and relation.derived not in region_targets
            and relation.view.start == relation.view.end
            for relation in source_relations
        ):
            _reject(ValidationCode.INVALID_RANGE)
        _check_group_overlap(coherence)
        _check_group_overlap(atomic)
        ownership_candidates = tuple(
            _resolve_ownership_if_acyclic(target, texts, source_by_derived, output_regions) for target in target_set
        )
        _check_ownership_overlap(frozenset(item for item in ownership_candidates if item is not None))
        source_cycle = _has_cycle((relation.derived, relation.view.source) for relation in source_relations)
        dependency_cycle = _has_cycle((dependency.prerequisite, dependency.dependent) for dependency in dependencies)
        if source_cycle or dependency_cycle:
            _reject(ValidationCode.CYCLE)
        _check_slices(texts, source_relations, output_regions)
        ownership = frozenset(item for item in ownership_candidates if item is not None)

        return ValidatedDataGraph(
            _VALIDATED_KEY,
            graph=self.graph,
            datums=frozenset(self.datums),
            targets=target_set,
            source_relations=frozenset(source_relations),
            contexts=frozenset(contexts),
            dependencies=frozenset(dependencies),
            coherence=_normalize_groups(target_set, coherence),
            atomic=_normalize_groups(target_set, atomic),
            output_regions=frozenset(output_regions),
            effective_ownership=ownership,
            limits=limits,
        )


def _check_input_types(
    targets: object,
    source_relations: object,
    contexts: object,
    dependencies: object,
    coherence: object,
    atomic: object,
    output_regions: object,
    limits: object,
) -> None:
    expected: tuple[tuple[object, type[object]], ...] = (
        (targets, DatumId),
        (source_relations, SourceRelation),
        (contexts, ContextView),
        (dependencies, DatumDependency),
        (coherence, CoherenceScope),
        (atomic, AtomicGroup),
        (output_regions, OutputRegion),
    )
    if any(not _is_tuple_of(values, value_type) for values, value_type in expected) or not isinstance(
        limits, DataLimits
    ):
        _reject(ValidationCode.INVALID_TYPE)


def _is_tuple_of(values: object, value_type: type[object]) -> bool:
    return isinstance(values, tuple) and all(isinstance(value, value_type) for value in values)


def _check_limits(
    datums: tuple[Datum, ...],
    targets: tuple[DatumId, ...],
    source_relations: tuple[SourceRelation, ...],
    contexts: tuple[ContextView, ...],
    dependencies: tuple[DatumDependency, ...],
    coherence: tuple[CoherenceScope, ...],
    atomic: tuple[AtomicGroup, ...],
    output_regions: tuple[OutputRegion, ...],
    limits: DataLimits,
    text_bytes: int,
) -> None:
    declarations = (
        len(source_relations) + len(contexts) + len(dependencies) + len(coherence) + len(atomic) + len(output_regions)
    )
    group_members = sum(len(group.members) for group in (*coherence, *atomic))
    counts = (len(datums), len(targets), text_bytes, declarations, group_members)
    maxima = (
        limits.max_datums,
        limits.max_targets,
        limits.max_text_bytes,
        limits.max_declarations,
        limits.max_group_members,
    )
    if any(count > maximum for count, maximum in zip(counts, maxima, strict=True)):
        _reject(ValidationCode.LIMIT_EXCEEDED)


def _text_bytes(datums: tuple[Datum, ...]) -> int:
    total = 0
    for datum in datums:
        try:
            size = len(datum.text.encode("utf-8", "strict"))
        except UnicodeEncodeError:
            size = None
        if size is None:
            _reject(ValidationCode.INVALID_VALUE)
        total += size
    return total


def _relation_ids(
    source_relations: tuple[SourceRelation, ...],
    contexts: tuple[ContextView, ...],
    dependencies: tuple[DatumDependency, ...],
    coherence: tuple[CoherenceScope, ...],
    atomic: tuple[AtomicGroup, ...],
    output_regions: tuple[OutputRegion, ...],
) -> tuple[DatumId, ...]:
    identifiers: list[DatumId] = []
    for relation in source_relations:
        identifiers.extend((relation.derived, relation.view.source))
    for context in contexts:
        identifiers.extend((context.target, context.view.source))
    for dependency in dependencies:
        identifiers.extend((dependency.prerequisite, dependency.dependent))
    for group in (*coherence, *atomic):
        identifiers.extend(group.members)
    for region in output_regions:
        identifiers.extend((region.target, region.source))
    return tuple(identifiers)


def _required_targets(
    contexts: tuple[ContextView, ...],
    dependencies: tuple[DatumDependency, ...],
    coherence: tuple[CoherenceScope, ...],
    atomic: tuple[AtomicGroup, ...],
    output_regions: tuple[OutputRegion, ...],
) -> tuple[DatumId, ...]:
    required = [context.target for context in contexts]
    required.extend(
        identifier for dependency in dependencies for identifier in (dependency.prerequisite, dependency.dependent)
    )
    required.extend(identifier for group in (*coherence, *atomic) for identifier in group.members)
    required.extend(region.target for region in output_regions)
    return tuple(required)


def _check_declaration_duplicates(
    source_relations: tuple[SourceRelation, ...],
    coherence: tuple[CoherenceScope, ...],
    atomic: tuple[AtomicGroup, ...],
    output_regions: tuple[OutputRegion, ...],
) -> None:
    source_by_derived: dict[DatumId, SourceRelation] = {}
    for relation in source_relations:
        prior = source_by_derived.get(relation.derived)
        if prior is not None and prior != relation:
            _reject(ValidationCode.DUPLICATE)
        source_by_derived[relation.derived] = relation
    region_targets = [region.target for region in output_regions]
    if len(region_targets) != len(set(region_targets)):
        _reject(ValidationCode.DUPLICATE)
    if any(len(group.members) != len(set(group.members)) for group in (*coherence, *atomic)):
        _reject(ValidationCode.DUPLICATE)


def _check_ranges(
    texts: dict[DatumId, str],
    source_relations: tuple[SourceRelation, ...],
    contexts: tuple[ContextView, ...],
    output_regions: tuple[OutputRegion, ...],
) -> None:
    views = [relation.view for relation in source_relations]
    views.extend(context.view for context in contexts)
    if any(not 0 <= view.start <= view.end <= len(texts[view.source]) for view in views):
        _reject(ValidationCode.INVALID_RANGE)
    if any(not 0 <= region.start < region.end <= len(texts[region.source]) for region in output_regions):
        _reject(ValidationCode.INVALID_RANGE)


def _check_group_overlap(groups: tuple[CoherenceScope, ...] | tuple[AtomicGroup, ...]) -> None:
    unique = {frozenset(group.members) for group in groups}
    for left, right in itertools.combinations(unique, 2):
        if left & right:
            _reject(ValidationCode.OVERLAP)


def _has_cycle(edges: Iterable[tuple[DatumId, DatumId]]) -> bool:
    adjacency: dict[DatumId, set[DatumId]] = {}
    for start, end in edges:
        adjacency.setdefault(start, set()).add(end)
    complete: set[DatumId] = set()
    for origin in tuple(adjacency):
        if origin in complete:
            continue
        active: set[DatumId] = set()
        stack: list[tuple[DatumId, bool]] = [(origin, False)]
        while stack:
            node, leaving = stack.pop()
            if leaving:
                active.remove(node)
                complete.add(node)
                continue
            if node in active:
                return True
            if node in complete:
                continue
            active.add(node)
            stack.append((node, True))
            stack.extend((successor, False) for successor in adjacency.get(node, ()))
    return False


def _resolve_ownership_if_acyclic(
    target: DatumId,
    texts: dict[DatumId, str],
    source_by_derived: dict[DatumId, SourceView],
    output_regions: tuple[OutputRegion, ...],
) -> OwnershipRange | None:
    explicit = next((region for region in output_regions if region.target == target), None)
    if explicit is None:
        view = source_by_derived.get(target)
        if view is None:
            source, start, end = target, 0, len(texts[target])
        else:
            source, start, end = view.source, view.start, view.end
    else:
        source, start, end = explicit.source, explicit.start, explicit.end
    visited: set[DatumId] = set()
    while source in source_by_derived:
        if source in visited:
            return None
        visited.add(source)
        view = source_by_derived[source]
        start += view.start
        end += view.start
        source = view.source
    return OwnershipRange(target=target, source=source, start=start, end=end)


def _check_ownership_overlap(ownership: frozenset[OwnershipRange]) -> None:
    for left, right in itertools.combinations(ownership, 2):
        if left.source == right.source and max(left.start, right.start) < min(left.end, right.end):
            _reject(ValidationCode.OVERLAP)


def _check_slices(
    texts: dict[DatumId, str],
    source_relations: tuple[SourceRelation, ...],
    output_regions: tuple[OutputRegion, ...],
) -> None:
    source_by_derived = {relation.derived: relation.view for relation in source_relations}
    for relation in source_relations:
        view = relation.view
        if texts[relation.derived] != texts[view.source][view.start : view.end]:
            _reject(ValidationCode.CONTRADICTORY)
    for region in output_regions:
        if texts[region.target] != texts[region.source][region.start : region.end]:
            _reject(ValidationCode.CONTRADICTORY)
        view = source_by_derived.get(region.target)
        if view is not None and (view.source, view.start, view.end) != (region.source, region.start, region.end):
            _reject(ValidationCode.CONTRADICTORY)


def _normalize_groups(
    targets: frozenset[DatumId], groups: tuple[CoherenceScope, ...] | tuple[AtomicGroup, ...]
) -> frozenset[frozenset[DatumId]]:
    normalized = {frozenset(group.members) for group in groups}
    mentioned = set().union(*normalized) if normalized else set()
    normalized.update(frozenset({target}) for target in targets - mentioned)
    return frozenset(normalized)
