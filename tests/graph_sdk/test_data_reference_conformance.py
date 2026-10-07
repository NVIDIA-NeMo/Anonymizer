# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conformance of product data graphs to the frozen neutral reference."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from anonymizer.engine.graph_sdk.data import (
    AtomicGroup,
    CoherenceScope,
    ContextView,
    DataGraph,
    DataLimits,
    Datum,
    DatumDependency,
    OutputRegion,
    SourceRelation,
    SourceView,
    ValidatedDataGraph,
)
from anonymizer.graph._values import ContractViolation, DatumId, GraphId, ValidationCode
from tests.graph_sdk.reference import data_v1

_CORPUS_PATH = Path(__file__).parent / "reference/data_v1_cases.json"


class _DataReferenceAdapter:
    """Map neutral fixture labels to fresh nominal product identities."""

    def __init__(self) -> None:
        self._graph = GraphId.new()
        self._foreign_graph = GraphId.new()
        self._identities: dict[tuple[str, int], DatumId] = {}
        self._labels: dict[DatumId, tuple[str, int]] = {}

    def validate(self, declaration: data_v1.DataDeclaration) -> ValidatedDataGraph:
        datums = tuple(self._datum(value) for value in declaration["datums"])
        graph = DataGraph(graph=self._graph, datums=datums)
        targets = tuple(self._identity(value) for value in declaration["targets"])
        source_relations = tuple(self._source(value) for value in declaration["source_relations"])
        contexts = tuple(self._context(value) for value in declaration["contexts"])
        dependencies = tuple(self._dependency(value) for value in declaration["dependencies"])
        coherence = tuple(self._coherence(value) for value in declaration["coherence"])
        atomic = tuple(self._atomic(value) for value in declaration["atomic"])
        output_regions = tuple(self._region(value) for value in declaration["output_regions"])
        raw_limits = declaration["limits"]
        limits = DataLimits(
            max_datums=cast(int, raw_limits["max_datums"]),
            max_targets=cast(int, raw_limits["max_targets"]),
            max_text_bytes=cast(int, raw_limits["max_text_bytes"]),
            max_declarations=cast(int, raw_limits["max_declarations"]),
            max_group_members=cast(int, raw_limits["max_group_members"]),
        )
        return graph.validate(
            targets=targets,
            source_relations=source_relations,
            contexts=contexts,
            dependencies=dependencies,
            coherence=coherence,
            atomic=atomic,
            output_regions=output_regions,
            limits=limits,
        )

    def normalized(self, graph: ValidatedDataGraph) -> data_v1.JsonObject:
        return {
            "datums": sorted(
                ({"id": self._json_id(datum.id), "text": datum.text} for datum in graph.datums),
                key=lambda value: _json_id_key(value["id"]),
            ),
            "targets": sorted((self._json_id(identifier) for identifier in graph.targets), key=_json_id_key),
            "source_relations": sorted(
                (
                    {
                        "derived": self._json_id(relation.derived),
                        "source": self._json_id(relation.view.source),
                        "start": relation.view.start,
                        "end": relation.view.end,
                    }
                    for relation in graph.source_relations
                ),
                key=_relation_key,
            ),
            "contexts": sorted(
                (
                    {
                        "target": self._json_id(context.target),
                        "source": self._json_id(context.view.source),
                        "start": context.view.start,
                        "end": context.view.end,
                    }
                    for context in graph.contexts
                ),
                key=_relation_key,
            ),
            "dependencies": sorted(
                (
                    {
                        "prerequisite": self._json_id(dependency.prerequisite),
                        "dependent": self._json_id(dependency.dependent),
                    }
                    for dependency in graph.dependencies
                ),
                key=lambda value: (_json_id_key(value["prerequisite"]), _json_id_key(value["dependent"])),
            ),
            "coherence": self._groups(graph.coherence),
            "atomic": self._groups(graph.atomic),
            "output_regions": sorted(
                (
                    {
                        "target": self._json_id(region.target),
                        "source": self._json_id(region.source),
                        "start": region.start,
                        "end": region.end,
                    }
                    for region in graph.output_regions
                ),
                key=_relation_key,
            ),
            "effective_ownership": sorted(
                (
                    {
                        "target": self._json_id(ownership.target),
                        "source": self._json_id(ownership.source),
                        "start": ownership.start,
                        "end": ownership.end,
                    }
                    for ownership in graph.effective_ownership
                ),
                key=_relation_key,
            ),
            "limits": {
                "max_datums": graph.limits.max_datums,
                "max_targets": graph.limits.max_targets,
                "max_text_bytes": graph.limits.max_text_bytes,
                "max_declarations": graph.limits.max_declarations,
                "max_group_members": graph.limits.max_group_members,
            },
        }

    def _identity(self, value: data_v1.JsonValue) -> DatumId:
        if not isinstance(value, list) or len(value) != 2:
            raise ContractViolation(ValidationCode.INVALID_TYPE)
        owner, index = value
        if owner not in ("local", "foreign") or not isinstance(owner, str):
            raise ContractViolation(ValidationCode.INVALID_TYPE)
        if isinstance(index, bool) or not isinstance(index, int):
            raise ContractViolation(ValidationCode.INVALID_TYPE)
        if index < 0:
            raise ContractViolation(ValidationCode.INVALID_VALUE)
        label = owner, index
        if label not in self._identities:
            graph = self._graph if owner == "local" else self._foreign_graph
            identifier = DatumId.new(graph=graph)
            self._identities[label] = identifier
            self._labels[identifier] = label
        return self._identities[label]

    def _json_id(self, identifier: DatumId) -> list[data_v1.JsonValue]:
        owner, index = self._labels[identifier]
        return [owner, index]

    def _datum(self, value: data_v1.DatumEnvelope) -> Datum:
        return Datum(id=self._identity(value["id"]), text=cast(str, value["text"]))

    def _source(self, value: data_v1.SourceRelationEnvelope) -> SourceRelation:
        return SourceRelation(
            derived=self._identity(value["derived"]),
            view=SourceView(
                source=self._identity(value["source"]),
                start=cast(int, value["start"]),
                end=cast(int, value["end"]),
            ),
        )

    def _context(self, value: data_v1.ContextEnvelope) -> ContextView:
        return ContextView(
            target=self._identity(value["target"]),
            view=SourceView(
                source=self._identity(value["source"]),
                start=cast(int, value["start"]),
                end=cast(int, value["end"]),
            ),
        )

    def _dependency(self, value: data_v1.DependencyEnvelope) -> DatumDependency:
        return DatumDependency(
            prerequisite=self._identity(value["prerequisite"]),
            dependent=self._identity(value["dependent"]),
        )

    def _coherence(self, value: data_v1.GroupEnvelope) -> CoherenceScope:
        return CoherenceScope(members=tuple(self._identity(member) for member in _list(value["members"])))

    def _atomic(self, value: data_v1.GroupEnvelope) -> AtomicGroup:
        return AtomicGroup(members=tuple(self._identity(member) for member in _list(value["members"])))

    def _region(self, value: data_v1.OutputRegionEnvelope) -> OutputRegion:
        return OutputRegion(
            target=self._identity(value["target"]),
            source=self._identity(value["source"]),
            start=cast(int, value["start"]),
            end=cast(int, value["end"]),
        )

    def _groups(self, groups: frozenset[frozenset[DatumId]]) -> list[data_v1.JsonValue]:
        normalized: list[list[data_v1.JsonValue]] = [
            sorted((self._json_id(member) for member in group), key=_json_id_key) for group in groups
        ]
        result = sorted(normalized, key=_group_key)
        # Recursive JSON aliases are invariant in mutable lists; the values are
        # fully built and only compared by this read-only fixture adapter.
        return cast(list[data_v1.JsonValue], result)


def _list(value: data_v1.JsonValue) -> list[data_v1.JsonValue]:
    return cast(list[data_v1.JsonValue], value)


def _json_id_key(value: data_v1.JsonValue) -> tuple[str, int]:
    raw = cast(list[data_v1.JsonValue], value)
    return cast(str, raw[0]), cast(int, raw[1])


def _group_key(group: list[data_v1.JsonValue]) -> tuple[tuple[str, int], ...]:
    return tuple(_json_id_key(member) for member in group)


def _relation_key(
    value: Mapping[str, data_v1.JsonValue],
) -> tuple[tuple[str, int], tuple[str, int], int, int]:
    first = value.get("target", value.get("derived"))
    return (
        _json_id_key(cast(data_v1.JsonValue, first)),
        _json_id_key(cast(data_v1.JsonValue, value["source"])),
        cast(int, value["start"]),
        cast(int, value["end"]),
    )


def _adapt(case: data_v1.FixtureCase) -> data_v1.ValidationResult:
    declaration = case["declaration"]
    if declaration["kind"] != "data":
        raise AssertionError("reference adapter received a non-data fixture")
    adapter = _DataReferenceAdapter()
    try:
        graph = adapter.validate(declaration)
    except ContractViolation as error:
        return {"verdict": "reject", "code": error.code.value}
    return {"verdict": "accept", "normalized": adapter.normalized(graph)}


def test_all_frozen_data_cases_match_product_validation_and_normalization() -> None:
    cases = data_v1._parse_cases(json.loads(_CORPUS_PATH.read_bytes()))
    data_cases = [case for case in cases if case["declaration"]["kind"] == "data"]
    assert len(data_cases) == 38_267
    for case in data_cases:
        assert _adapt(case) == case["expected"], case["case_id"]
