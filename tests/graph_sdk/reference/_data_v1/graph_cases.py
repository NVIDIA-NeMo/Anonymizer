# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent data_v1 reference: graph cases."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator, Sequence
from copy import deepcopy
from typing import cast

from tests.graph_sdk.reference._data_v1.model import (
    LIMIT_KEYS,
    DependencyEnvelope,
    GroupEnvelope,
    JsonObject,
    JsonValue,
    Ref,
    _as_list,
    _as_object,
)
from tests.graph_sdk.reference._data_v1.record_cases import (
    _record_cases,
    _record_precedence_cases,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _context_json,
    _datum_json,
    _dependency_json,
    _json_ref,
    _raw_counts,
    _region_json,
    _source_json,
)


def _base_cases() -> Iterator[tuple[str, JsonObject, str]]:
    yield from _identity_cases()
    yield from _context_cases()
    yield from _dependency_cases()
    yield from _group_cases()
    yield from _range_cases()
    yield from _separation_cases()
    yield from _ownership_negative_cases()
    yield from _encoding_cases()
    yield from _precedence_cases()
    yield from _record_cases()
    yield from _record_precedence_cases()


def _data(datums: Sequence[tuple[Ref, str]], targets: Sequence[Ref] = ()) -> JsonObject:
    declaration: JsonObject = {
        "kind": "data",
        "datums": [_datum_json(identifier, text) for identifier, text in datums],
        "targets": [_json_ref(identifier) for identifier in targets],
        "source_relations": [],
        "contexts": [],
        "dependencies": [],
        "coherence": [],
        "atomic": [],
        "output_regions": [],
        "limits": {key: 1_000_000 for key in LIMIT_KEYS},
    }
    return declaration


def _identity_cases() -> Iterator[tuple[str, JsonObject, str]]:
    for count in range(4):
        identifiers = [("local", index) for index in range(count)]
        for texts in itertools.product(("x", "y"), repeat=count):
            datums = list(zip(identifiers, texts, strict=True))
            for mask in range(1 << count):
                targets = [identifier for index, identifier in enumerate(identifiers) if mask & (1 << index)]
                yield "identity", _data(datums, targets), ""
    yield (
        "identity",
        _data([(("local", 0), "x"), (("local", 1), "x")], [("local", 0), ("local", 1)]),
        "equal-text-distinct",
    )


def _directed_pairs(count: int) -> list[tuple[Ref, Ref]]:
    identifiers = [("local", index) for index in range(count)]
    return [(left, right) for left in identifiers for right in identifiers if left != right]


def _context_cases() -> Iterator[tuple[str, JsonObject, str]]:
    for count in range(4):
        datums = [(("local", index), "x") for index in range(count)]
        identifiers = [identifier for identifier, _ in datums]
        pairs = _directed_pairs(count)
        for mask in range(1 << len(pairs)):
            declaration = _data(datums, identifiers)
            declaration["contexts"] = [
                _context_json((target, source, 0, 1))
                for index, (target, source) in enumerate(pairs)
                if mask & (1 << index)
            ]
            yield "context", declaration, ""
    declaration = _data([(("local", 0), "x")], [("local", 0)])
    declaration["contexts"] = [_context_json((("local", 0), ("local", 0), 0, 1))]
    yield "context", declaration, "self"


def _dependency_cases() -> Iterator[tuple[str, JsonObject, str]]:
    for count in range(4):
        datums = [(("local", index), "x") for index in range(count)]
        identifiers = [identifier for identifier, _ in datums]
        pairs = _directed_pairs(count)
        for mask in range(1 << len(pairs)):
            declaration = _data(datums, identifiers)
            declaration["dependencies"] = [
                _dependency_json(pair) for index, pair in enumerate(pairs) if mask & (1 << index)
            ]
            yield "dependency", declaration, ""
    declaration = _data([(("local", 0), "x")], [("local", 0)])
    declaration["dependencies"] = [_dependency_json((("local", 0), ("local", 0)))]
    yield "dependency", declaration, "self"


def _partitions(items: tuple[Ref, ...]) -> tuple[tuple[tuple[Ref, ...], ...], ...]:
    if not items:
        return ((),)
    first, *rest = items
    result: list[tuple[tuple[Ref, ...], ...]] = []
    for partition in _partitions(tuple(rest)):
        result.append(((first,), *partition))
        for index in range(len(partition)):
            merged = tuple(sorted((first, *partition[index])))
            result.append((*partition[:index], merged, *partition[index + 1 :]))
    unique = {tuple(sorted(partition)) for partition in result}
    return tuple(sorted(unique))


def _group_json(members: Iterable[Ref]) -> GroupEnvelope:
    return {"members": [_json_ref(member) for member in members]}


def _group_cases() -> Iterator[tuple[str, JsonObject, str]]:
    for count in range(4):
        datums = [(("local", index), "x") for index in range(count)]
        identifiers = tuple(identifier for identifier, _ in datums)
        for size in range(count + 1):
            for targets in itertools.combinations(identifiers, size):
                for coherence, atomic in itertools.product(_partitions(targets), repeat=2):
                    declaration = _data(datums, targets)
                    declaration["coherence"] = [_group_json(group) for group in coherence]
                    declaration["atomic"] = [_group_json(group) for group in atomic]
                    yield "groups", declaration, ""
    base = _data([(("local", index), "x") for index in range(3)], [("local", index) for index in range(3)])
    specials = (
        ("identical", "coherence", [[("local", 0), ("local", 1)], [("local", 0), ("local", 1)]]),
        ("duplicate-member", "coherence", [[("local", 0), ("local", 0)]]),
        ("empty", "coherence", [[]]),
        ("coherence-partial", "coherence", [[("local", 0), ("local", 1)], [("local", 1), ("local", 2)]]),
        ("coherence-subset", "coherence", [[("local", 0), ("local", 1)], [("local", 0)]]),
        ("atomic-partial", "atomic", [[("local", 0), ("local", 1)], [("local", 1), ("local", 2)]]),
        ("atomic-subset", "atomic", [[("local", 0), ("local", 1)], [("local", 0)]]),
        ("foreign", "coherence", [[("foreign", 0)]]),
        ("missing", "coherence", [[("local", 99)]]),
    )
    for label, key, groups in specials:
        declaration = deepcopy(base)
        declaration[key] = [_group_json(group) for group in groups]
        yield "groups", declaration, label
    declaration = deepcopy(base)
    declaration["targets"] = [_json_ref(("local", 0)), _json_ref(("local", 1))]
    declaration["coherence"] = [_group_json([("local", 2)])]
    yield "groups", declaration, "nontarget"


def _range_cases() -> Iterator[tuple[str, JsonObject, str]]:
    intervals = [(start, end) for start in range(4) for end in range(start + 1, 5)]
    for left, right in itertools.product(intervals, repeat=2):
        source = ("local", 0)
        read = _data([(source, "abcd"), (("local", 1), "x"), (("local", 2), "x")], [("local", 1), ("local", 2)])
        read["contexts"] = [
            _context_json((("local", 1), source, *left)),
            _context_json((("local", 2), source, *right)),
        ]
        yield "ranges", read, "reads"
        owned = _data(
            [(source, "abcd"), (("local", 1), "abcd"[slice(*left)]), (("local", 2), "abcd"[slice(*right)])],
            [("local", 1), ("local", 2)],
        )
        owned["source_relations"] = [
            _source_json((("local", 1), source, *left)),
            _source_json((("local", 2), source, *right)),
        ]
        yield "ranges", owned, "outputs"
    yield from _range_specials()


def _range_specials() -> Iterator[tuple[str, JsonObject, str]]:
    base = _data([(("local", 0), "abcd"), (("local", 1), "a")], [("local", 1)])
    for label, start, end in (("negative", -1, 1), ("past-end", 0, 5), ("reversed", 2, 1)):
        declaration = deepcopy(base)
        declaration["contexts"] = [_context_json((("local", 1), ("local", 0), start, end))]
        yield "ranges", declaration, label
    for label, value in (("noninteger", "0"), ("bool-offset", True)):
        declaration = deepcopy(base)
        declaration["contexts"] = [{"target": ["local", 1], "source": ["local", 0], "start": value, "end": 1}]
        yield "ranges", declaration, label
    declaration = deepcopy(base)
    declaration["output_regions"] = [_region_json((("local", 1), ("local", 0), 0, 0))]
    yield "ranges", declaration, "explicit-zero"
    declaration = deepcopy(base)
    declaration["source_relations"] = [_source_json((("local", 1), ("local", 0), 1, 2))]
    yield "ranges", declaration, "wrong-slice"
    yield "ranges", _data([(("local", 0), "")], [("local", 0)]), "empty-whole"
    declaration = _data([(("local", 0), ""), (("local", 1), "x")], [("local", 1)])
    declaration["contexts"] = [_context_json((("local", 1), ("local", 0), 0, 0))]
    yield "ranges", declaration, "empty-read"
    declaration = _data([(("local", 0), "abcd"), (("local", 1), "ab")], [("local", 0), ("local", 1)])
    declaration["source_relations"] = [_source_json((("local", 1), ("local", 0), 0, 2))]
    yield "ranges", declaration, "whole-plus-subview"
    declaration = _data(
        [(("local", 0), "abcd"), (("local", 1), "abc"), (("local", 2), "bc"), (("local", 3), "bc")],
        [("local", 2), ("local", 3)],
    )
    declaration["source_relations"] = [
        _source_json((("local", 1), ("local", 0), 0, 3)),
        _source_json((("local", 2), ("local", 1), 1, 3)),
        _source_json((("local", 3), ("local", 0), 1, 3)),
    ]
    yield "ranges", declaration, "nested-overlap"
    declaration = _data([(("local", 0), "x"), (("local", 1), "x")], [("local", 0), ("local", 1)])
    declaration["source_relations"] = [
        _source_json((("local", 0), ("local", 1), 0, 1)),
        _source_json((("local", 1), ("local", 0), 0, 1)),
    ]
    yield "ranges", declaration, "source-cycle"
    declaration = _data([(("local", 0), "x")], [("local", 0)])
    declaration["source_relations"] = [_source_json((("local", 0), ("local", 0), 0, 2))]
    _as_object(declaration["limits"]).update(_raw_counts(declaration))
    yield "ranges", declaration, "self-source-invalid-range-before-cycle"
    declaration = deepcopy(base)
    declaration["contexts"] = [_context_json((("local", 1), ("foreign", 0), 0, 1))]
    yield "ranges", declaration, "foreign-source"


def _separation_cases() -> Iterator[tuple[str, JsonObject, str]]:
    datums = [(("local", index), "x") for index in range(3)]
    targets = [("local", index) for index in range(3)]
    singleton = [[target] for target in targets]
    joined = [[targets[0], targets[1]], [targets[2]]]
    for coherence, atomic in itertools.product((singleton, joined), repeat=2):
        declaration = _data(datums, targets)
        declaration["contexts"] = [
            _context_json((targets[0], targets[1], 0, 1)),
            _context_json((targets[1], targets[0], 0, 1)),
        ]
        declaration["coherence"] = [_group_json(group) for group in coherence]
        declaration["atomic"] = [_group_json(group) for group in atomic]
        yield "relation-separation", declaration, ""
    declaration = _data(datums, targets)
    declaration["dependencies"] = [_dependency_json((targets[0], targets[1]))]
    yield "relation-separation", declaration, "dependency"
    declaration = deepcopy(declaration)
    cast(list[DependencyEnvelope], declaration["dependencies"]).append(_dependency_json((targets[1], targets[0])))
    yield "relation-separation", declaration, "dependency-cycle"


def _ownership_negative_cases() -> Iterator[tuple[str, JsonObject, str]]:
    base = _data([(("local", 0), "x"), (("local", 1), "x")], [("local", 0)])
    modifications: tuple[tuple[str, str, JsonValue], ...] = (
        ("foreign-target", "targets", [["foreign", 0]]),
        ("missing-target", "targets", [["local", 99]]),
        ("duplicate-target", "targets", [["local", 0], ["local", 0]]),
        ("absent-version", "targets", [["local", 2]]),
        (
            "conflicting-source",
            "source_relations",
            [
                _source_json((("local", 0), ("local", 1), 0, 1)),
                _source_json((("local", 0), ("local", 0), 0, 1)),
            ],
        ),
        (
            "duplicate-region",
            "output_regions",
            [
                _region_json((("local", 0), ("local", 1), 0, 1)),
                _region_json((("local", 0), ("local", 0), 0, 1)),
            ],
        ),
        ("foreign-root", "output_regions", [_region_json((("local", 0), ("foreign", 0), 0, 1))]),
    )
    for label, key, value in modifications:
        declaration = deepcopy(base)
        declaration[key] = value
        yield "ownership-negatives", declaration, label
    declaration = deepcopy(base)
    declaration["datums"] = [*_as_list(declaration["datums"]), _datum_json(("local", 0), "x")]
    yield "ownership-negatives", declaration, "duplicate-registry"


def _encoding_cases() -> Iterator[tuple[str, JsonObject, str]]:
    for label, text in (("empty", ""), ("e-acute", "é"), ("emoji", "😀")):
        yield "encoding-bounds", _data([(("local", 0), text)], [("local", 0)]), label
    yield "encoding-bounds", _data([(("local", 0), "x"), (("local", 1), "x")], [("local", 0)]), "repeated"
    yield "encoding-bounds", _data([(("local", 0), "\ud800")], []), "surrogate"
    for key, value, label in (("max_datums", -1, "negative-limit"), ("max_targets", True, "bool-limit")):
        declaration = _data([], [])
        cast(JsonObject, declaration["limits"])[key] = value
        yield "encoding-bounds", declaration, label
    yield "encoding-bounds", _data([], []), "zero-counts"


def _precedence_cases() -> Iterator[tuple[str, JsonObject, str]]:
    targets = [("local", index) for index in range(3)]
    base = _data([(target, "x") for target in targets], targets)
    cases: list[JsonObject] = []
    case = deepcopy(base)
    cast(JsonObject, case["limits"])["max_datums"] = True
    cast(JsonObject, case["limits"])["max_targets"] = -1
    cases.append(case)
    case = deepcopy(base)
    cast(JsonObject, case["limits"])["max_targets"] = -1
    cast(JsonObject, case["limits"])["max_datums"] = 2
    cases.append(case)
    case = deepcopy(base)
    cast(JsonObject, case["limits"])["max_datums"] = 2
    case["targets"] = [["foreign", 0]]
    cases.append(case)
    case = deepcopy(base)
    case["targets"] = [["local", 0], ["local", 0], ["foreign", 0]]
    cases.append(case)
    case = deepcopy(base)
    case["targets"] = [["local", 0], ["local", 0], ["local", 99]]
    cases.append(case)
    case = deepcopy(base)
    case["targets"] = [["local", 0], ["local", 1], ["local", 99]]
    case["contexts"] = [_context_json((targets[0], targets[1], 0, 2))]
    cases.append(case)
    case = deepcopy(base)
    case["contexts"] = [_context_json((targets[0], targets[1], 0, 2))]
    case["coherence"] = [_group_json(targets[:2]), _group_json(targets[1:])]
    cases.append(case)
    case = deepcopy(base)
    case["coherence"] = [_group_json(targets[:2]), _group_json(targets[1:])]
    case["dependencies"] = [_dependency_json((targets[0], targets[1])), _dependency_json((targets[1], targets[0]))]
    cases.append(case)
    case = _data([(targets[0], "x"), (targets[1], "x"), (targets[2], "y")], targets[:2])
    case["source_relations"] = [_source_json((targets[0], targets[2], 0, 1))]
    case["dependencies"] = [_dependency_json((targets[0], targets[1])), _dependency_json((targets[1], targets[0]))]
    cases.append(case)
    case = deepcopy(base)
    case["dependencies"] = [_dependency_json((targets[0], ("local", 99))), _dependency_json((targets[1], targets[1]))]
    cases.append(case)
    case = deepcopy(base)
    case["contexts"] = [_context_json((targets[0], targets[2], 0, 2))]
    case["dependencies"] = [_dependency_json((targets[0], targets[1])), _dependency_json((targets[1], targets[0]))]
    cases.append(case)
    for index, declaration in enumerate(cases, 1):
        yield "precedence", declaration, f"p{index}"
