# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent activation_v1 reference: facade."""

from __future__ import annotations

import hashlib
import itertools
import platform
from collections.abc import Sequence
from typing import cast

from tests.graph_sdk.reference._activation_v1.aggregate_cases import (
    _aggregate_decl as _aggregate_decl,
)
from tests.graph_sdk.reference._activation_v1.aggregate_cases import (
    _coverage_cases as _coverage_cases,
)
from tests.graph_sdk.reference._activation_v1.aggregate_cases import (
    _join_cases as _join_cases,
)
from tests.graph_sdk.reference._activation_v1.aggregate_cases import (
    _loop_cases as _loop_cases,
)
from tests.graph_sdk.reference._activation_v1.aggregate_cases import (
    _map_cases as _map_cases,
)
from tests.graph_sdk.reference._activation_v1.aggregate_cases import (
    _map_events as _map_events,
)
from tests.graph_sdk.reference._activation_v1.aggregate_cases import (
    _nested_cases as _nested_cases,
)
from tests.graph_sdk.reference._activation_v1.aggregate_cases import (
    _precedence_cases as _precedence_cases,
)
from tests.graph_sdk.reference._activation_v1.builders import (
    _case as _case,
)
from tests.graph_sdk.reference._activation_v1.builders import (
    _decl as _decl,
)
from tests.graph_sdk.reference._activation_v1.builders import (
    _event as _event,
)
from tests.graph_sdk.reference._activation_v1.builders import (
    _ordinary as _ordinary,
)
from tests.graph_sdk.reference._activation_v1.builders import (
    _restrict_loop_outcomes as _restrict_loop_outcomes,
)
from tests.graph_sdk.reference._activation_v1.builders import (
    _seed as _seed,
)
from tests.graph_sdk.reference._activation_v1.builders import (
    _select as _select,
)
from tests.graph_sdk.reference._activation_v1.builders import (
    _spare as _spare,
)
from tests.graph_sdk.reference._activation_v1.builders import (
    _terminal as _terminal,
)
from tests.graph_sdk.reference._activation_v1.evaluation import (
    _state_object as _state_object,
)
from tests.graph_sdk.reference._activation_v1.evaluation import (
    admit_dynamic_workflow as admit_dynamic_workflow,
)
from tests.graph_sdk.reference._activation_v1.evaluation import (
    admit_static_support as admit_static_support,
)
from tests.graph_sdk.reference._activation_v1.evaluation import (
    alpha_normalize_state as alpha_normalize_state,
)
from tests.graph_sdk.reference._activation_v1.evaluation import (
    reduce_trace as reduce_trace,
)
from tests.graph_sdk.reference._activation_v1.model import (
    ACTIVATIONS as ACTIVATIONS,
)
from tests.graph_sdk.reference._activation_v1.model import (
    ALPHABET as ALPHABET,
)
from tests.graph_sdk.reference._activation_v1.model import (
    CATEGORIES as CATEGORIES,
)
from tests.graph_sdk.reference._activation_v1.model import (
    CONSUMPTION_ADDENDUM_SHA256 as CONSUMPTION_ADDENDUM_SHA256,
)
from tests.graph_sdk.reference._activation_v1.model import (
    CONTRACT_SHA256 as CONTRACT_SHA256,
)
from tests.graph_sdk.reference._activation_v1.model import (
    CORPUS_PATH as CORPUS_PATH,
)
from tests.graph_sdk.reference._activation_v1.model import (
    ERROR_ORDER as ERROR_ORDER,
)
from tests.graph_sdk.reference._activation_v1.model import (
    FAMILY_IDS as FAMILY_IDS,
)
from tests.graph_sdk.reference._activation_v1.model import (
    GENERATOR_VERSION as GENERATOR_VERSION,
)
from tests.graph_sdk.reference._activation_v1.model import (
    INVERSE_ACTIVATION as INVERSE_ACTIVATION,
)
from tests.graph_sdk.reference._activation_v1.model import (
    INVERSE_TEMPLATE as INVERSE_TEMPLATE,
)
from tests.graph_sdk.reference._activation_v1.model import (
    INVOCATIONS as INVOCATIONS,
)
from tests.graph_sdk.reference._activation_v1.model import (
    RENAME_ACTIVATION as RENAME_ACTIVATION,
)
from tests.graph_sdk.reference._activation_v1.model import (
    RENAME_TEMPLATE as RENAME_TEMPLATE,
)
from tests.graph_sdk.reference._activation_v1.model import (
    RULE_IDS as RULE_IDS,
)
from tests.graph_sdk.reference._activation_v1.model import (
    SELF_TEST_VERSION as SELF_TEST_VERSION,
)
from tests.graph_sdk.reference._activation_v1.model import (
    SUPPORT_PATH as SUPPORT_PATH,
)
from tests.graph_sdk.reference._activation_v1.model import (
    TEMPLATES as TEMPLATES,
)
from tests.graph_sdk.reference._activation_v1.model import (
    TERMINAL as TERMINAL,
)
from tests.graph_sdk.reference._activation_v1.model import (
    Aggregate as Aggregate,
)
from tests.graph_sdk.reference._activation_v1.model import (
    Boundary as Boundary,
)
from tests.graph_sdk.reference._activation_v1.model import (
    Category as Category,
)
from tests.graph_sdk.reference._activation_v1.model import (
    Choice as Choice,
)
from tests.graph_sdk.reference._activation_v1.model import (
    Close as Close,
)
from tests.graph_sdk.reference._activation_v1.model import (
    Code as Code,
)
from tests.graph_sdk.reference._activation_v1.model import (
    Declaration as Declaration,
)
from tests.graph_sdk.reference._activation_v1.model import (
    Entry as Entry,
)
from tests.graph_sdk.reference._activation_v1.model import (
    Event as Event,
)
from tests.graph_sdk.reference._activation_v1.model import (
    Expansion as Expansion,
)
from tests.graph_sdk.reference._activation_v1.model import (
    Initialize as Initialize,
)
from tests.graph_sdk.reference._activation_v1.model import (
    InputDependency as InputDependency,
)
from tests.graph_sdk.reference._activation_v1.model import (
    Json as Json,
)
from tests.graph_sdk.reference._activation_v1.model import (
    Limits as Limits,
)
from tests.graph_sdk.reference._activation_v1.model import (
    LoopAggregate as LoopAggregate,
)
from tests.graph_sdk.reference._activation_v1.model import (
    MapAggregate as MapAggregate,
)
from tests.graph_sdk.reference._activation_v1.model import (
    Membership as Membership,
)
from tests.graph_sdk.reference._activation_v1.model import (
    Object as Object,
)
from tests.graph_sdk.reference._activation_v1.model import (
    Outcome as Outcome,
)
from tests.graph_sdk.reference._activation_v1.model import (
    Overflow as Overflow,
)
from tests.graph_sdk.reference._activation_v1.model import (
    ReferenceState as ReferenceState,
)
from tests.graph_sdk.reference._activation_v1.model import (
    Rejected as Rejected,
)
from tests.graph_sdk.reference._activation_v1.model import (
    Role as Role,
)
from tests.graph_sdk.reference._activation_v1.model import (
    Seed as Seed,
)
from tests.graph_sdk.reference._activation_v1.model import (
    Select as Select,
)
from tests.graph_sdk.reference._activation_v1.model import (
    Start as Start,
)
from tests.graph_sdk.reference._activation_v1.model import (
    Status as Status,
)
from tests.graph_sdk.reference._activation_v1.model import (
    Subgraph as Subgraph,
)
from tests.graph_sdk.reference._activation_v1.model import (
    TerminalEvent as TerminalEvent,
)
from tests.graph_sdk.reference._activation_v1.model import (
    _canonical as _canonical,
)
from tests.graph_sdk.reference._activation_v1.model import (
    _rename as _rename,
)
from tests.graph_sdk.reference._activation_v1.model import (
    canonical_bytes as canonical_bytes,
)
from tests.graph_sdk.reference._activation_v1.model import (
    completion_reserve as completion_reserve,
)
from tests.graph_sdk.reference._activation_v1.model import (
    initial_capacity as initial_capacity,
)
from tests.graph_sdk.reference._activation_v1.ordinary_cases import (
    _choice_cases as _choice_cases,
)
from tests.graph_sdk.reference._activation_v1.ordinary_cases import (
    _mutation_cases as _mutation_cases,
)
from tests.graph_sdk.reference._activation_v1.ordinary_cases import (
    _sequence_cases as _sequence_cases,
)
from tests.graph_sdk.reference._activation_v1.ordinary_cases import (
    _subgraph_cases as _subgraph_cases,
)
from tests.graph_sdk.reference._activation_v1.parsing import (
    _closed as _closed,
)
from tests.graph_sdk.reference._activation_v1.parsing import (
    _int as _int,
)
from tests.graph_sdk.reference._activation_v1.parsing import (
    _list as _list,
)
from tests.graph_sdk.reference._activation_v1.parsing import (
    _obj as _obj,
)
from tests.graph_sdk.reference._activation_v1.parsing import (
    _parse_loop_binding as _parse_loop_binding,
)
from tests.graph_sdk.reference._activation_v1.parsing import (
    _raw_depth as _raw_depth,
)
from tests.graph_sdk.reference._activation_v1.parsing import (
    _strs as _strs,
)
from tests.graph_sdk.reference._activation_v1.parsing import (
    parse_declaration as parse_declaration,
)
from tests.graph_sdk.reference._activation_v1.parsing import (
    parse_event as parse_event,
)
from tests.graph_sdk.reference._activation_v1.reducer import (
    _apply as _apply,
)
from tests.graph_sdk.reference._activation_v1.reducer import (
    _complete as _complete,
)
from tests.graph_sdk.reference._activation_v1.reducer import (
    _initialize as _initialize,
)
from tests.graph_sdk.reference._activation_v1.reducer import (
    _materialize as _materialize,
)
from tests.graph_sdk.reference._activation_v1.reducer import (
    _normalize as _normalize,
)
from tests.graph_sdk.reference._activation_v1.reducer import (
    _reserve as _reserve,
)
from tests.graph_sdk.reference._activation_v1.reducer import (
    _subgraph_body_closed as _subgraph_body_closed,
)


def generate_cases() -> tuple[Object, ...]:
    cases = tuple(
        itertools.chain(
            _sequence_cases(),
            _mutation_cases(),
            _choice_cases(),
            _subgraph_cases(),
            _map_cases(),
            _join_cases(),
            _loop_cases(),
            _nested_cases(),
            _precedence_cases(),
            _coverage_cases(),
        )
    )
    identifiers = [case["case_id"] for case in cases]
    payloads = [_canonical({key: value for key, value in case.items() if key != "case_id"}) for case in cases]
    if len(identifiers) != len(set(cast(list[str], identifiers))) or len(payloads) != len(set(payloads)):
        raise AssertionError("duplicate finite case")
    return cases


def load_cases(value: Json) -> tuple[Object, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("invalid corpus")
    exact = {"boundary", "case_id", "declaration", "events", "expected", "family", "mode", "traces"}
    cases: list[Object] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != exact:
            raise ValueError("invalid case shape")
        cases.append(cast(Object, item))
    return tuple(cases)


def _labels(value: Json, universe: Sequence[str]) -> list[str]:
    if isinstance(value, str):
        return [value] if value in universe else []
    if isinstance(value, list):
        return list(itertools.chain.from_iterable(_labels(item, universe) for item in value))
    if isinstance(value, dict):
        return list(itertools.chain.from_iterable(_labels(item, universe) for item in value.values()))
    return []


def counts(cases: Sequence[Object]) -> Object:
    return {
        "case_count": len(cases),
        "event_count": sum(
            len(cast(list[Json], case["events"]))
            + sum(len(cast(list[Json], cast(Object, trace)["events"])) for trace in cast(list[Json], case["traces"]))
            for case in cases
        ),
        "max_activations": max(len(set(_labels(case, ACTIVATIONS))) for case in cases),
        "max_dynamic_depth": 2,
        "max_loop_iterations": 3,
        "max_map_children": 3,
        "max_templates": max(len(set(_labels(case, TEMPLATES))) for case in cases),
        "trace_count": len(cases) + sum(len(cast(list[Json], case["traces"])) for case in cases),
    }


def manifest(cases: Sequence[Object], *, generator_sha256: str, self_test_sha256: str, support_sha256: str) -> Object:
    return {
        "alphabet": list(ALPHABET),
        "capability": "workflow_activation_v1",
        "contract_sha256": CONTRACT_SHA256,
        "consumption_addendum_sha256": CONSUMPTION_ADDENDUM_SHA256,
        "corpus_path": CORPUS_PATH,
        "corpus_sha256": hashlib.sha256(canonical_bytes(cases)).hexdigest(),
        "counts": counts(cases),
        "family_bounds": {
            "activation_universe_size": 12,
            "family_ids": list(FAMILY_IDS),
            "nested_dynamic_depth": 2,
            "one_over": 3,
            "positive_bounds": [0, 1, 2],
            "template_universe_size": 3,
        },
        "generation_provenance": {
            "byte_identical": True,
            "generations": 2,
            "tools": {
                "generator": GENERATOR_VERSION,
                "python": f"{platform.python_implementation()} {platform.python_version()}",
                "self_test": SELF_TEST_VERSION,
            },
        },
        "generator_sha256": generator_sha256,
        "independence": {"kind": "conditional-symmetric-v1", "rule_ids": list(RULE_IDS)},
        "manifest_version": "workflow-activation-reference-v5",
        "packet_id": "R1b",
        "schema_version": 4,
        "self_test_sha256": self_test_sha256,
        "support_path": SUPPORT_PATH,
        "support_sha256": support_sha256,
    }
