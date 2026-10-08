# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent effects_v1 reference: facade."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from typing import cast

from tests.graph_sdk.reference._effects_v1.admission import (
    admit as admit,
)
from tests.graph_sdk.reference._effects_v1.admission import (
    recheck_capabilities as recheck_capabilities,
)
from tests.graph_sdk.reference._effects_v1.admission_cases import (
    admission_cases as admission_cases,
)
from tests.graph_sdk.reference._effects_v1.binding_cases import (
    binding_cases as binding_cases,
)
from tests.graph_sdk.reference._effects_v1.binding_correction_cases import (
    _binding_correction_specs as _binding_correction_specs,
)
from tests.graph_sdk.reference._effects_v1.budget_cases import (
    budget_cases as budget_cases,
)
from tests.graph_sdk.reference._effects_v1.builders import (
    _bind as _bind,
)
from tests.graph_sdk.reference._effects_v1.builders import (
    _binding_decl as _binding_decl,
)
from tests.graph_sdk.reference._effects_v1.builders import (
    _decl as _decl,
)
from tests.graph_sdk.reference._effects_v1.builders import (
    _dispatch as _dispatch,
)
from tests.graph_sdk.reference._effects_v1.builders import (
    _items as _items,
)
from tests.graph_sdk.reference._effects_v1.builders import (
    _map_decl as _map_decl,
)
from tests.graph_sdk.reference._effects_v1.builders import (
    _map_event as _map_event,
)
from tests.graph_sdk.reference._effects_v1.builders import (
    _materialization_category as _materialization_category,
)
from tests.graph_sdk.reference._effects_v1.builders import (
    _materialization_decl as _materialization_decl,
)
from tests.graph_sdk.reference._effects_v1.builders import (
    _materialization_event as _materialization_event,
)
from tests.graph_sdk.reference._effects_v1.builders import (
    _materialization_limits as _materialization_limits,
)
from tests.graph_sdk.reference._effects_v1.builders import (
    _materialization_spec as _materialization_spec,
)
from tests.graph_sdk.reference._effects_v1.builders import (
    _materialization_trace as _materialization_trace,
)
from tests.graph_sdk.reference._effects_v1.builders import (
    _operation_start as _operation_start,
)
from tests.graph_sdk.reference._effects_v1.builders import (
    _policy as _policy,
)
from tests.graph_sdk.reference._effects_v1.builders import (
    _policy_runtime_decl as _policy_runtime_decl,
)
from tests.graph_sdk.reference._effects_v1.builders import (
    _publication_event as _publication_event,
)
from tests.graph_sdk.reference._effects_v1.builders import (
    _reserve as _reserve,
)
from tests.graph_sdk.reference._effects_v1.builders import (
    _result as _result,
)
from tests.graph_sdk.reference._effects_v1.builders import (
    _runtime_decl as _runtime_decl,
)
from tests.graph_sdk.reference._effects_v1.builders import (
    _source_failure as _source_failure,
)
from tests.graph_sdk.reference._effects_v1.builders import (
    _source_result as _source_result,
)
from tests.graph_sdk.reference._effects_v1.builders import (
    _trace as _trace,
)
from tests.graph_sdk.reference._effects_v1.builders import (
    binding as binding,
)
from tests.graph_sdk.reference._effects_v1.case import (
    _case as _case,
)
from tests.graph_sdk.reference._effects_v1.correction_cases import (
    correction_cases as correction_cases,
)
from tests.graph_sdk.reference._effects_v1.evaluation import (
    _map_execution_preflight as _map_execution_preflight,
)
from tests.graph_sdk.reference._effects_v1.evaluation import (
    _map_local_result as _map_local_result,
)
from tests.graph_sdk.reference._effects_v1.evaluation import (
    _materialization_preflight as _materialization_preflight,
)
from tests.graph_sdk.reference._effects_v1.evaluation import (
    evaluate_case as evaluate_case,
)
from tests.graph_sdk.reference._effects_v1.latest_cases import (
    _latest_selection_specs as _latest_selection_specs,
)
from tests.graph_sdk.reference._effects_v1.map import (
    _admit_map as _admit_map,
)
from tests.graph_sdk.reference._effects_v1.map import (
    _empty_map_state as _empty_map_state,
)
from tests.graph_sdk.reference._effects_v1.map import (
    _reduce_map as _reduce_map,
)
from tests.graph_sdk.reference._effects_v1.map_cases import (
    _map_specs as _map_specs,
)
from tests.graph_sdk.reference._effects_v1.materialization import (
    _materialization_declaration as _materialization_declaration,
)
from tests.graph_sdk.reference._effects_v1.materialization import (
    _materialize as _materialize,
)
from tests.graph_sdk.reference._effects_v1.materialization import (
    _natural as _natural,
)
from tests.graph_sdk.reference._effects_v1.materialization import (
    _publish_operation as _publish_operation,
)
from tests.graph_sdk.reference._effects_v1.materialization import (
    _requires_materialization as _requires_materialization,
)
from tests.graph_sdk.reference._effects_v1.materialization import (
    _source_followup_available as _source_followup_available,
)
from tests.graph_sdk.reference._effects_v1.materialization_admission import (
    _admit_materializations as _admit_materializations,
)
from tests.graph_sdk.reference._effects_v1.materialization_cases import (
    _materialization_specs as _materialization_specs,
)
from tests.graph_sdk.reference._effects_v1.model import (
    ACCEPTED_PREDECESSOR_CASE_COUNT as ACCEPTED_PREDECESSOR_CASE_COUNT,
)
from tests.graph_sdk.reference._effects_v1.model import (
    ACCEPTED_PREDECESSOR_CORPUS_SHA256 as ACCEPTED_PREDECESSOR_CORPUS_SHA256,
)
from tests.graph_sdk.reference._effects_v1.model import (
    BASE_CORPUS_SHA256 as BASE_CORPUS_SHA256,
)
from tests.graph_sdk.reference._effects_v1.model import (
    BINDING_SUCCESS_ADDENDUM_SHA256 as BINDING_SUCCESS_ADDENDUM_SHA256,
)
from tests.graph_sdk.reference._effects_v1.model import (
    CONTRACT_SHA256 as CONTRACT_SHA256,
)
from tests.graph_sdk.reference._effects_v1.model import (
    CORPUS_PATH as CORPUS_PATH,
)
from tests.graph_sdk.reference._effects_v1.model import (
    FAILURE_CLASSES as FAILURE_CLASSES,
)
from tests.graph_sdk.reference._effects_v1.model import (
    FAMILIES as FAMILIES,
)
from tests.graph_sdk.reference._effects_v1.model import (
    GENERATOR_VERSION as GENERATOR_VERSION,
)
from tests.graph_sdk.reference._effects_v1.model import (
    MAP_EXECUTION_ADDENDUM_SHA256 as MAP_EXECUTION_ADDENDUM_SHA256,
)
from tests.graph_sdk.reference._effects_v1.model import (
    MATERIALIZATION_ADDENDUM_SHA256 as MATERIALIZATION_ADDENDUM_SHA256,
)
from tests.graph_sdk.reference._effects_v1.model import (
    MATERIALIZED_VERSION_CONTRACT_SHA256 as MATERIALIZED_VERSION_CONTRACT_SHA256,
)
from tests.graph_sdk.reference._effects_v1.model import (
    OPTIONAL_OMISSION_ADDENDUM_SHA256 as OPTIONAL_OMISSION_ADDENDUM_SHA256,
)
from tests.graph_sdk.reference._effects_v1.model import (
    POLICY_CONDITIONS as POLICY_CONDITIONS,
)
from tests.graph_sdk.reference._effects_v1.model import (
    PREDECESSOR_CASE_COUNT as PREDECESSOR_CASE_COUNT,
)
from tests.graph_sdk.reference._effects_v1.model import (
    PREDECESSOR_CORPUS_SHA256 as PREDECESSOR_CORPUS_SHA256,
)
from tests.graph_sdk.reference._effects_v1.model import (
    REQUEST_BOUNDARY_ADDENDUM_SHA256 as REQUEST_BOUNDARY_ADDENDUM_SHA256,
)
from tests.graph_sdk.reference._effects_v1.model import (
    RUNTIME_CONDITIONS as RUNTIME_CONDITIONS,
)
from tests.graph_sdk.reference._effects_v1.model import (
    SELF_TEST_VERSION as SELF_TEST_VERSION,
)
from tests.graph_sdk.reference._effects_v1.model import (
    Json as Json,
)
from tests.graph_sdk.reference._effects_v1.model import (
    MappingKey as MappingKey,
)
from tests.graph_sdk.reference._effects_v1.model import (
    Object as Object,
)
from tests.graph_sdk.reference._effects_v1.model import (
    _array as _array,
)
from tests.graph_sdk.reference._effects_v1.model import (
    _expected_mapping_keys as _expected_mapping_keys,
)
from tests.graph_sdk.reference._effects_v1.model import (
    _mapping_key as _mapping_key,
)
from tests.graph_sdk.reference._effects_v1.model import (
    _object as _object,
)
from tests.graph_sdk.reference._effects_v1.model import (
    _strings as _strings,
)
from tests.graph_sdk.reference._effects_v1.model import (
    canonical_bytes as canonical_bytes,
)
from tests.graph_sdk.reference._effects_v1.model import (
    load_cases as load_cases,
)
from tests.graph_sdk.reference._effects_v1.race_cases import (
    race_cases as race_cases,
)
from tests.graph_sdk.reference._effects_v1.reducer import (
    _advance as _advance,
)
from tests.graph_sdk.reference._effects_v1.reducer import (
    reduce_trace as reduce_trace,
)
from tests.graph_sdk.reference._effects_v1.request_facts import (
    _apply_embedded_settlement as _apply_embedded_settlement,
)
from tests.graph_sdk.reference._effects_v1.request_facts import (
    _embedded_settlement as _embedded_settlement,
)
from tests.graph_sdk.reference._effects_v1.request_facts import (
    _initial as _initial,
)
from tests.graph_sdk.reference._effects_v1.request_facts import (
    _record_binding_success as _record_binding_success,
)
from tests.graph_sdk.reference._effects_v1.request_facts import (
    _record_late_terminal_conflict as _record_late_terminal_conflict,
)
from tests.graph_sdk.reference._effects_v1.request_facts import (
    _record_request_failure as _record_request_failure,
)
from tests.graph_sdk.reference._effects_v1.request_facts import (
    _reject as _reject,
)
from tests.graph_sdk.reference._effects_v1.request_facts import (
    _remove as _remove,
)
from tests.graph_sdk.reference._effects_v1.request_facts import (
    _request_fact as _request_fact,
)
from tests.graph_sdk.reference._effects_v1.request_facts import (
    _terminal as _terminal,
)
from tests.graph_sdk.reference._effects_v1.request_facts import (
    _unique as _unique,
)
from tests.graph_sdk.reference._effects_v1.request_facts import (
    _valid_settlement as _valid_settlement,
)
from tests.graph_sdk.reference._effects_v1.request_facts import (
    _valid_usage as _valid_usage,
)
from tests.graph_sdk.reference._effects_v1.runtime_cases import (
    runtime_cases as runtime_cases,
)


def generate_cases() -> tuple[Object, ...]:
    return _generate_specs()


def case_by_id(case_id: str) -> Object:
    return next(case for case in _CASES if case["case_id"] == case_id)


def trace_count(cases: Sequence[Object]) -> int:
    return sum(1 + len(_array(case["traces"])) for case in cases)


def event_count(cases: Sequence[Object]) -> int:
    return sum(
        len(_array(case["events"])) + sum(len(_array(_object(trace)["events"])) for trace in _array(case["traces"]))
        for case in cases
    )


def build_manifest(cases: Sequence[Object]) -> Object:
    corpus = canonical_bytes(cases)
    return {
        "case_count": len(cases),
        "historical_base_case_count": 156,
        "historical_base_corpus_sha256": BASE_CORPUS_SHA256,
        "predecessor_case_count": PREDECESSOR_CASE_COUNT,
        "predecessor_corpus_sha256": PREDECESSOR_CORPUS_SHA256,
        "accepted_predecessor_case_count": ACCEPTED_PREDECESSOR_CASE_COUNT,
        "accepted_predecessor_corpus_sha256": ACCEPTED_PREDECESSOR_CORPUS_SHA256,
        "accepted_predecessor_prefix_sha256": hashlib.sha256(
            canonical_bytes(cases[:ACCEPTED_PREDECESSOR_CASE_COUNT])
        ).hexdigest(),
        "corrected_predecessor_prefix_sha256": hashlib.sha256(
            canonical_bytes(cases[:PREDECESSOR_CASE_COUNT])
        ).hexdigest(),
        "contract_sha256": CONTRACT_SHA256,
        "materialization_addendum_sha256": MATERIALIZATION_ADDENDUM_SHA256,
        "materialized_version_contract_sha256": MATERIALIZED_VERSION_CONTRACT_SHA256,
        "optional_omission_addendum_sha256": OPTIONAL_OMISSION_ADDENDUM_SHA256,
        "map_execution_addendum_sha256": MAP_EXECUTION_ADDENDUM_SHA256,
        "binding_success_addendum_sha256": BINDING_SUCCESS_ADDENDUM_SHA256,
        "request_boundary_addendum_sha256": REQUEST_BOUNDARY_ADDENDUM_SHA256,
        "corpus_path": CORPUS_PATH,
        "corpus_sha256": hashlib.sha256(corpus).hexdigest(),
        "event_count": event_count(cases),
        "family_counts": dict(sorted(FAMILY_COUNTS.items())),
        "generator_version": GENERATOR_VERSION,
        "self_test_version": SELF_TEST_VERSION,
        "trace_count": trace_count(cases),
    }


def _generate_specs() -> tuple[Object, ...]:
    c = [
        value
        for family in (
            budget_cases,
            race_cases,
            binding_cases,
            runtime_cases,
            admission_cases,
            correction_cases,
        )
        for value in family()
    ]
    c.extend(_materialization_specs())

    c.extend(_map_specs())

    c.extend(_binding_correction_specs())

    oversize = next(case for case in c if case["case_id"] == "binding/oversize_retrieved_known_usage")

    optional_decl = dict(_object(oversize["declaration"]))

    optional_decl["binding_requirements"] = {"D0": "optional"}

    c.append(
        _case(
            "binding",
            "optional_oversize_partial",
            optional_decl,
            [_object(event) for event in _array(oversize["events"])],
        )
    )

    c.extend(_latest_selection_specs())

    return tuple(c)


_CASES = _generate_specs()


FAMILY_COUNTS = Counter(cast(str, case["family"]) for case in _CASES)
