# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent qualification_v1 reference: facade."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from typing import cast

from tests.graph_sdk.reference._qualification_v1.authentication import (
    authenticate as authenticate,
)
from tests.graph_sdk.reference._qualification_v1.authentication import (
    cleanup_codes as cleanup_codes,
)
from tests.graph_sdk.reference._qualification_v1.authentication import (
    reconcile as reconcile,
)
from tests.graph_sdk.reference._qualification_v1.authentication import (
    request_codes as request_codes,
)
from tests.graph_sdk.reference._qualification_v1.authentication import (
    typed_endpoint_owner as typed_endpoint_owner,
)
from tests.graph_sdk.reference._qualification_v1.authentication import (
    validity as validity,
)
from tests.graph_sdk.reference._qualification_v1.case import (
    VERSIONED_CASES as VERSIONED_CASES,
)
from tests.graph_sdk.reference._qualification_v1.case import (
    _materialize_initial_versions as _materialize_initial_versions,
)
from tests.graph_sdk.reference._qualification_v1.case import (
    _replace_refs as _replace_refs,
)
from tests.graph_sdk.reference._qualification_v1.case import (
    case as case,
)
from tests.graph_sdk.reference._qualification_v1.case import (
    mutate as mutate,
)
from tests.graph_sdk.reference._qualification_v1.declarations import (
    admit as admit,
)
from tests.graph_sdk.reference._qualification_v1.declarations import (
    declaration as declaration,
)
from tests.graph_sdk.reference._qualification_v1.declarations import (
    set_output_dependency as set_output_dependency,
)
from tests.graph_sdk.reference._qualification_v1.declarations import (
    set_production_consumed as set_production_consumed,
)
from tests.graph_sdk.reference._qualification_v1.dynamic_builders import (
    assessed_map_declaration as assessed_map_declaration,
)
from tests.graph_sdk.reference._qualification_v1.dynamic_builders import (
    assessed_map_events as assessed_map_events,
)
from tests.graph_sdk.reference._qualification_v1.dynamic_builders import (
    inject_member_assessment as inject_member_assessment,
)
from tests.graph_sdk.reference._qualification_v1.dynamic_builders import (
    non_success_member_events as non_success_member_events,
)
from tests.graph_sdk.reference._qualification_v1.dynamic_builders import (
    remove_member_evidence as remove_member_evidence,
)
from tests.graph_sdk.reference._qualification_v1.dynamic_builders import (
    submit_dynamic_occurrences as submit_dynamic_occurrences,
)
from tests.graph_sdk.reference._qualification_v1.dynamic_cases import (
    dynamic_cases as dynamic_cases,
)
from tests.graph_sdk.reference._qualification_v1.events import (
    artifact as artifact,
)
from tests.graph_sdk.reference._qualification_v1.events import (
    assessment as assessment,
)
from tests.graph_sdk.reference._qualification_v1.events import (
    assessment_submission as assessment_submission,
)
from tests.graph_sdk.reference._qualification_v1.events import (
    base_events as base_events,
)
from tests.graph_sdk.reference._qualification_v1.events import (
    binding_declaration as binding_declaration,
)
from tests.graph_sdk.reference._qualification_v1.events import (
    binding_receipt as binding_receipt,
)
from tests.graph_sdk.reference._qualification_v1.events import (
    binding_ref as binding_ref,
)
from tests.graph_sdk.reference._qualification_v1.events import (
    entry as entry,
)
from tests.graph_sdk.reference._qualification_v1.events import (
    final as final,
)
from tests.graph_sdk.reference._qualification_v1.events import (
    input_producer as input_producer,
)
from tests.graph_sdk.reference._qualification_v1.events import (
    port as port,
)
from tests.graph_sdk.reference._qualification_v1.events import (
    provenance as provenance,
)
from tests.graph_sdk.reference._qualification_v1.events import (
    revisions as revisions,
)
from tests.graph_sdk.reference._qualification_v1.events import (
    terminal as terminal,
)
from tests.graph_sdk.reference._qualification_v1.family_builders import (
    binding_cleanup_events as binding_cleanup_events,
)
from tests.graph_sdk.reference._qualification_v1.family_builders import (
    bound_events as bound_events,
)
from tests.graph_sdk.reference._qualification_v1.family_builders import (
    execution_only_shape as execution_only_shape,
)
from tests.graph_sdk.reference._qualification_v1.family_builders import (
    initial_collection_events as initial_collection_events,
)
from tests.graph_sdk.reference._qualification_v1.family_builders import (
    initial_version_seed as initial_version_seed,
)
from tests.graph_sdk.reference._qualification_v1.family_builders import (
    map_version_events as map_version_events,
)
from tests.graph_sdk.reference._qualification_v1.family_builders import (
    multi_promise_declaration as multi_promise_declaration,
)
from tests.graph_sdk.reference._qualification_v1.family_builders import (
    multi_promise_events as multi_promise_events,
)
from tests.graph_sdk.reference._qualification_v1.family_builders import (
    request_history as request_history,
)
from tests.graph_sdk.reference._qualification_v1.family_builders import (
    retain_root_input as retain_root_input,
)
from tests.graph_sdk.reference._qualification_v1.family_builders import (
    wire_evidence_consumed as wire_evidence_consumed,
)
from tests.graph_sdk.reference._qualification_v1.family_builders import (
    wire_result as wire_result,
)
from tests.graph_sdk.reference._qualification_v1.lifecycle_cases import (
    lifecycle_cases as lifecycle_cases,
)
from tests.graph_sdk.reference._qualification_v1.lineage_cases import (
    lineage_cases as lineage_cases,
)
from tests.graph_sdk.reference._qualification_v1.map_builders import (
    map_declaration as map_declaration,
)
from tests.graph_sdk.reference._qualification_v1.map_builders import (
    map_events as map_events,
)
from tests.graph_sdk.reference._qualification_v1.map_builders import (
    map_item_events as map_item_events,
)
from tests.graph_sdk.reference._qualification_v1.map_evidence_builders import (
    block_keyed_join as block_keyed_join,
)
from tests.graph_sdk.reference._qualification_v1.map_evidence_builders import (
    direct_item_declaration as direct_item_declaration,
)
from tests.graph_sdk.reference._qualification_v1.map_evidence_builders import (
    direct_item_events as direct_item_events,
)
from tests.graph_sdk.reference._qualification_v1.map_evidence_builders import (
    map_item_endpoint as map_item_endpoint,
)
from tests.graph_sdk.reference._qualification_v1.map_evidence_cases import (
    map_evidence_cases as map_evidence_cases,
)
from tests.graph_sdk.reference._qualification_v1.model import (
    CONTRACT_SHA256 as CONTRACT_SHA256,
)
from tests.graph_sdk.reference._qualification_v1.model import (
    CORPUS_PATH as CORPUS_PATH,
)
from tests.graph_sdk.reference._qualification_v1.model import (
    GENERATOR_VERSION as GENERATOR_VERSION,
)
from tests.graph_sdk.reference._qualification_v1.model import (
    LIMIT_KEYS as LIMIT_KEYS,
)
from tests.graph_sdk.reference._qualification_v1.model import (
    MAP_ITEM_EVIDENCE_CONTRACT_SHA256 as MAP_ITEM_EVIDENCE_CONTRACT_SHA256,
)
from tests.graph_sdk.reference._qualification_v1.model import (
    MATERIALIZED_VERSION_CONTRACT_SHA256 as MATERIALIZED_VERSION_CONTRACT_SHA256,
)
from tests.graph_sdk.reference._qualification_v1.model import (
    REASON_CODES as REASON_CODES,
)
from tests.graph_sdk.reference._qualification_v1.model import (
    SELF_TEST_VERSION as SELF_TEST_VERSION,
)
from tests.graph_sdk.reference._qualification_v1.model import (
    STRUCTURAL_CONTRACT_SHA256 as STRUCTURAL_CONTRACT_SHA256,
)
from tests.graph_sdk.reference._qualification_v1.model import (
    TERMINAL_CATEGORIES as TERMINAL_CATEGORIES,
)
from tests.graph_sdk.reference._qualification_v1.model import (
    V10_IDS_SHA256 as V10_IDS_SHA256,
)
from tests.graph_sdk.reference._qualification_v1.model import (
    Json as Json,
)
from tests.graph_sdk.reference._qualification_v1.model import (
    Obj as Obj,
)
from tests.graph_sdk.reference._qualification_v1.model import (
    arr as arr,
)
from tests.graph_sdk.reference._qualification_v1.model import (
    canonical_bytes as canonical_bytes,
)
from tests.graph_sdk.reference._qualification_v1.model import (
    limits as limits,
)
from tests.graph_sdk.reference._qualification_v1.model import (
    obj as obj,
)
from tests.graph_sdk.reference._qualification_v1.model import (
    reject as reject,
)
from tests.graph_sdk.reference._qualification_v1.mutation_cases import (
    mutation_cases as mutation_cases,
)
from tests.graph_sdk.reference._qualification_v1.ownership_cases import (
    ownership_cases as ownership_cases,
)
from tests.graph_sdk.reference._qualification_v1.provenance import (
    validate_provenance as validate_provenance,
)
from tests.graph_sdk.reference._qualification_v1.qualification import (
    qualify as qualify,
)
from tests.graph_sdk.reference._qualification_v1.qualification import (
    reduce as reduce,
)
from tests.graph_sdk.reference._qualification_v1.records import (
    add_unique as add_unique,
)
from tests.graph_sdk.reference._qualification_v1.records import (
    advance as advance,
)
from tests.graph_sdk.reference._qualification_v1.records import (
    initial as initial,
)
from tests.graph_sdk.reference._qualification_v1.release_cases import (
    release_cases as release_cases,
)
from tests.graph_sdk.reference._qualification_v1.version_cases import (
    version_cases as version_cases,
)


def trace_count(xs: Sequence[Obj]) -> int:
    return sum(1 + len(arr(x["traces"])) for x in xs)


def event_count(xs: Sequence[Obj]) -> int:
    return sum(len(arr(x["events"])) + sum(len(arr(obj(t)["events"])) for t in arr(x["traces"])) for x in xs)


def manifest(xs: Sequence[Obj]) -> Obj:
    data = canonical_bytes(xs)
    return {
        "case_count": len(xs),
        "contract_sha256": CONTRACT_SHA256,
        "corpus_path": CORPUS_PATH,
        "corpus_sha256": hashlib.sha256(data).hexdigest(),
        "event_count": event_count(xs),
        "family_counts": dict(sorted(FAMILY_COUNTS.items())),
        "generator_version": GENERATOR_VERSION,
        "map_item_evidence_contract_sha256": MAP_ITEM_EVIDENCE_CONTRACT_SHA256,
        "materialized_version_contract_sha256": MATERIALIZED_VERSION_CONTRACT_SHA256,
        "self_test_version": SELF_TEST_VERSION,
        "structural_contract_sha256": STRUCTURAL_CONTRACT_SHA256,
        "trace_count": trace_count(xs),
        "v10_ids_sha256": V10_IDS_SHA256,
    }


def generate_cases() -> tuple[Obj, ...]:
    lineage = lineage_cases()
    direct = next(value for value in lineage if value["case_id"] == "decisions/direct")
    return tuple(
        [
            *release_cases(),
            *lifecycle_cases(),
            *lineage,
            *dynamic_cases(),
            *map_evidence_cases(),
            *version_cases(),
            *mutation_cases(direct),
            *ownership_cases(direct),
        ]
    )


CASES = generate_cases()


FAMILY_COUNTS = Counter(cast(str, x["family"]) for x in CASES)
