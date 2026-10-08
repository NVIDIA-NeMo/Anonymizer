# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent workflow_static_v1 reference: model."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Literal, TypeAlias, cast

Json: TypeAlias = str | int | bool | None | list["Json"] | dict[str, "Json"]


Object: TypeAlias = dict[str, Json]


ValidationCode: TypeAlias = Literal[
    "invalid_type",
    "invalid_value",
    "limit_exceeded",
    "foreign_owner",
    "duplicate",
    "missing",
    "invalid_range",
    "overlap",
    "cycle",
    "contradictory",
]


CONTRACT_SHA256 = "5b66dbedad875b95d93d79372dae2ee25e045403b794d40001fdb3bbc6471fa6"


CORPUS_PATH = "tests/graph_sdk/reference/workflow_static_v1_cases.json"


GENERATOR_VERSION = "workflow-static-v1-generator-1"


SELF_TEST_VERSION = "workflow-static-v1-self-test-1"


ALPHABET = (
    "new_workflow",
    "declare_interface",
    "declare_node",
    "declare_subgraph",
    "bind_input",
    "bind_output",
    "add_sequence",
    "add_choice",
    "declare_protection",
    "admit",
    "substitute",
)


FAMILY_IDS = (
    "sequence_topology",
    "typed_ports_bindings",
    "outcome_choice",
    "declared_subgraph",
    "substitution",
    "protection_eligibility",
    "lineage_projection",
    "limits_ownership",
)


RULE_IDS = (
    "distinct_node_declarations",
    "bindings_distinct_destinations",
    "distinct_sequence_edges",
    "choices_disjoint_selectors_and_members",
    "protection_distinct_outcome_meaning_subject",
)


FAMILIES = (
    "topology",
    "ports",
    "choice",
    "subgraph",
    "substitution",
    "protection",
    "lineage",
    "limits_ownership",
)


RESOURCE_FIELDS = (
    "max_activations",
    "max_model_requests",
    "max_input_bytes",
    "max_output_bytes",
)


LIMIT_FIELDS = (
    "max_nodes",
    "max_bindings",
    "max_sequence_edges",
    "max_choices",
    "max_branch_members",
    "max_subgraph_depth",
    "max_choice_states",
)


SEMANTIC_ORDERS = {
    "input_port": ("i0", "i1"),
    "output_port": ("o0", "o1"),
    "context": ("context", "context_alt"),
    "evidence": ("assessment", "assessment_alt"),
    "state": ("state", "state_alt"),
    "model": ("model", "model_alt"),
    "coverage": ("field", "field_alt", "source", "absence"),
}


def _canonical(value: Json) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sorted(values: Iterable[Json]) -> list[Json]:
    return sorted(values, key=_canonical)


def _obj(value: Json) -> Object:
    if not isinstance(value, dict):
        raise ValueError("invalid neutral object")
    return cast(Object, value)
