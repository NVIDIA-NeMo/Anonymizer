# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent workflow_static_v1 reference: facade."""

from __future__ import annotations

import hashlib
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from tests.graph_sdk.reference._workflow_static_v1.builders import (
    _ceiling as _ceiling,
)
from tests.graph_sdk.reference._workflow_static_v1.builders import (
    _choice as _choice,
)
from tests.graph_sdk.reference._workflow_static_v1.builders import (
    _context as _context,
)
from tests.graph_sdk.reference._workflow_static_v1.builders import (
    _dependency as _dependency,
)
from tests.graph_sdk.reference._workflow_static_v1.builders import (
    _evidence as _evidence,
)
from tests.graph_sdk.reference._workflow_static_v1.builders import (
    _input as _input,
)
from tests.graph_sdk.reference._workflow_static_v1.builders import (
    _l as _l,
)
from tests.graph_sdk.reference._workflow_static_v1.builders import (
    _l01 as _l01,
)
from tests.graph_sdk.reference._workflow_static_v1.builders import (
    _limits as _limits,
)
from tests.graph_sdk.reference._workflow_static_v1.builders import (
    _node as _node,
)
from tests.graph_sdk.reference._workflow_static_v1.builders import (
    _one as _one,
)
from tests.graph_sdk.reference._workflow_static_v1.builders import (
    _operation as _operation,
)
from tests.graph_sdk.reference._workflow_static_v1.builders import (
    _outcome as _outcome,
)
from tests.graph_sdk.reference._workflow_static_v1.builders import (
    _outcome_binding as _outcome_binding,
)
from tests.graph_sdk.reference._workflow_static_v1.builders import (
    _output as _output,
)
from tests.graph_sdk.reference._workflow_static_v1.builders import (
    _p as _p,
)
from tests.graph_sdk.reference._workflow_static_v1.builders import (
    _pipe as _pipe,
)
from tests.graph_sdk.reference._workflow_static_v1.builders import (
    _q as _q,
)
from tests.graph_sdk.reference._workflow_static_v1.builders import (
    _ref as _ref,
)
from tests.graph_sdk.reference._workflow_static_v1.builders import (
    _reowner_workflow as _reowner_workflow,
)
from tests.graph_sdk.reference._workflow_static_v1.builders import (
    _workflow as _workflow,
)
from tests.graph_sdk.reference._workflow_static_v1.builders import (
    _wrap as _wrap,
)
from tests.graph_sdk.reference._workflow_static_v1.builders import (
    _z as _z,
)
from tests.graph_sdk.reference._workflow_static_v1.cases import (
    _case as _case,
)
from tests.graph_sdk.reference._workflow_static_v1.cases import (
    _change_semantic as _change_semantic,
)
from tests.graph_sdk.reference._workflow_static_v1.cases import (
    _choice_cases as _choice_cases,
)
from tests.graph_sdk.reference._workflow_static_v1.cases import (
    _consistent_body_mutation as _consistent_body_mutation,
)
from tests.graph_sdk.reference._workflow_static_v1.cases import (
    _exact_limits as _exact_limits,
)
from tests.graph_sdk.reference._workflow_static_v1.cases import (
    _limits_cases as _limits_cases,
)
from tests.graph_sdk.reference._workflow_static_v1.cases import (
    _lineage_cases as _lineage_cases,
)
from tests.graph_sdk.reference._workflow_static_v1.cases import (
    _one_q as _one_q,
)
from tests.graph_sdk.reference._workflow_static_v1.cases import (
    _pipe_q as _pipe_q,
)
from tests.graph_sdk.reference._workflow_static_v1.cases import (
    _ports_cases as _ports_cases,
)
from tests.graph_sdk.reference._workflow_static_v1.cases import (
    _protection_cases as _protection_cases,
)
from tests.graph_sdk.reference._workflow_static_v1.cases import (
    _replacement_mutation as _replacement_mutation,
)
from tests.graph_sdk.reference._workflow_static_v1.cases import (
    _subgraph_cases as _subgraph_cases,
)
from tests.graph_sdk.reference._workflow_static_v1.cases import (
    _substitution_cases as _substitution_cases,
)
from tests.graph_sdk.reference._workflow_static_v1.cases import (
    _topology_cases as _topology_cases,
)
from tests.graph_sdk.reference._workflow_static_v1.model import (
    ALPHABET as ALPHABET,
)
from tests.graph_sdk.reference._workflow_static_v1.model import (
    CONTRACT_SHA256 as CONTRACT_SHA256,
)
from tests.graph_sdk.reference._workflow_static_v1.model import (
    CORPUS_PATH as CORPUS_PATH,
)
from tests.graph_sdk.reference._workflow_static_v1.model import (
    FAMILIES as FAMILIES,
)
from tests.graph_sdk.reference._workflow_static_v1.model import (
    FAMILY_IDS as FAMILY_IDS,
)
from tests.graph_sdk.reference._workflow_static_v1.model import (
    GENERATOR_VERSION as GENERATOR_VERSION,
)
from tests.graph_sdk.reference._workflow_static_v1.model import (
    LIMIT_FIELDS as LIMIT_FIELDS,
)
from tests.graph_sdk.reference._workflow_static_v1.model import (
    RESOURCE_FIELDS as RESOURCE_FIELDS,
)
from tests.graph_sdk.reference._workflow_static_v1.model import (
    RULE_IDS as RULE_IDS,
)
from tests.graph_sdk.reference._workflow_static_v1.model import (
    SELF_TEST_VERSION as SELF_TEST_VERSION,
)
from tests.graph_sdk.reference._workflow_static_v1.model import (
    SEMANTIC_ORDERS as SEMANTIC_ORDERS,
)
from tests.graph_sdk.reference._workflow_static_v1.model import (
    Json as Json,
)
from tests.graph_sdk.reference._workflow_static_v1.model import (
    Object as Object,
)
from tests.graph_sdk.reference._workflow_static_v1.model import (
    ValidationCode as ValidationCode,
)
from tests.graph_sdk.reference._workflow_static_v1.model import (
    _canonical as _canonical,
)
from tests.graph_sdk.reference._workflow_static_v1.model import (
    _obj as _obj,
)
from tests.graph_sdk.reference._workflow_static_v1.model import (
    _sorted as _sorted,
)
from tests.graph_sdk.reference._workflow_static_v1.traces import (
    _events as _events,
)
from tests.graph_sdk.reference._workflow_static_v1.traces import (
    _rename_maps as _rename_maps,
)
from tests.graph_sdk.reference._workflow_static_v1.traces import (
    _rewrite as _rewrite,
)
from tests.graph_sdk.reference._workflow_static_v1.traces import (
    _rewrite_role as _rewrite_role,
)
from tests.graph_sdk.reference._workflow_static_v1.traces import (
    _traces as _traces,
)
from tests.graph_sdk.reference._workflow_static_v1.traces import (
    independent as independent,
)
from tests.graph_sdk.reference._workflow_static_v1.validation import (
    _admission_code as _admission_code,
)
from tests.graph_sdk.reference._workflow_static_v1.validation import (
    _contradictory as _contradictory,
)
from tests.graph_sdk.reference._workflow_static_v1.validation import (
    _cycle_and_sinks as _cycle_and_sinks,
)
from tests.graph_sdk.reference._workflow_static_v1.validation import (
    _duplicates as _duplicates,
)
from tests.graph_sdk.reference._workflow_static_v1.validation import (
    _foreign as _foreign,
)
from tests.graph_sdk.reference._workflow_static_v1.validation import (
    _has_closed_vocabularies as _has_closed_vocabularies,
)
from tests.graph_sdk.reference._workflow_static_v1.validation import (
    _id_label as _id_label,
)
from tests.graph_sdk.reference._workflow_static_v1.validation import (
    _identity_endpoints as _identity_endpoints,
)
from tests.graph_sdk.reference._workflow_static_v1.validation import (
    _incompatible_operation as _incompatible_operation,
)
from tests.graph_sdk.reference._workflow_static_v1.validation import (
    _invalid_choice as _invalid_choice,
)
from tests.graph_sdk.reference._workflow_static_v1.validation import (
    _list as _list,
)
from tests.graph_sdk.reference._workflow_static_v1.validation import (
    _metrics as _metrics,
)
from tests.graph_sdk.reference._workflow_static_v1.validation import (
    _missing as _missing,
)
from tests.graph_sdk.reference._workflow_static_v1.validation import (
    _nodes as _nodes,
)
from tests.graph_sdk.reference._workflow_static_v1.validation import (
    _normalized as _normalized,
)
from tests.graph_sdk.reference._workflow_static_v1.validation import (
    _operation_outcomes as _operation_outcomes,
)
from tests.graph_sdk.reference._workflow_static_v1.validation import (
    _operation_ports as _operation_ports,
)
from tests.graph_sdk.reference._workflow_static_v1.validation import (
    _overlap as _overlap,
)
from tests.graph_sdk.reference._workflow_static_v1.validation import (
    _protection as _protection,
)
from tests.graph_sdk.reference._workflow_static_v1.validation import (
    _semantic_endpoint_code as _semantic_endpoint_code,
)
from tests.graph_sdk.reference._workflow_static_v1.validation import (
    _substituted as _substituted,
)
from tests.graph_sdk.reference._workflow_static_v1.validation import (
    judge as judge,
)


def generate_cases() -> tuple[Object, ...]:
    """Enumerate the eight exact finite families and reject collisions."""
    generators = (
        _topology_cases,
        _ports_cases,
        _choice_cases,
        _subgraph_cases,
        _substitution_cases,
        _protection_cases,
        _lineage_cases,
        _limits_cases,
    )
    cases = [case for generator in generators for case in generator()]
    ids: set[str] = set()
    payloads: dict[bytes, str] = {}
    for case in cases:
        case_id = cast(str, case["case_id"])
        if case_id in ids:
            raise ValueError("duplicate case id")
        ids.add(case_id)
        payload = _canonical({key: case[key] for key in ("family", "mode", "declaration", "replacement", "expected")})
        if payload in payloads:
            raise ValueError("duplicate canonical payload")
        payloads[payload] = case_id
    return tuple(sorted(cases, key=lambda case: cast(str, case["case_id"]).encode()))


def canonical_bytes(cases: Sequence[Object]) -> bytes:
    return _canonical({"cases": list(cases), "schema_version": 1}) + b"\n"


def counts(cases: Sequence[Object]) -> Object:
    metrics = [_metrics(_obj(case["declaration"])) for case in cases]
    return {
        "case_count": len(cases),
        "event_count": sum(len(_list(trace["events"])) for case in cases for trace in map(_obj, _list(case["traces"]))),
        "max_choice_states": max(item["choice_states"] for item in metrics),
        "max_nodes": max(item["nodes"] for item in metrics),
        "max_subgraph_depth": max(item["subgraph_depth"] for item in metrics),
        "trace_count": sum(len(_list(case["traces"])) for case in cases),
    }


def build_manifest(
    cases: Sequence[Object],
    corpus: bytes,
    *,
    generator_sha256: str,
    self_test_sha256: str,
) -> Object:
    """Build the exact manifest for already generated canonical corpus bytes."""
    return {
        "alphabet": list(ALPHABET),
        "capability": "workflow_static_v1",
        "contract_sha256": CONTRACT_SHA256,
        "corpus_path": CORPUS_PATH,
        "corpus_sha256": hashlib.sha256(corpus).hexdigest(),
        "counts": counts(cases),
        "family_bounds": {
            "choice_state_max": 4,
            "expanded_node_count_max": 3,
            "family_ids": list(FAMILY_IDS),
            "subgraph_depth_max": 2,
            "topology_node_counts": [1, 2, 3],
        },
        "generation_provenance": {
            "byte_identical": True,
            "generations": 2,
            "tools": {
                "generator": GENERATOR_VERSION,
                "python": producer_python(),
                "self_test": SELF_TEST_VERSION,
            },
        },
        "generator_sha256": generator_sha256,
        "independence": {"kind": "conditional-symmetric-v1", "rule_ids": list(RULE_IDS)},
        "manifest_version": "workflow-static-reference-v1",
        "packet_id": "R1a",
        "schema_version": 1,
        "self_test_sha256": self_test_sha256,
    }


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def producer_python() -> str:
    return f"{platform.python_implementation()} {platform.python_version()}"


def load_cases(value: Json) -> tuple[Object, ...]:
    root = _obj(value)
    if set(root) != {"cases", "schema_version"} or root["schema_version"] != 1:
        raise ValueError("invalid workflow fixture structure")
    cases = tuple(_obj(item) for item in _list(root["cases"]))
    required = {"case_id", "family", "mode", "declaration", "replacement", "expected", "traces"}
    for case in cases:
        if set(case) != required or case["family"] not in FAMILIES or case["mode"] not in {"admission", "substitution"}:
            raise ValueError("invalid workflow fixture structure")
        replacement = _obj(case["replacement"]) if case["replacement"] is not None else None
        if not _has_closed_vocabularies(_obj(case["declaration"])) or (
            replacement is not None and not _has_closed_vocabularies(replacement)
        ):
            raise ValueError("invalid workflow fixture structure")
        expected = _obj(case["expected"])
        if set(expected) != {
            "status",
            "code",
            "topology",
            "normalized",
            "protection_eligible_outcomes",
            "unmet_protection",
        }:
            raise ValueError("invalid workflow fixture structure")
        for trace in map(_obj, _list(case["traces"])):
            if set(trace) != {"transformation", "events", "declaration", "replacement", "expected"}:
                raise ValueError("invalid workflow fixture structure")
            trace_replacement = _obj(trace["replacement"]) if trace["replacement"] is not None else None
            if not _has_closed_vocabularies(_obj(trace["declaration"])) or (
                trace_replacement is not None and not _has_closed_vocabularies(trace_replacement)
            ):
                raise ValueError("invalid workflow fixture structure")
            if any(_obj(event).get("op") not in ALPHABET for event in _list(trace["events"])):
                raise ValueError("invalid workflow fixture structure")
    return cases


def module_is_independent() -> bool:
    return not any(name == "anonymizer" or name.startswith("anonymizer.") for name in sys.modules)
