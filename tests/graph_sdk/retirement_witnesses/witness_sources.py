# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rebuild retired witness JSON from authenticated frozen corpora plus small residuals.

Four witness files were committed as pretty-printed copies of data that mostly
already lives in ``reference/frozen``. Each loader below selects that data by
case ID and adds only the committed residual that the frozen corpora cannot
supply. ``ORIGINAL_SHA256`` records the digests of the removed files;
``original_bytes`` regenerates those exact bytes. Selection is by ID, so no
seed is involved.
"""

from __future__ import annotations

import json
from copy import deepcopy
from functools import cache
from pathlib import Path
from typing import Any

from tests.graph_sdk.reference.corpora import CorpusName, corpus_bytes

_ROOT = Path(__file__).parent

ORIGINAL_SHA256 = {
    "workflow-mutant-cases.json": "7ece3700b32758c28240b22e37fae6d288c9821e65c6594cc3a23e8ebd0bf859",
    "data-mutant-cases.json": "5f1b1a76cec80ed3d20a68b007062904c4d1bbc2addea84c3a1056c0df078b09",
    "extra-effects-traces.json": "8ff006c6ff95e43e878acad748573afd3a0e0c40eed08524cbc8212f98c74c3e",
    "activation-differences.json": "16578c4cfdafff0357de3e920948bf6605b3e981e4a2fed142ebb4c2ca605878",
}

WORKFLOW_MUTANT_IDS = (
    "ports/000/001/input_type_A1",
    "ports/000/007/remove_output_dependency",
    "lineage/002/zero_upstream",
    "choice/004/unknown_outcome",
    "subgraph/001/009/context_alt",
    "subgraph/001/010/evidence_meaning_alt",
    "subgraph/001/011/remove_field_coverage",
    "subgraph/001/012/state_alt",
    "subgraph/001/013/model_revision_2",
    "subgraph/000/002/widen_max_activations_by_1",
    "choice/003/overlap_member",
    "lineage/000/PIPE_L",
    "ports/001/015/internal_type_A1",
)

DATA_MUTANT_IDS = (
    ("collapse_equal_text_identities", "identity-003245-equal-text-distinct"),
    ("infer_atomic_groups_from_context", "relation-separation-037474"),
    ("admit_overlapping_output_ownership", "ranges-020090-outputs"),
    ("reject_cyclic_context", "context-003326"),
    ("accept_foreign_identity", "ownership-negatives-037990-foreign-target"),
    ("publish_successful_unassessed_target", "record-038267-closed-unassessedclosed-unassessed"),
)


def _by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["case_id"]: row for row in rows}


@cache
def _frozen(name: CorpusName) -> dict[str, dict[str, Any]]:
    """Index a frozen corpus once; callers must deepcopy what they return."""
    decoded = json.loads(corpus_bytes(name))
    return _by_id(decoded["cases"] if name == "workflow_static" else decoded)


@cache
def _extracted_activation() -> dict[str, dict[str, Any]]:
    return _by_id(json.loads((_ROOT / "activation-extracted-cases.json").read_text()))


def _residual(name: str) -> list[dict[str, Any]]:
    lines = (_ROOT / name).read_text().splitlines()
    return [json.loads(line) for line in lines]


def workflow_mutant_cases() -> list[dict[str, Any]]:
    cases = _frozen("workflow_static")
    return [deepcopy(cases[case_id]) for case_id in WORKFLOW_MUTANT_IDS]


def data_mutant_cases() -> list[dict[str, Any]]:
    cases = _frozen("data")
    return [{"case": deepcopy(cases[case_id]), "mutant": mutant} for mutant, case_id in DATA_MUTANT_IDS]


def extra_effects_traces() -> list[dict[str, Any]]:
    """Legacy request histories; declarations come from the frozen effects corpus."""
    cases = _frozen("effects")
    rows = []
    for residual in _residual("extra-effects-residual.jsonl"):
        row = dict(residual)
        row["declaration"] = deepcopy(cases[row.pop("declaration_from")]["declaration"])
        rows.append(row)
    return rows


def activation_differences() -> list[dict[str, Any]]:
    """Historical product/answer differences for extracted activation cases.

    Rows without residual ``events`` repeat an extracted activation case
    verbatim. The rest keep their legacy events and answer with the frozen
    activation corpus declaration of the same ID.
    """
    extracted = _extracted_activation()
    frozen = _frozen("activation")
    rows = []
    for residual in _residual("activation-differences-residual.jsonl"):
        case_id = residual["case_id"]
        source = frozen[case_id] if "events" in residual else extracted[case_id]
        answer = residual if "events" in residual else source
        rows.append(
            {
                "case_id": case_id,
                "actual": residual["actual"],
                "expected": deepcopy(answer["expected"]),
                "declaration": deepcopy(source["declaration"]),
                "events": deepcopy(answer["events"]),
            }
        )
    return rows


@cache
def original_bytes(name: str) -> bytes:
    """Regenerate the exact bytes of a removed witness file."""
    if name == "workflow-mutant-cases.json":
        return (json.dumps(workflow_mutant_cases(), indent=2, sort_keys=True) + "\n").encode()
    if name == "data-mutant-cases.json":
        return (json.dumps(data_mutant_cases(), indent=2, sort_keys=True) + "\n").encode()
    if name == "extra-effects-traces.json":
        return (json.dumps(extra_effects_traces(), indent=2, sort_keys=True) + "\n").encode()
    if name == "activation-differences.json":
        return json.dumps(activation_differences(), indent=2).encode()
    raise ValueError(f"Unknown witness file: {name!r}")
