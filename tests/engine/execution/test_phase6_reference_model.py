# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
from pathlib import Path

from tests.engine.execution.phase6_reference_model import (
    ReferenceCandidate,
    ReferenceCase,
    ReferenceEvidence,
    finite_reference_cases,
    reduce_reference,
    reference_manifest,
)


def test_phase6_reference_model_is_independent_and_manifest_is_frozen() -> None:
    source_path = Path(__file__).with_name("phase6_reference_model.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert imported_roots.isdisjoint({"anonymizer", "pandas", "pytest"})
    manifest = json.loads(Path(__file__).with_name("phase6_reference_manifest.json").read_text(encoding="utf-8"))
    assert reference_manifest() == manifest
    assert manifest["case_count"] == len(finite_reference_cases())
    assert manifest["canonical_trace_count"] == manifest["case_count"]
    assert manifest["max_event_count"] > 0


def test_reference_oracle_keeps_repeated_occurrences_anchored_and_reconstructs_exactly() -> None:
    result = reduce_reference(
        ReferenceCase(
            "repeated",
            ("Alice and Alice",),
            (ReferenceCandidate(0, 0, 5, "Alice", "name"),),
        )
    )

    assert result.rejection is None
    assert result.outputs == ("[REDACTED] and Alice",)
    assert result.clusters == ((0,),)
    assert result.released_groups == (0,)


def test_reference_oracle_clusters_only_explicit_evidence_and_rejects_transitive_contradiction() -> None:
    candidates = (
        ReferenceCandidate(0, 0, 1, "A", "name"),
        ReferenceCandidate(0, 2, 3, "B", "name"),
        ReferenceCandidate(0, 4, 5, "C", "name"),
    )
    separate = reduce_reference(ReferenceCase("separate", ("A B C",), candidates))
    same = reduce_reference(
        ReferenceCase(
            "same",
            ("A B C",),
            candidates,
            (ReferenceEvidence("same_subject", 0, 1),),
        )
    )
    contradictory = reduce_reference(
        ReferenceCase(
            "contradictory",
            ("A B C",),
            candidates,
            (
                ReferenceEvidence("same_subject", 0, 1),
                ReferenceEvidence("same_subject", 1, 2),
                ReferenceEvidence("distinct_subject", 0, 2),
            ),
        )
    )

    assert separate.clusters == ((0,), (1,), (2,))
    assert same.clusters == ((0, 1), (2,))
    assert contradictory.rejection == "evidence_contradiction"
    assert contradictory.released_groups == ()


def test_reference_group_predicate_failure_is_monotone_through_dependencies() -> None:
    result = reduce_reference(
        ReferenceCase(
            "propagation",
            ("A", "B"),
            (),
            dependencies=((0, 1),),
            groups=((0,), (1,)),
            group_passes=(False, True),
        )
    )

    assert result.rejection is None
    assert result.released_groups == ()
