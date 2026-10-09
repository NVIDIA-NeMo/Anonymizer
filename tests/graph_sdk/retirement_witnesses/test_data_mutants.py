# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Six historical semantic mutants calibrated at actual data/record boundaries."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

from anonymizer.engine.graph_sdk.records import TargetStatus
from tests.graph_sdk.test_data_reference_conformance import _adapt
from tests.graph_sdk.test_record_values import _RecordAdapter

CASES = json.loads(Path(__file__).with_name("data-mutant-cases.json").read_text())


@pytest.mark.parametrize("row", CASES, ids=lambda row: row["mutant"])
def test_data_semantic_mutant_is_killed_at_real_boundary(row: dict[str, Any]) -> None:
    case = row["case"]
    name = row["mutant"]
    if name == "publish_successful_unassessed_target":
        declaration = case["declaration"]
        status = cast(
            TargetStatus,
            _RecordAdapter(declaration["facts"]).construct(declaration["boundary"]),
        )
        assert status.qualification == "not_assessed" and status.artifact_available and not status.protection_available
        actual = deepcopy(case["expected"])
        actual["normalized"]["protection_available"] = status.protection_available
    else:
        actual = _adapt(case)
        assert actual == case["expected"]
    mutated = cast(dict[str, Any], deepcopy(actual))
    if name == "collapse_equal_text_identities":
        for field in (
            "datums",
            "targets",
            "coherence",
            "atomic",
            "effective_ownership",
        ):
            mutated["normalized"][field] = mutated["normalized"][field][:1]
    elif name == "infer_atomic_groups_from_context":
        mutated["normalized"]["atomic"] = deepcopy(mutated["normalized"]["coherence"])
    elif name in ("admit_overlapping_output_ownership", "accept_foreign_identity"):
        mutated = {"verdict": "accept", "normalized": {}}
    elif name == "reject_cyclic_context":
        mutated = {"verdict": "reject", "code": "cycle"}
    elif name == "publish_successful_unassessed_target":
        mutated["normalized"]["protection_available"] = True
    else:
        raise AssertionError(name)
    with pytest.raises(AssertionError, match=name):
        assert mutated == case["expected"], name
