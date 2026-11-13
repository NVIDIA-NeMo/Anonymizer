# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A fresh recording of each built-in route must match the committed one; see ``harness`` to regenerate."""

from __future__ import annotations

import json

import pytest

from tests.parity.harness import CASES, Case, differences, record_case, recording_path


@pytest.fixture(scope="module")
def fresh_recordings() -> dict[str, dict]:
    """Each case recorded once, as it reads back from JSON."""
    return {case.name: json.loads(json.dumps(record_case(case))) for case in CASES}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_recording_matches_committed(case: Case, fresh_recordings: dict[str, dict]) -> None:
    recorded = json.loads(recording_path(case).read_text())
    diffs = differences(recorded, fresh_recordings[case.name])
    assert not diffs, f"{len(diffs)} field(s) differ from {recording_path(case).name}:\n" + "\n".join(diffs[:40])


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_every_run_happens_inside_an_activation_and_results_stay_consumable(
    case: Case, fresh_recordings: dict[str, dict]
) -> None:
    for call, observed in fresh_recordings[case.name].items():
        outside = [run["workflow"] for run in observed["workflow_runs"] if not run["inside_activation"]]
        assert not outside, f"{call}: workflow runs outside a graph activation: {outside}"
        consumers = observed["consumers"]
        assert consumers["user_columns_in_trace"], f"{call}: user columns are not a subset of the trace columns"
        assert all(consumers["pickle"].values()), f"{call}: pickling the result changed it: {consumers['pickle']}"
        assert consumers["parquet"]["columns"] == observed["user_frame"]["columns"], f"{call}: Parquet columns differ"
