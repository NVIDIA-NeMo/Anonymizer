# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A fresh recording of each built-in route must match the committed one; see ``harness`` to regenerate."""

from __future__ import annotations

import io
import json
import tempfile

import pytest

from anonymizer import Anonymizer
from anonymizer.config.anonymizer_config import TextRecord, TextRecordsInput
from tests.parity.harness import CASES, RECORDS, Case, differences, record_case, recording_path, scripted


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


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_trace_frames_write_to_parquet(case: Case) -> None:
    """Every trace frame, for run and evaluate, must be writable with ``DataFrame.to_parquet``."""
    data = TextRecordsInput(records=[TextRecord(id=i, text=t) for i, t in RECORDS])
    with tempfile.TemporaryDirectory() as artifacts:
        anonymizer = Anonymizer(artifact_path=artifacts)
        with scripted(case):
            run = anonymizer.run(config=case.config(), data=data)
        with scripted(case):
            evaluated = anonymizer.evaluate(run, config=case.evaluate)
    for call, result in (("run", run), ("evaluate", evaluated)):
        unwritable = []
        for column in result.trace_dataframe.columns:
            try:
                result.trace_dataframe[[column]].to_parquet(io.BytesIO())
            except Exception as error:
                unwritable.append(f"{column}: {type(error).__name__}")
        assert not unwritable, f"{call}: trace columns that cannot be written to Parquet: {unwritable}"
        result.trace_dataframe.to_parquet(io.BytesIO())


@pytest.mark.parametrize("case", [case for case in CASES if case.missed], ids=lambda case: case.name)
def test_scripted_coverage_miss_reaches_the_evaluated_result(case: Case, fresh_recordings: dict[str, dict]) -> None:
    rows = [r for r in fresh_recordings[case.name]["evaluate"]["user_frame"]["rows"] if case.missed in r["text"]]
    assert len(rows) == 1, f"expected one evaluated row containing {case.missed!r}"
    missed = [entity["value"] for entity in rows[0]["missed_entities"]]
    assert missed == [case.missed_entity["value"]], f"coverage judge miss did not survive: {rows[0]['missed_entities']}"
    assert rows[0]["entity_coverage"] < 1.0


@pytest.mark.parametrize(
    ("recorded", "fresh"),
    [
        ({"missed_entities": []}, {"missed_entities": {}}),
        ({"missed_entities": {}}, {"missed_entities": []}),
        ({"final_entities": {"entities": []}}, {"final_entities": {}}),
        ({"final_entities": {}}, {"final_entities": {"entities": []}}),
        ({"rows": [[]]}, {"rows": []}),
        ([], {}),
    ],
    ids=["list-to-dict", "dict-to-list", "empty-list-removed", "empty-list-added", "nested-empty-removed", "root"],
)
def test_differences_reports_empty_container_changes(recorded: object, fresh: object) -> None:
    assert differences(recorded, fresh)


def test_differences_is_empty_for_equal_values_with_empty_containers() -> None:
    value = {"a": [], "b": {}, "c": [{"d": [], "e": 1}], "f": None}
    assert differences(value, json.loads(json.dumps(value))) == []
