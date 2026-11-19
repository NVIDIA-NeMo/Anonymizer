# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every built-in route hands ``evaluate()`` rows that carry a unique record id, in every judge stage.

``evaluate()`` places a judge's rows and the run outcomes only by record id, so these tests pin that each built-in
route (Replace, Rewrite staged and combined, the coverage and detection-validity judges, a preview) keeps them.
"""

from __future__ import annotations

import dataclasses
import tempfile
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

import anonymizer.interface.anonymizer as facade
from anonymizer import Anonymizer
from anonymizer.config.anonymizer_config import TextRecord, TextRecordsInput
from anonymizer.engine.constants import RECORD_ID_COLUMN
from tests.parity.harness import CASES, PREVIEW_ROWS, RECORDS, Case, scripted


@dataclasses.dataclass
class _Seen:
    stages: list[tuple[str, pd.DataFrame]] = dataclasses.field(default_factory=list)
    source: pd.DataFrame | None = None
    output: pd.DataFrame | None = None
    ran_ids: tuple[str, ...] | None = None


@contextmanager
def _watched() -> Generator[_Seen]:
    """Record the frame each judge stage returns and the frames the outcomes are placed from."""
    seen = _Seen()
    evaluate_on_graph, reported_outcomes = facade._evaluate_on_graph, facade._reported_outcomes

    def stage(name: str, run: Callable[[pd.DataFrame], Any]) -> Callable[[pd.DataFrame], Any]:
        def watched(frame: pd.DataFrame) -> Any:
            returned, failed = run(frame)
            seen.stages.append((name, returned.copy()))
            return returned, failed

        return watched

    def on_graph(frame: pd.DataFrame, column: str, judges: Any) -> Any:
        return evaluate_on_graph(frame, column, [j._replace(run=stage(j.name, j.run)) for j in judges])

    def outcomes(result: Any, source: pd.DataFrame, output: pd.DataFrame, ran: Any, ran_ids: Any) -> Any:
        seen.source, seen.output, seen.ran_ids = source.copy(), output.copy(), ran_ids
        return reported_outcomes(result, source, output, ran, ran_ids)

    with patch.object(facade, "_evaluate_on_graph", on_graph), patch.object(facade, "_reported_outcomes", outcomes):
        yield seen


def _ids(frame: pd.DataFrame) -> list[str]:
    assert RECORD_ID_COLUMN in frame.columns, f"no record id column in {list(frame.columns)}"
    ids = [str(i) for i in frame[RECORD_ID_COLUMN]]
    assert len(set(ids)) == len(ids), f"duplicate record ids: {ids}"
    return ids


def _assert_placed_by_id(seen: _Seen) -> None:
    assert seen.source is not None and seen.output is not None
    source = set(_ids(seen.source))
    assert seen.stages, "no judge stage ran"
    for name, frame in seen.stages:
        assert set(_ids(frame)) <= source, f"{name} returned ids that are not in its source"
    assert set(_ids(seen.output)) <= source


def _without_judges(outcome: Any, judges: set[str]) -> Any:
    return dataclasses.replace(outcome, advisories=tuple(a for a in outcome.advisories if a.verifier not in judges))


def _run(case: Case, *, preview: bool = False) -> tuple[Anonymizer, Any]:
    data = TextRecordsInput(records=[TextRecord(id=i, text=t) for i, t in RECORDS])
    anonymizer = Anonymizer(artifact_path=tempfile.mkdtemp())
    with scripted(case):
        if preview:
            return anonymizer, anonymizer.preview(config=case.config(), data=data, num_records=PREVIEW_ROWS)
        return anonymizer, anonymizer.run(config=case.config(), data=data)


@pytest.mark.parametrize("preview", [False, True], ids=["run", "preview"])
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_evaluate_receives_unique_record_ids_on_every_built_in_route(case: Case, preview: bool) -> None:
    anonymizer, ran = _run(case, preview=preview)
    with scripted(case), _watched() as seen:
        evaluated = anonymizer.evaluate(ran, config=case.evaluate)

    _assert_placed_by_id(seen)
    assert seen.source is not None and seen.output is not None
    assert seen.ran_ids is not None and set(seen.ran_ids) == set(_ids(seen.source))
    by_id = dict(zip(ran._outcome_ids, ran._outcomes, strict=True))
    judges = {a.verifier for a in evaluated.outcome(0).advisories} - {a.verifier for a in ran._outcomes[0].advisories}
    for k, record_id in enumerate(_ids(seen.output)):
        assert _without_judges(evaluated.outcome(k), judges) == by_id[record_id]
        assert not any(
            a.status == "could_not_assess" and a.cause == "unplaced" for a in evaluated.outcome(k).advisories
        )


def test_detection_validity_and_coverage_stages_are_routes_here() -> None:
    """The cases above include the coverage judge and detection validity in both modes."""
    assert any(c.evaluate.compute_detection_validity and c.config().rewrite is not None for c in CASES)
    assert any(c.evaluate.compute_detection_validity and c.config().replace is not None for c in CASES)
    assert any((rewrite := c.config().rewrite) is not None and rewrite.use_combined_graph for c in CASES)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_a_result_pickled_before_outcomes_were_kept_reports_judges_but_no_run_outcome(case: Case) -> None:
    """Such a result has no saved outcomes: its rows are still placed by id, so each row reports its own judges, but
    no run outcome is borrowed, so every row reads ``not_assessed`` and unprotected."""
    anonymizer, ran = _run(case)
    for name in ("_outcomes", "_outcome_ids"):
        object.__delattr__(ran, name)  # as unpickling a result saved before these fields existed leaves it
    with scripted(case), _watched() as seen:
        evaluated = anonymizer.evaluate(ran, config=case.evaluate)

    _assert_placed_by_id(seen)
    assert seen.ran_ids is None
    for k in range(len(evaluated.trace_dataframe)):
        outcome = evaluated.outcome(k)
        assert (outcome.qualification, outcome.protected, outcome.reasons) == ("not_assessed", False, ())
        assert outcome.advisories
