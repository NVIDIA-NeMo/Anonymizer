# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``evaluate()`` places a judge's rows and the run outcomes only by a record id that occurs once.

The invariant: no output row shows another record's run outcome or judge advisory, whatever a custom runner returns.
A row that cannot be placed is ``not_assessed``, unprotected, and each judge reads ``could_not_assess``.
"""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from anonymizer.config.anonymizer_config import AnonymizerConfig, EvaluateConfig, Rewrite
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import (
    COL_DETECTION_VALID,
    COL_ENTITY_COVERAGE,
    COL_JUDGE_EVALUATION,
    RECORD_ID_COLUMN,
)
from anonymizer.engine.evaluation.entity_coverage_judge import EntityCoverageWorkflow
from anonymizer.engine.graph_core.execution import Reason, TargetOutcome
from anonymizer.engine.replace.replace_runner import ReplacementResult, ReplacementWorkflow
from anonymizer.engine.rewrite.rewrite_workflow import RewriteResult
from tests.interface.test_anonymizer_interface import _StubRewriteRunner, _two_row_anonymizer

_TRUTH = "probe_truth"  # which record a row really is; judges carry it along with the row
_MODES = {"replace": AnonymizerConfig(replace=Redact()), "rewrite": AnonymizerConfig(rewrite=Rewrite())}
_UNASSESSED = TargetOutcome("success", "not_assessed", False, False)
_STATUS = {
    COL_ENTITY_COVERAGE: ("coverage", {0.0: "failed", 1.0: "passed", None: "could_not_assess"}),
    COL_DETECTION_VALID: ("detection-validity", {True: "passed", False: "failed", None: "could_not_assess"}),
    COL_JUDGE_EVALUATION: ("rewrite-judge", {"scored": "scored", None: "could_not_assess"}),
}
Change = Callable[[pd.DataFrame], pd.DataFrame]


def _ran(k: int) -> TargetOutcome:
    """A run outcome that names record ``k``; even records are protected."""
    if k % 2:
        return TargetOutcome("success", "not_met", True, False, (Reason("failed", "replacements_applied", f"v{k}"),))
    return TargetOutcome("success", "met", True, True, (Reason("passed", f"checked-{k}", f"v{k}"),))


def _cell(column: str, verdict: Any) -> Any:
    if column == COL_JUDGE_EVALUATION:
        return {"privacy": 1} if verdict == "scored" else None
    return float("nan") if verdict is None and column == COL_ENTITY_COVERAGE else verdict


class _Judges:
    """Custom runners whose verdicts are fixed per record and whose returned frame ``change`` reshapes."""

    def __init__(self, verdicts: dict[int, dict[str, Any]], changes: dict[str, Change]) -> None:
        self.verdicts, self.changes = verdicts, changes

    def score(self, stage: str, frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
        truth = [int(t) for t in frame[_TRUTH]]
        scored = frame.assign(**{c: [_cell(c, self.verdicts[t][c]) for t in truth] for c in columns})
        return self.changes.get(stage, lambda f: f)(scored)


def _anonymizer(tmp_path: Path, mode: str, texts: list[str], judges: _Judges) -> tuple[Any, Any]:
    anonymizer, _, data = _two_row_anonymizer(tmp_path, record_ids=True)
    pd.DataFrame({"text": texts}).to_csv(tmp_path / "input.csv", index=False)

    class Replace(ReplacementWorkflow):
        def evaluate(self, dataframe: pd.DataFrame, **_: object) -> ReplacementResult:
            return ReplacementResult(judges.score("judge", dataframe, (COL_ENTITY_COVERAGE, COL_DETECTION_VALID)), [])

    class Rewriter(_StubRewriteRunner):
        def evaluate(self, dataframe: pd.DataFrame, **_: object) -> RewriteResult:
            return RewriteResult(judges.score("judge", dataframe, (COL_JUDGE_EVALUATION, COL_DETECTION_VALID)), [])

    anonymizer._replace_runner, anonymizer._rewrite_runner = Replace(), Rewriter(Mock())
    with patch.object(EntityCoverageWorkflow, "run_non_critical", lambda *a, **k: (a[1], [])):
        run = anonymizer.run(config=_MODES[mode], data=data)
    trace = run.trace_dataframe.assign(**{_TRUTH: range(len(run.trace_dataframe))})
    ids = tuple(str(i) for i in trace[RECORD_ID_COLUMN])
    return anonymizer, dataclasses.replace(
        run, trace_dataframe=trace, _outcomes=tuple(_ran(k) for k in range(len(ids))), _outcome_ids=ids
    )


def _evaluate(anonymizer: Any, run: Any, judges: _Judges) -> Any:
    def coverage(_self: object, dataframe: pd.DataFrame, **_: object) -> tuple[pd.DataFrame, list]:
        return judges.score("coverage", dataframe, (COL_ENTITY_COVERAGE,)), []

    with patch.object(EntityCoverageWorkflow, "run_non_critical", coverage):
        return anonymizer.evaluate(run, config=EvaluateConfig(compute_detection_validity=True))


def _violations(evaluated: Any, judges: _Judges, source_ids: list[str]) -> list[str]:
    """Rows that show another record's outcome or advisory, or that are unplaced but not reported so."""
    if evaluated._outcomes is None:
        return []
    found = []
    trace = evaluated.trace_dataframe
    ids: list[str | None] = (
        [str(i) for i in trace[RECORD_ID_COLUMN]] if RECORD_ID_COLUMN in trace.columns else [None] * len(trace)
    )
    for k, truth in enumerate(int(t) for t in trace[_TRUTH]):
        outcome = evaluated.outcome(k)
        bare = dataclasses.replace(outcome, advisories=())
        if bare not in (_ran(truth), _UNASSESSED):
            found.append(f"row {k} (record {truth}) shows outcome {bare}")
        if bare == _UNASSESSED and ids.count(ids[k]) == 1 and ids[k] == source_ids[truth]:
            found.append(f"row {k} (record {truth}) carries its unique id but was not placed")
        if bare == _UNASSESSED and any(a.cause != "unplaced" for a in outcome.advisories):
            found.append(f"row {k} (record {truth}) is unplaced but reports judges")
        own = {check: codes[judges.verdicts[truth][col]] for col, (check, codes) in _STATUS.items()}
        for a in outcome.advisories:
            if a.check in own and a.status not in (own[a.check], "could_not_assess"):
                found.append(f"row {k} (record {truth}) shows {a.check}={a.status}, its own is {own[a.check]}")
    return found


# The P1 and P2 reproductions from review.


def test_rows_with_the_same_original_text_and_no_ids_take_no_outcome(tmp_path: Path) -> None:
    verdicts: dict[int, dict[str, Any]] = {0: dict.fromkeys(_STATUS, None) | {COL_ENTITY_COVERAGE: 0.0}}
    verdicts[1] = verdicts[0] | {COL_ENTITY_COVERAGE: 1.0}
    reverse_without_ids: dict[str, Change] = {
        "judge": lambda f: f.iloc[::-1].reset_index(drop=True).drop(columns=[RECORD_ID_COLUMN])
    }
    judges = _Judges(verdicts, reverse_without_ids)
    anonymizer, run = _anonymizer(tmp_path, "replace", ["Alice lives here", "Alice lives here"], judges)

    evaluated = _evaluate(anonymizer, run, judges)

    assert evaluated._outcomes is None  # nothing could be placed: no outcomes, never a borrowed one


def test_rows_with_the_same_original_text_take_only_their_own_outcome(tmp_path: Path) -> None:
    verdicts: dict[int, dict[str, Any]] = {0: dict.fromkeys(_STATUS, None) | {COL_ENTITY_COVERAGE: 0.0}}
    verdicts[1] = verdicts[0] | {COL_ENTITY_COVERAGE: 1.0}
    # Reverse both rows; drop the id of the first returned row only.
    judges = _Judges(
        verdicts,
        {
            "judge": lambda f: (
                f.iloc[::-1]
                .reset_index(drop=True)
                .assign(**{RECORD_ID_COLUMN: lambda g: [None, *g[RECORD_ID_COLUMN].iloc[1:]]})
            )
        },
    )
    anonymizer, run = _anonymizer(tmp_path, "replace", ["Alice lives here", "Alice lives here"], judges)

    evaluated = _evaluate(anonymizer, run, judges)

    assert [dataclasses.replace(evaluated.outcome(k), advisories=()) for k in range(2)] == [_UNASSESSED, _ran(0)]
    assert {a.status for a in evaluated.outcome(0).advisories} == {"could_not_assess"}
    assert {a.check: a.status for a in evaluated.outcome(1).advisories}["coverage"] == "failed"


def test_each_judge_stage_places_its_own_rows_whatever_its_order(tmp_path: Path) -> None:
    verdicts = {
        0: {COL_JUDGE_EVALUATION: "scored", COL_DETECTION_VALID: True, COL_ENTITY_COVERAGE: 1.0},
        1: {COL_JUDGE_EVALUATION: None, COL_DETECTION_VALID: False, COL_ENTITY_COVERAGE: 0.0},
    }
    judges = _Judges(verdicts, {"coverage": lambda f: f.iloc[::-1].reset_index(drop=True)})
    anonymizer, run = _anonymizer(tmp_path, "rewrite", ["Alice Smith is here", "Bob Jones is here"], judges)

    evaluated = _evaluate(anonymizer, run, judges)

    assert _violations(evaluated, judges, list(run._outcome_ids)) == []
    bob = [int(t) for t in evaluated.trace_dataframe[_TRUTH]].index(1)
    assert {a.check: a.status for a in evaluated.outcome(bob).advisories} == {
        "coverage": "failed",
        "rewrite-judge": "could_not_assess",
        "detection-validity": "failed",
    }


# The randomized invariant.


def _random_change(rng: random.Random, *, last: bool) -> Change:
    """One judge stage's reshaping: reorder, drop rows, duplicate a row or an id, null or retype an id, drop ids."""
    kinds = ["shuffle", "drop", "dup_row", "null_id", "float_id", "no_ids", *(["dup_id"] if last else [])]
    steps = rng.sample(kinds, rng.randint(0, 3))
    # A row relabelled with another record's id, which then occurs once, is outside the contract (ids are trusted):
    # duplicate ids only in the last stage, after its drops, so a duplicated id always stays duplicated.
    steps.sort(key=lambda step: step == "dup_id")

    def change(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.reset_index(drop=True)
        for step in steps:
            n = len(out)
            if step == "shuffle":
                out = out.sample(frac=1, random_state=rng.randrange(10**6)).reset_index(drop=True)
            elif step == "drop" and n:
                out = out.drop(index=rng.randrange(n)).reset_index(drop=True)
            elif step == "dup_row" and n:
                out = pd.concat([out, out.iloc[[rng.randrange(n)]]], ignore_index=True)
            elif step == "dup_id" and n > 1 and RECORD_ID_COLUMN in out:
                a, b = rng.sample(range(n), 2)
                out.loc[b, RECORD_ID_COLUMN] = out.loc[a, RECORD_ID_COLUMN]  # both rows now share one id
            elif step == "null_id" and n and RECORD_ID_COLUMN in out:
                out[RECORD_ID_COLUMN] = out[RECORD_ID_COLUMN].astype(object)
                out.loc[rng.randrange(n), RECORD_ID_COLUMN] = None
            elif step == "float_id" and n and RECORD_ID_COLUMN in out:
                out[RECORD_ID_COLUMN] = [f"{i}.0" for i in out[RECORD_ID_COLUMN]]
            elif step == "no_ids" and RECORD_ID_COLUMN in out:
                out = out.drop(columns=[RECORD_ID_COLUMN])
        return out

    return change


@pytest.mark.parametrize("seed", range(200))
def test_no_row_shows_another_records_outcome_or_advisory(tmp_path: Path, seed: int) -> None:
    rng = random.Random(seed)
    mode = rng.choice(list(_MODES))
    size = rng.randint(2, 5)
    texts = [f"{rng.choice(['Alice', 'Bob'])} Smith is here" for _ in range(size)]  # duplicate texts are common
    verdicts = {k: {col: rng.choice(list(codes)) for col, (_, codes) in _STATUS.items()} for k in range(size)}
    judges = _Judges(
        verdicts, {"judge": _random_change(rng, last=mode == "replace"), "coverage": _random_change(rng, last=True)}
    )
    anonymizer, run = _anonymizer(tmp_path, mode, texts, judges)

    evaluated = _evaluate(anonymizer, run, judges)

    assert _violations(evaluated, judges, list(run._outcome_ids)) == []
