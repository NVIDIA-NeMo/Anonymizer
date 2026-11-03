# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from typing import Any, get_args

import pandas as pd
import pytest

from anonymizer.config.rewrite import (
    _RISK_TOLERANCE_BUNDLES,
    EvaluationCriteria,
    Finding,
    RepairFinding,
    RiskTolerance,
)
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_LEAKAGE_MASS,
    COL_REWRITE_REPLACEMENT_READY,
    COL_REWRITTEN_TEXT,
    COL_UTILITY_SCORE,
)
from anonymizer.engine.rewrite.findings import needs_human_review, needs_repair, present_findings

_NAN = math.nan
_MISSING = object()
_CLEAN: dict[str, Any] = {
    COL_REWRITTEN_TEXT: "rewritten",
    COL_REWRITE_REPLACEMENT_READY: True,
    COL_ANY_HIGH_LEAKED: False,
    COL_UTILITY_SCORE: 0.9,
    COL_LEAKAGE_MASS: 0.1,
}


def _row(**overrides: Any) -> dict[str, Any]:
    row = {**_CLEAN, **overrides}
    return {key: value for key, value in row.items() if value is not _MISSING}


_LOW = EvaluationCriteria(risk_tolerance=RiskTolerance.low)
_REPAIR, _FLAG_LEAKAGE, _FLAG_UTILITY = _LOW.repair_threshold, _LOW.flag_leakage_above, _LOW.flag_utility_below
_CASES = [
    (Finding.rewrite_missing, {}, False),
    (Finding.rewrite_missing, {COL_REWRITTEN_TEXT: None}, True),
    (Finding.rewrite_missing, {COL_REWRITTEN_TEXT: _NAN}, True),
    (Finding.replacement_unavailable, {COL_REWRITE_REPLACEMENT_READY: False}, True),
    (Finding.replacement_unavailable, {COL_REWRITE_REPLACEMENT_READY: None}, True),
    (Finding.replacement_unavailable, {COL_REWRITE_REPLACEMENT_READY: _MISSING}, False),
    (Finding.high_leak, {COL_ANY_HIGH_LEAKED: True}, True),
    (Finding.high_leak, {COL_ANY_HIGH_LEAKED: None}, False),
    (Finding.leakage_above_repair, {COL_LEAKAGE_MASS: _REPAIR}, False),
    (Finding.leakage_above_repair, {COL_LEAKAGE_MASS: _REPAIR + 0.01}, True),
    (Finding.leakage_above_repair, {COL_LEAKAGE_MASS: _NAN}, False),
    (Finding.leakage_above_flag, {COL_LEAKAGE_MASS: _FLAG_LEAKAGE}, False),
    (Finding.leakage_above_flag, {COL_LEAKAGE_MASS: _FLAG_LEAKAGE + 0.01}, True),
    (Finding.leakage_above_flag, {COL_LEAKAGE_MASS: None}, False),
    (Finding.utility_below_flag, {COL_UTILITY_SCORE: _FLAG_UTILITY}, False),
    (Finding.utility_below_flag, {COL_UTILITY_SCORE: _FLAG_UTILITY - 0.01}, True),
    (Finding.utility_below_flag, {COL_UTILITY_SCORE: _NAN}, False),
    (Finding.unavailable_evidence, {}, False),
    (Finding.unavailable_evidence, {COL_LEAKAGE_MASS: _NAN}, True),
    (Finding.unavailable_evidence, {COL_UTILITY_SCORE: _NAN}, True),
    (Finding.unavailable_evidence, {COL_ANY_HIGH_LEAKED: _MISSING}, True),
]


@pytest.mark.parametrize(("finding", "overrides", "expected"), _CASES)
def test_finding_rule(finding: Finding, overrides: dict[str, Any], expected: bool) -> None:
    assert (finding in present_findings(_row(**overrides), _LOW)) is expected


def test_every_finding_holds_somewhere_and_unavailable_evidence_only_flags() -> None:
    assert {finding for finding, _, expected in _CASES if expected} == set(Finding)
    for bundle in _RISK_TOLERANCE_BUNDLES.values():
        assert Finding.unavailable_evidence in bundle.flag_on
        assert Finding.unavailable_evidence not in bundle.repair_on


def test_presets_repair_only_on_repair_findings() -> None:
    """Runtime mirror of the ``RepairFinding`` annotation, which ty enforces statically."""
    assert Finding.unavailable_evidence not in get_args(RepairFinding)
    for bundle in _RISK_TOLERANCE_BUNDLES.values():
        assert set(bundle.repair_on) <= set(get_args(RepairFinding))


# ---------------------------------------------------------------------------
# Equivalence with the inline formulas these findings replaced. Since the
# unavailable-evidence fix, missing or NaN metrics flag for review (where the
# old formulas passed or raised) and never trigger repair (where they raised).
# ---------------------------------------------------------------------------


def _outcome(decide: Callable[..., bool], *args: Any) -> bool | type[Exception]:
    try:
        return bool(decide(*args))
    except Exception as exc:
        return type(exc)


def _unavailable(row: dict[str, Any]) -> bool:
    return Finding.unavailable_evidence in present_findings(row, EvaluationCriteria())


def _old_repair(row: dict[str, Any], ev: EvaluationCriteria) -> bool:
    if ev.repair_any_high_leak and bool(row.get(COL_ANY_HIGH_LEAKED, False)):
        return True
    return float(row.get(COL_LEAKAGE_MASS, 0.0)) > ev.repair_threshold


def _old_review_combined(row: dict[str, Any], ev: EvaluationCriteria) -> bool:
    return (
        row[COL_REWRITTEN_TEXT] is None
        or not bool(row.get(COL_REWRITE_REPLACEMENT_READY, True))
        or bool(row[COL_ANY_HIGH_LEAKED])
        or float(row[COL_UTILITY_SCORE]) < ev.flag_utility_below
        or float(row[COL_LEAKAGE_MASS]) > ev.flag_leakage_above
    )


def _old_review_staged(df: pd.DataFrame, ev: EvaluationCriteria) -> list[bool]:
    ready = (
        ~df[COL_REWRITE_REPLACEMENT_READY].apply(bool)
        if COL_REWRITE_REPLACEMENT_READY in df.columns
        else pd.Series(False, index=df.index)
    )
    review = df[COL_REWRITTEN_TEXT].isna() | ready | df[COL_ANY_HIGH_LEAKED].apply(bool)
    review = review | (df[COL_UTILITY_SCORE].apply(float) < ev.flag_utility_below)
    return (review | (df[COL_LEAKAGE_MASS].apply(float) > ev.flag_leakage_above)).tolist()


_FLAGS = (True, False, None, _NAN, _MISSING)
_NUMBERS = (0.0, 0.45, 0.65, 1.2, 1.7, 2.2, 3.5, _NAN, None)


@pytest.mark.parametrize("risk_tolerance", list(RiskTolerance))
def test_findings_reproduce_previous_formulas(risk_tolerance: RiskTolerance) -> None:
    ev = EvaluationCriteria(risk_tolerance=risk_tolerance)
    for high, leakage in itertools.product(_FLAGS, (*_NUMBERS, _MISSING)):
        row = _row(**{COL_ANY_HIGH_LEAKED: high, COL_LEAKAGE_MASS: leakage})
        old = _outcome(_old_repair, row, ev)
        assert _outcome(needs_repair, row, ev) == (False if old is TypeError else old), row

    # The combined path compared text with ``is None``; it never sees a float-NaN rewrite.
    grid = list(itertools.product(("rewritten", None), _FLAGS, _FLAGS[:4], _NUMBERS, _NUMBERS))
    for text, ready, high, utility, leakage in grid:
        row = _row(**dict(zip(_CLEAN, (text, ready, high, utility, leakage), strict=True)))
        old = True if _unavailable(row) else _outcome(_old_review_combined, row, ev)
        assert _outcome(needs_human_review, row, ev) == old, row

    values = itertools.product(("rewritten", None, _NAN), _FLAGS[:4], _FLAGS[:4], _NUMBERS, _NUMBERS)
    frame = pd.DataFrame([dict(zip(_CLEAN, row, strict=True)) for row in values])
    for df in (frame, frame.drop(columns=COL_REWRITE_REPLACEMENT_READY)):
        rows = df.to_dict("records")
        old = [_unavailable(row) or flagged for row, flagged in zip(rows, _old_review_staged(df, ev), strict=True)]
        assert [needs_human_review(row, ev) for row in rows] == old
