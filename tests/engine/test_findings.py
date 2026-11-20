# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from typing import Any, get_args

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
from anonymizer.engine.rewrite.findings import needs_repair, present_findings

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


@pytest.mark.parametrize(
    ("risk_tolerance", "expect_repair"),
    [
        (RiskTolerance.minimal, True),
        (RiskTolerance.low, True),
        (RiskTolerance.moderate, True),
        (RiskTolerance.high, False),
    ],
)
def test_nan_any_high_leaked_is_high_leak_and_repairs_except_high(
    risk_tolerance: RiskTolerance, expect_repair: bool
) -> None:
    """Compatibility exception: ``bool(nan)`` is truthy, so NaN reads as a high leak, not only unavailable evidence."""
    ev = EvaluationCriteria(risk_tolerance=risk_tolerance)
    row = _row(**{COL_ANY_HIGH_LEAKED: _NAN})
    findings = present_findings(row, ev)
    assert Finding.high_leak in findings
    assert Finding.unavailable_evidence in findings
    assert needs_repair(row, ev) is expect_repair


def test_every_finding_holds_somewhere_and_unavailable_evidence_never_repairs() -> None:
    assert {finding for finding, _, expected in _CASES if expected} == set(Finding)
    for bundle in _RISK_TOLERANCE_BUNDLES.values():
        assert Finding.unavailable_evidence not in bundle.repair_on


def test_no_preset_lists_findings_that_flag_for_review() -> None:
    """The review flag follows the rewrite's final checks, not a list of findings, so none is published."""
    for risk_tolerance in RiskTolerance:
        assert not hasattr(EvaluationCriteria(risk_tolerance=risk_tolerance), "flag_on")


def test_presets_repair_only_on_repair_findings() -> None:
    """Runtime mirror of the ``RepairFinding`` annotation, which ty enforces statically."""
    assert Finding.unavailable_evidence not in get_args(RepairFinding)
    for bundle in _RISK_TOLERANCE_BUNDLES.values():
        assert set(bundle.repair_on) <= set(get_args(RepairFinding))


# ---------------------------------------------------------------------------
# Equivalence with the inline repair formula these findings replaced. Since the
# unavailable-evidence fix, missing or NaN metrics never trigger repair (where
# the old formula raised).
# ---------------------------------------------------------------------------


def _outcome(decide: Callable[..., bool], *args: Any) -> bool | type[Exception]:
    try:
        return bool(decide(*args))
    except Exception as exc:
        return type(exc)


def _old_repair(row: dict[str, Any], ev: EvaluationCriteria) -> bool:
    if ev.repair_any_high_leak and bool(row.get(COL_ANY_HIGH_LEAKED, False)):
        return True
    return float(row.get(COL_LEAKAGE_MASS, 0.0)) > ev.repair_threshold


_FLAGS = (True, False, None, _NAN, _MISSING)
_NUMBERS = (0.0, 0.45, 0.65, 1.2, 1.7, 2.2, 3.5, _NAN, None)


@pytest.mark.parametrize("risk_tolerance", list(RiskTolerance))
def test_findings_reproduce_previous_repair_formula(risk_tolerance: RiskTolerance) -> None:
    ev = EvaluationCriteria(risk_tolerance=risk_tolerance)
    for high, leakage in itertools.product(_FLAGS, (*_NUMBERS, _MISSING)):
        row = _row(**{COL_ANY_HIGH_LEAKED: high, COL_LEAKAGE_MASS: leakage})
        old = _outcome(_old_repair, row, ev)
        assert _outcome(needs_repair, row, ev) == (False if old is TypeError else old), row
