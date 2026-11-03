# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest

from anonymizer.config.rewrite import _RISK_TOLERANCE_BUNDLES, EvaluationCriteria, RiskTolerance
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_LEAKAGE_MASS,
    COL_PRIVACY_QA,
    COL_PRIVACY_QA_REANSWER,
    COL_REWRITE_REPLACEMENT_READY,
    COL_REWRITTEN_TEXT,
    COL_SENSITIVITY_DISPOSITION,
    COL_UTILITY_SCORE,
)
from anonymizer.engine.rewrite.findings import _RULES, Finding, needs_human_review, needs_repair, present_findings
from anonymizer.engine.rewrite.qa_generation import generate_privacy_qa_from_disposition
from anonymizer.engine.schemas.rewrite import PrivacyAnswersSchema, SensitivityDispositionSchema

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


@pytest.mark.parametrize(
    ("finding", "overrides", "expected"),
    [
        (Finding.rewrite_missing, {}, False),
        (Finding.rewrite_missing, {COL_REWRITTEN_TEXT: None}, True),
        (Finding.rewrite_missing, {COL_REWRITTEN_TEXT: _NAN}, True),
        (Finding.replacement_unavailable, {COL_REWRITE_REPLACEMENT_READY: False}, True),
        (Finding.replacement_unavailable, {COL_REWRITE_REPLACEMENT_READY: None}, True),
        (Finding.replacement_unavailable, {COL_REWRITE_REPLACEMENT_READY: _MISSING}, False),
        (Finding.high_leak, {COL_ANY_HIGH_LEAKED: True}, True),
        (Finding.high_leak, {COL_ANY_HIGH_LEAKED: None}, False),
        (Finding.leakage_above_threshold, {COL_LEAKAGE_MASS: 1.0}, False),
        (Finding.leakage_above_threshold, {COL_LEAKAGE_MASS: 1.01}, True),
        (Finding.leakage_above_threshold, {COL_LEAKAGE_MASS: _NAN}, False),
        (Finding.utility_below_flag, {COL_UTILITY_SCORE: 0.5}, False),
        (Finding.utility_below_flag, {COL_UTILITY_SCORE: 0.49}, True),
        (Finding.utility_below_flag, {COL_UTILITY_SCORE: _NAN}, False),
        (Finding.unavailable_evidence, {}, False),
        (Finding.unavailable_evidence, {COL_LEAKAGE_MASS: _NAN}, True),
        (Finding.unavailable_evidence, {COL_UTILITY_SCORE: _NAN}, True),
        (Finding.unavailable_evidence, {COL_ANY_HIGH_LEAKED: _MISSING}, True),
    ],
)
def test_finding_rule(finding: Finding, overrides: dict[str, Any], expected: bool) -> None:
    present = present_findings(_row(**overrides), [finding], leakage_above=1.0, utility_below=0.5)
    assert (finding in present) is expected


@pytest.mark.parametrize(("answer", "expected"), [("yes", True), ("no", False)])
def test_protected_leak_rule(answer: str, expected: bool) -> None:
    entity = {"source": "tagged", "category": "direct_identifier", "sensitivity": "high", "entity_label": "first_name"}
    disposition = SensitivityDispositionSchema.model_validate(
        {
            "sensitivity_disposition": [
                {**entity, "id": 1, "entity_value": "Alice", "protection_reason": "Names the patient directly.",
                 "protection_method_suggestion": "replace", "combined_risk_level": "high"},
                {**entity, "id": 2, "entity_value": "Bob", "protection_reason": "Public figure, no risk here.",
                 "protection_method_suggestion": "leave_as_is", "combined_risk_level": "low"},
            ]
        }
    )  # fmt: skip
    answers = PrivacyAnswersSchema.model_validate(
        {"answers": [{"id": 1, "answer": answer, "confidence": 1.0, "reason": "stub"}]}
    )
    row = _row(
        **{
            COL_SENSITIVITY_DISPOSITION: disposition,
            COL_PRIVACY_QA: generate_privacy_qa_from_disposition(disposition),
            COL_PRIVACY_QA_REANSWER: answers,
        }
    )
    present = present_findings(row, [Finding.protected_leak], leakage_above=1.0, utility_below=0.5)
    assert (Finding.protected_leak in present) is expected


def test_every_finding_has_a_rule_and_presets_name_only_known_findings() -> None:
    assert set(_RULES) == set(Finding)
    for bundle in _RISK_TOLERANCE_BUNDLES.values():
        assert set(bundle.repair_on) | set(bundle.flag_on) <= set(Finding)
        assert Finding.protected_leak not in set(bundle.repair_on) | set(bundle.flag_on)
        assert Finding.unavailable_evidence in bundle.flag_on
        assert Finding.unavailable_evidence not in bundle.repair_on


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
    return bool(present_findings(row, [Finding.unavailable_evidence], leakage_above=0.0, utility_below=0.0))


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
