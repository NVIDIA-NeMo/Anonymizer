# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named findings behind the rewrite repair and human-review decisions.

A finding is a named condition on one evaluated rewrite row. Each risk-tolerance
preset lists, as data, which findings send a row to repair and which flag it for
human review (``_RISK_TOLERANCE_BUNDLES`` in ``anonymizer.config.rewrite``).
This module only says what each finding means. To add one, add a member and a
rule here, then list it in the presets that should act on it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import pandas as pd

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
from anonymizer.engine.rewrite.parsers import parse_privacy_answers, parse_privacy_qa, parse_sensitivity_disposition
from anonymizer.engine.schemas.rewrite import PrivacyAnswer

if TYPE_CHECKING:
    from anonymizer.config.rewrite import EvaluationCriteria


class Finding(str, Enum):
    """Named conditions a preset can repair on or flag for human review."""

    rewrite_missing = "rewrite_missing"
    replacement_unavailable = "replacement_unavailable"
    high_leak = "high_leak"
    protected_leak = "protected_leak"
    leakage_above_threshold = "leakage_above_threshold"
    utility_below_flag = "utility_below_flag"
    unavailable_evidence = "unavailable_evidence"


@dataclass(frozen=True)
class _Limits:
    leakage_above: float
    utility_below: float


def _protected_leak(row: Mapping[str, Any]) -> bool:
    """A privacy question answered "yes" for an entity whose disposition is not ``leave_as_is``."""
    questions = {
        item.id: (item.entity_label, item.entity_value) for item in parse_privacy_qa(row.get(COL_PRIVACY_QA)).items
    }
    disposition = parse_sensitivity_disposition(row.get(COL_SENSITIVITY_DISPOSITION))
    protected = {(entity.entity_label, entity.entity_value) for entity in disposition.protected_entities}
    return any(
        answer.answer == PrivacyAnswer.yes and questions.get(answer.id) in protected
        for answer in parse_privacy_answers(row.get(COL_PRIVACY_QA_REANSWER))
    )


def _number(value: Any) -> float:
    """Unavailable evidence (None/NaN) reads as NaN, so it never passes or fails a threshold."""
    return math.nan if pd.isna(value) else float(value)


_RULES: dict[Finding, Callable[[Mapping[str, Any], _Limits], bool]] = {
    Finding.rewrite_missing: lambda row, _: bool(pd.isna(row.get(COL_REWRITTEN_TEXT))),
    Finding.replacement_unavailable: lambda row, _: not bool(row.get(COL_REWRITE_REPLACEMENT_READY, True)),
    Finding.high_leak: lambda row, _: bool(row.get(COL_ANY_HIGH_LEAKED)),
    Finding.protected_leak: lambda row, _: _protected_leak(row),
    Finding.leakage_above_threshold: lambda row, limits: _number(row.get(COL_LEAKAGE_MASS)) > limits.leakage_above,
    Finding.utility_below_flag: lambda row, limits: _number(row.get(COL_UTILITY_SCORE)) < limits.utility_below,
    Finding.unavailable_evidence: lambda row, _: any(
        bool(pd.isna(row.get(column))) for column in (COL_LEAKAGE_MASS, COL_UTILITY_SCORE, COL_ANY_HIGH_LEAKED)
    ),
}


def present_findings(
    row: Mapping[str, Any], among: Iterable[Finding], *, leakage_above: float, utility_below: float
) -> frozenset[Finding]:
    """Return the findings from ``among`` that hold for ``row``."""
    limits = _Limits(leakage_above=leakage_above, utility_below=utility_below)
    return frozenset(finding for finding in among if _RULES[finding](row, limits))


def needs_repair(row: Mapping[str, Any], evaluation: EvaluationCriteria) -> bool:
    """True when any of the preset's repair findings holds, against the repair threshold."""
    limits = _Limits(leakage_above=evaluation.repair_threshold, utility_below=evaluation.flag_utility_below)
    return any(_RULES[finding](row, limits) for finding in evaluation.repair_on)


def needs_human_review(row: Mapping[str, Any], evaluation: EvaluationCriteria) -> bool:
    """True when any of the preset's flag findings holds, against the review thresholds."""
    limits = _Limits(leakage_above=evaluation.flag_leakage_above, utility_below=evaluation.flag_utility_below)
    return any(_RULES[finding](row, limits) for finding in evaluation.flag_on)
