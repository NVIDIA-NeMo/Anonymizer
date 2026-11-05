# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What each finding means for one evaluated rewrite row.

Each risk-tolerance preset lists, as data, which findings send a row to repair and which flag it
for human review (``_RISK_TOLERANCE_BUNDLES`` in ``anonymizer.config.rewrite``). To add a finding,
add a ``Finding`` member there (and to ``RepairFinding`` if repair can act on it), its condition in
``present_findings`` here, and list it in the presets that should act on it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from anonymizer.config.rewrite import EvaluationCriteria, Finding
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_LEAKAGE_MASS,
    COL_REWRITE_REPLACEMENT_READY,
    COL_REWRITTEN_TEXT,
    COL_UTILITY_SCORE,
)


def present_findings(row: Mapping[str, Any], evaluation: EvaluationCriteria) -> frozenset[Finding]:
    """Return the findings that hold for ``row`` under the preset's thresholds.

    A missing or NaN metric never crosses a threshold; it raises ``unavailable_evidence`` instead.
    """
    leakage = _metric(row.get(COL_LEAKAGE_MASS))
    utility = _metric(row.get(COL_UTILITY_SCORE))
    high_leaked = row.get(COL_ANY_HIGH_LEAKED)
    holds = {
        Finding.rewrite_missing: pd.isna(row.get(COL_REWRITTEN_TEXT)),
        Finding.replacement_unavailable: not row.get(COL_REWRITE_REPLACEMENT_READY, True),
        Finding.high_leak: bool(high_leaked),
        Finding.leakage_above_repair: leakage is not None and leakage > evaluation.repair_threshold,
        Finding.leakage_above_flag: leakage is not None and leakage > evaluation.flag_leakage_above,
        Finding.utility_below_flag: utility is not None and utility < evaluation.flag_utility_below,
        Finding.unavailable_evidence: leakage is None or utility is None or pd.isna(high_leaked),
    }
    return frozenset(finding for finding, held in holds.items() if held)


def needs_repair(row: Mapping[str, Any], evaluation: EvaluationCriteria) -> bool:
    """True when a finding the preset repairs on holds."""
    return not present_findings(row, evaluation).isdisjoint(evaluation.repair_on)


def needs_human_review(row: Mapping[str, Any], evaluation: EvaluationCriteria) -> bool:
    """True when a finding the preset flags for review holds."""
    return not present_findings(row, evaluation).isdisjoint(evaluation.flag_on)


def _metric(value: Any) -> float | None:
    return None if pd.isna(value) else float(value)
