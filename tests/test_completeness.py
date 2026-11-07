# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail when a new enum member, workflow name or column literal lacks its mapping entry."""

from __future__ import annotations

import re
import typing
from pathlib import Path

import pytest

import anonymizer
from anonymizer.config.rewrite import _RISK_TOLERANCE_BUNDLES, SENSITIVITY_WEIGHTS, RiskTolerance
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_ATTRIBUTE_FIDELITY_VALID,
    COL_DETECTION_VALID,
    COL_LEAKAGE_MASS,
    COL_NEEDS_HUMAN_REVIEW,
    COL_RELATIONAL_CONSISTENCY_VALID,
    COL_TYPE_FIDELITY_VALID,
    COL_UTILITY_SCORE,
    COL_WEIGHTED_LEAKAGE_RATE,
)
from anonymizer.engine.schemas.rewrite import SensitivityLevel
from anonymizer.interface.anonymizer import _step_to_field
from anonymizer.measurement.metrics.rewrite import _rewrite_record_fields
from anonymizer.measurement.records.row import _EvaluationBoolField

_SRC = Path(anonymizer.__file__).parent
_WORKFLOW_NAME_RE = re.compile(r"""workflow_name=f?["']([^"'{]+)""")


def _workflow_names() -> set[str]:
    return {m.group(1) for path in _SRC.rglob("*.py") for m in _WORKFLOW_NAME_RE.finditer(path.read_text())}


def test_workflow_name_scan_finds_literals() -> None:
    assert {"entity-detection", "rewrite-combined", "replace-judges"} <= _workflow_names()


# Steps that run only from evaluate(), which emits no telemetry, so no failure count reads them.
_EVALUATE_ONLY = {"replace-judges"}


@pytest.mark.parametrize("name", sorted(_workflow_names() - _EVALUATE_ONLY))
def test_workflow_name_has_failure_count_key(name: str) -> None:
    # f-string names like ``rewrite-repair-{iteration}`` are captured by their prefix.
    step = f"{name}0" if name.endswith("-") else name
    assert _step_to_field(step) != "unknown"


@pytest.mark.parametrize("tolerance", list(RiskTolerance))
def test_risk_tolerance_has_preset(tolerance: RiskTolerance) -> None:
    assert tolerance in _RISK_TOLERANCE_BUNDLES


@pytest.mark.parametrize("level", list(SensitivityLevel))
def test_sensitivity_level_has_weight(level: SensitivityLevel) -> None:
    assert level.value in SENSITIVITY_WEIGHTS


def test_rewrite_metric_fields_match_column_constants() -> None:
    fields = _rewrite_record_fields({}, columns=set())
    for col in (
        COL_UTILITY_SCORE,
        COL_LEAKAGE_MASS,
        COL_WEIGHTED_LEAKAGE_RATE,
        COL_ANY_HIGH_LEAKED,
        COL_NEEDS_HUMAN_REVIEW,
    ):
        assert col in fields


def test_evaluation_bool_fields_match_column_constants() -> None:
    assert set(typing.get_args(_EvaluationBoolField)) == {
        COL_DETECTION_VALID,
        COL_TYPE_FIDELITY_VALID,
        COL_RELATIONAL_CONSISTENCY_VALID,
        COL_ATTRIBUTE_FIDELITY_VALID,
    }
