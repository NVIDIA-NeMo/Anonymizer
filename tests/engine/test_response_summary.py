# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import random

import pandas as pd
import pytest

from anonymizer.engine.constants import (
    COL_DOMAIN,
    COL_DOMAIN_UNAVAILABLE,
    COL_EVALUATION_UNAVAILABLE,
    COL_FULL_REWRITE,
    COL_MEANING_UNITS,
    COL_PRIVACY_QA_REANSWER,
    COL_QUALITY_QA,
    COL_QUALITY_QA_COMPARE,
    COL_QUALITY_QA_REANSWER,
    COL_REWRITTEN_TEXT,
)
from anonymizer.engine.rewrite.domain_classification import UNAVAILABLE_DOMAIN
from anonymizer.engine.rewrite.response_summary import mark_evaluation_unavailable, warn_kept_without_response

_RESPONSE = {"answers": []}
_DOMAIN = {"domain": "FINANCIAL", "domain_confidence": 0.9}


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        COL_DOMAIN: _DOMAIN,
        COL_DOMAIN_UNAVAILABLE: False,
        COL_MEANING_UNITS: {"units": []},
        COL_QUALITY_QA: {"items": []},
        COL_FULL_REWRITE: {"rewritten_text": "Secret value 7312."},
        COL_REWRITTEN_TEXT: "Secret value 7312.",
        COL_QUALITY_QA_REANSWER: _RESPONSE,
        COL_PRIVACY_QA_REANSWER: _RESPONSE,
        COL_QUALITY_QA_COMPARE: {"per_item": []},
    }
    return {**row, **overrides}


def test_one_warning_counts_each_step_and_holds_no_values(caplog: pytest.LogCaptureFixture) -> None:
    frame = pd.DataFrame(
        [
            _row(),
            _row(**{COL_DOMAIN: dict(UNAVAILABLE_DOMAIN), COL_DOMAIN_UNAVAILABLE: True}),
            # The rewrite never parsed: nothing downstream was asked, and only the rewrite is counted.
            _row(
                **{
                    COL_FULL_REWRITE: None,
                    COL_REWRITTEN_TEXT: None,
                    COL_PRIVACY_QA_REANSWER: None,
                    COL_QUALITY_QA_REANSWER: None,
                    COL_QUALITY_QA_COMPARE: None,
                }
            ),
            _row(**{COL_FULL_REWRITE: None, COL_REWRITTEN_TEXT: None, COL_PRIVACY_QA_REANSWER: float("nan")}),
            _row(**{COL_PRIVACY_QA_REANSWER: None}),
            # Meaning units never parsed: the quality questions were not asked and are not counted again.
            _row(**{COL_MEANING_UNITS: None, COL_QUALITY_QA: None, COL_QUALITY_QA_REANSWER: None}),
            _row(**{COL_QUALITY_QA: None, COL_QUALITY_QA_REANSWER: None, COL_QUALITY_QA_COMPARE: None}),
            _row(**{COL_QUALITY_QA_COMPARE: None}),
        ]
    )

    with caplog.at_level(logging.WARNING, logger="anonymizer.rewrite.responses"):
        warn_kept_without_response(frame)

    (message,) = {record.getMessage() for record in caplog.records if record.name == "anonymizer.rewrite.responses"}
    assert message.startswith(
        "Rows kept without a usable model response after their retries, by step: domain classification=1, "
        "meaning units=1, quality questions=1, rewrite=2, privacy evaluation=1, quality evaluation=1."
    )
    assert "7312" not in caplog.text


def test_a_valid_other_response_with_no_confidence_is_not_counted_as_the_fallback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The model said OTHER at confidence 0.0: the same value as the fallback, but a response that was used.
    assert {"domain": "OTHER", "domain_confidence": 0.0} == UNAVAILABLE_DOMAIN
    frame = pd.DataFrame([_row(**{COL_DOMAIN: dict(UNAVAILABLE_DOMAIN), COL_DOMAIN_UNAVAILABLE: False})])

    with caplog.at_level(logging.WARNING, logger="anonymizer.rewrite.responses"):
        warn_kept_without_response(frame)

    assert [record for record in caplog.records if record.name == "anonymizer.rewrite.responses"] == []


def test_the_marker_is_internal_and_leaves_the_frame_after_the_summary() -> None:
    frame = pd.DataFrame([_row()])

    warn_kept_without_response(frame)

    assert COL_DOMAIN_UNAVAILABLE not in frame.columns


def test_no_warning_when_every_response_was_used(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="anonymizer.rewrite.responses"):
        warn_kept_without_response(
            pd.DataFrame([_row(), _row(**{COL_DOMAIN: '{"domain": "OTHER", "domain_confidence": 0.4}'})])
        )

    assert [record for record in caplog.records if record.name == "anonymizer.rewrite.responses"] == []


def test_frames_without_the_columns_log_nothing(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="anonymizer.rewrite.responses"):
        warn_kept_without_response(pd.DataFrame({"text": ["a"]}))

    assert [record for record in caplog.records if record.name == "anonymizer.rewrite.responses"] == []


def _summary(caplog: pytest.LogCaptureFixture) -> str:
    (message,) = {record.getMessage() for record in caplog.records if record.name == "anonymizer.rewrite.responses"}
    return message


def test_the_evaluation_marker_keeps_earlier_rounds_and_leaves_the_frame(caplog: pytest.LogCaptureFixture) -> None:
    frame = pd.DataFrame([_row(**{COL_QUALITY_QA_REANSWER: None}), _row(**{COL_PRIVACY_QA_REANSWER: None}), _row()])
    mark_evaluation_unavailable(frame)
    # The next round answers every step and overwrites the columns.
    for column in (COL_QUALITY_QA_REANSWER, COL_PRIVACY_QA_REANSWER):
        frame[column] = [_RESPONSE] * len(frame)
    mark_evaluation_unavailable(frame)

    with caplog.at_level(logging.WARNING, logger="anonymizer.rewrite.responses"):
        warn_kept_without_response(frame)

    assert _summary(caplog).startswith(
        "Rows kept without a usable model response after their retries, by step: "
        "privacy evaluation=1, quality evaluation=1."
    )
    assert COL_EVALUATION_UNAVAILABLE not in frame.columns


@pytest.mark.parametrize("seed", range(20))
def test_each_evaluation_step_counts_the_rows_without_a_response_in_any_round(
    seed: int, caplog: pytest.LogCaptureFixture
) -> None:
    """Rounds overwrite the evaluation columns; the count is the rows that had no response in at least one round."""
    rng = random.Random(seed)
    rows, rounds = rng.randint(1, 12), rng.randint(1, 4)
    steps = {
        "quality": COL_QUALITY_QA_REANSWER,
        "compare": COL_QUALITY_QA_COMPARE,
        "privacy": COL_PRIVACY_QA_REANSWER,
    }
    missing = {step: [[rng.random() < 0.2 for _ in range(rounds)] for _ in range(rows)] for step in steps}
    frame = pd.DataFrame([_row() for _ in range(rows)])
    for k in range(rounds):
        for step, column in steps.items():
            good = {"per_item": []} if step == "compare" else _RESPONSE
            frame[column] = [None if missing[step][i][k] else good for i in range(rows)]
        # Without the quality answers the comparison is not asked, so it has nothing to miss.
        frame.loc[frame[COL_QUALITY_QA_REANSWER].isna(), COL_QUALITY_QA_COMPARE] = None
        if k < rounds - 1:  # the last round is read from the frame
            mark_evaluation_unavailable(frame)

    def rows_missing(*names: str) -> int:
        return sum(any(missing[name][i][k] for name in names for k in range(rounds)) for i in range(rows))

    expected = {
        "privacy evaluation": rows_missing("privacy"),
        "quality evaluation": rows_missing("quality", "compare"),
    }
    with caplog.at_level(logging.WARNING, logger="anonymizer.rewrite.responses"):
        warn_kept_without_response(frame)

    # One record, which caplog can list twice once DataDesigner has reconfigured logging in an earlier test.
    counted = [r.getMessage() for r in dict.fromkeys(caplog.records) if r.name == "anonymizer.rewrite.responses"]
    shown = ", ".join(f"{step}={count}" for step, count in expected.items() if count)
    message = f"Rows kept without a usable model response after their retries, by step: {shown}. "
    assert counted == ([message + "If most rows are listed, check that step's model."] if shown else [])
