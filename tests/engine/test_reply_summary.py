# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import pandas as pd
import pytest

from anonymizer.engine.constants import (
    COL_DOMAIN,
    COL_FULL_REWRITE,
    COL_MEANING_UNITS,
    COL_PRIVACY_QA_REANSWER,
    COL_QUALITY_QA,
    COL_QUALITY_QA_COMPARE,
    COL_QUALITY_QA_REANSWER,
    COL_REWRITTEN_TEXT,
)
from anonymizer.engine.rewrite.domain_classification import UNAVAILABLE_DOMAIN
from anonymizer.engine.rewrite.reply_summary import warn_kept_without_reply

_REPLY = {"answers": []}
_DOMAIN = {"domain": "FINANCIAL", "domain_confidence": 0.9}


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        COL_DOMAIN: _DOMAIN,
        COL_MEANING_UNITS: {"units": []},
        COL_QUALITY_QA: {"items": []},
        COL_FULL_REWRITE: {"rewritten_text": "Secret value 7312."},
        COL_REWRITTEN_TEXT: "Secret value 7312.",
        COL_QUALITY_QA_REANSWER: _REPLY,
        COL_PRIVACY_QA_REANSWER: _REPLY,
        COL_QUALITY_QA_COMPARE: {"per_item": []},
    }
    return {**row, **overrides}


def test_one_warning_counts_each_step_and_holds_no_values(caplog: pytest.LogCaptureFixture) -> None:
    frame = pd.DataFrame(
        [
            _row(),
            _row(**{COL_DOMAIN: dict(UNAVAILABLE_DOMAIN)}),
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

    with caplog.at_level(logging.WARNING, logger="anonymizer.rewrite.replies"):
        warn_kept_without_reply(frame)

    (message,) = {record.getMessage() for record in caplog.records if record.name == "anonymizer.rewrite.replies"}
    assert message.startswith(
        "Rows kept without a usable model reply after their retries, by step: domain classification=1, "
        "meaning units=1, quality questions=1, rewrite=2, privacy evaluation=1, quality evaluation=1."
    )
    assert "7312" not in caplog.text


def test_no_warning_when_every_reply_was_used(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="anonymizer.rewrite.replies"):
        warn_kept_without_reply(
            pd.DataFrame([_row(), _row(**{COL_DOMAIN: '{"domain": "OTHER", "domain_confidence": 0.4}'})])
        )

    assert [record for record in caplog.records if record.name == "anonymizer.rewrite.replies"] == []


def test_frames_without_the_columns_log_nothing(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="anonymizer.rewrite.replies"):
        warn_kept_without_reply(pd.DataFrame({"text": ["a"]}))

    assert [record for record in caplog.records if record.name == "anonymizer.rewrite.replies"] == []
