# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One end-of-run warning counting the rows each Rewrite step kept without a usable model response."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import pandas as pd

from anonymizer.engine.constants import (
    COL_DOMAIN_UNAVAILABLE,
    COL_EVALUATION_UNAVAILABLE,
    COL_FULL_REWRITE,
    COL_MEANING_UNITS,
    COL_PRIVACY_QA_REANSWER,
    COL_QUALITY_QA,
    COL_QUALITY_QA_COMPARE,
    COL_QUALITY_QA_REANSWER,
    COL_REPAIR_ITERATIONS,
    COL_REWRITTEN_TEXT,
    COL_SENSITIVITY_DISPOSITION,
)
from anonymizer.engine.ndd.model_response import no_response

logger = logging.getLogger("anonymizer.rewrite.responses")

QUALITY_EVALUATION_UNAVAILABLE = 1
PRIVACY_EVALUATION_UNAVAILABLE = 2


def evaluation_unavailable(
    row: Mapping[str, object],
    *,
    rewritten_text: str = COL_REWRITTEN_TEXT,
    quality_reanswer: str = COL_QUALITY_QA_REANSWER,
    quality_compare: str = COL_QUALITY_QA_COMPARE,
    privacy_reanswer: str = COL_PRIVACY_QA_REANSWER,
) -> int:
    """The evaluation steps of one round that kept ``row`` without a usable response, as bit flags.

    The keywords name the round's columns (combined mode keeps one set per round). A step is counted only when it
    was asked: the privacy and quality answers need the rewritten text (and quality its questions), the comparison
    needs the quality answers.
    """
    text = not no_response(row.get(rewritten_text))
    answers = not no_response(row.get(quality_reanswer))
    quality = (not answers and text and not no_response(row.get(COL_QUALITY_QA))) or (
        answers and no_response(row.get(quality_compare))
    )
    privacy = text and no_response(row.get(privacy_reanswer))
    return QUALITY_EVALUATION_UNAVAILABLE * quality | PRIVACY_EVALUATION_UNAVAILABLE * privacy


def mark_evaluation_unavailable(frame: pd.DataFrame) -> None:
    """OR this round's ``evaluation_unavailable`` flags into the row's marker, which later rounds do not overwrite."""
    current = pd.Series([evaluation_unavailable(row) for row in frame.to_dict("records")], index=frame.index)
    earlier = _marker(frame)
    frame[COL_EVALUATION_UNAVAILABLE] = (earlier | current).astype(int)


def _marker(frame: pd.DataFrame) -> pd.Series:
    if COL_EVALUATION_UNAVAILABLE not in frame.columns:
        return pd.Series(0, index=frame.index, dtype=int)
    return frame[COL_EVALUATION_UNAVAILABLE].fillna(0).astype(int)


def warn_kept_without_response(frame: pd.DataFrame) -> None:
    """Log one WARNING with the number of rows each step kept without a usable response, when there are any.

    Each such row already logged its own warning; this makes a model that never returns a parsable response visible
    once per run. Only counts are logged, never values. Call it before the rewritten text is withheld. It removes the
    internal domain and evaluation marker columns from ``frame``, which are not part of the trace. An evaluation step
    counts a row that had no response in any round: the marker holds the rounds the repair loop has replaced, and the
    final round is read from the frame.
    """

    def null(column: str) -> pd.Series:
        if column not in frame.columns:
            return pd.Series(False, index=frame.index)
        return frame[column].map(no_response).astype(bool)

    def present(column: str) -> pd.Series:
        return ~null(column) if column in frame.columns else pd.Series(False, index=frame.index)

    # A response can equal the fallback value, so the domain step marks when it wrote the fallback.
    domain = (
        frame.pop(COL_DOMAIN_UNAVAILABLE).eq(True)
        if COL_DOMAIN_UNAVAILABLE in frame.columns
        else pd.Series(False, index=frame.index)
    )
    evaluation = _marker(frame) | pd.Series(
        [evaluation_unavailable(row) for row in frame.to_dict("records")], index=frame.index, dtype=int
    )
    frame.drop(columns=[COL_EVALUATION_UNAVAILABLE], errors="ignore", inplace=True)
    # Only an unparsed repair response leaves a repaired row without text; the rewrite step's rows are never repaired.
    repaired = (
        frame[COL_REPAIR_ITERATIONS].fillna(0).astype(int).gt(0)
        if COL_REPAIR_ITERATIONS in frame.columns
        else pd.Series(False, index=frame.index)
    )
    # A row without a disposition asks neither the meaning units nor the rewrite, so only the disposition counts it.
    disposition = null(COL_SENSITIVITY_DISPOSITION)
    steps = {
        "domain classification": domain,
        "sensitivity disposition": disposition,
        "meaning units": null(COL_MEANING_UNITS) & ~disposition,
        "quality questions": null(COL_QUALITY_QA) & present(COL_MEANING_UNITS),
        "rewrite": null(COL_FULL_REWRITE) & ~disposition,
        "privacy evaluation": (evaluation & PRIVACY_EVALUATION_UNAVAILABLE).astype(bool),
        "quality evaluation": (evaluation & QUALITY_EVALUATION_UNAVAILABLE).astype(bool),
        "repair": null(COL_REWRITTEN_TEXT) & repaired,
    }
    counts = {step: int(rows.sum()) for step, rows in steps.items() if rows.any()}
    if counts:
        logger.warning(
            "Rows kept without a usable model response after their retries, by step: %s. "
            "If most rows are listed, check that step's model.",
            ", ".join(f"{step}={count}" for step, count in counts.items()),
        )
