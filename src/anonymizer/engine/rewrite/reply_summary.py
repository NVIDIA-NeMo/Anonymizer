# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One end-of-run warning counting the rows each Rewrite step kept without a usable model reply."""

from __future__ import annotations

import logging

import pandas as pd

from anonymizer.engine.constants import (
    COL_DOMAIN_UNAVAILABLE,
    COL_FULL_REWRITE,
    COL_MEANING_UNITS,
    COL_PRIVACY_QA_REANSWER,
    COL_QUALITY_QA,
    COL_QUALITY_QA_COMPARE,
    COL_QUALITY_QA_REANSWER,
    COL_REWRITTEN_TEXT,
)
from anonymizer.engine.ndd.model_reply import no_reply

logger = logging.getLogger("anonymizer.rewrite.replies")


def warn_kept_without_reply(frame: pd.DataFrame) -> None:
    """Log one WARNING with the number of rows each step kept without a usable reply, when there are any.

    Each such row already logged its own warning; this makes a model that never returns a parsable reply visible
    once per run. Only counts are logged, never values. Call it before the rewritten text is withheld. It removes the
    internal domain marker column from ``frame``, which is not part of the trace.
    """

    def null(column: str) -> pd.Series:
        if column not in frame.columns:
            return pd.Series(False, index=frame.index)
        return frame[column].map(no_reply).astype(bool)

    def present(column: str) -> pd.Series:
        return ~null(column) if column in frame.columns else pd.Series(False, index=frame.index)

    # A reply can equal the fallback value, so the domain step marks when it wrote the fallback.
    domain = (
        frame.pop(COL_DOMAIN_UNAVAILABLE).eq(True)
        if COL_DOMAIN_UNAVAILABLE in frame.columns
        else pd.Series(False, index=frame.index)
    )
    steps = {
        "domain classification": domain,
        "meaning units": null(COL_MEANING_UNITS),
        "quality questions": null(COL_QUALITY_QA) & present(COL_MEANING_UNITS),
        "rewrite": null(COL_FULL_REWRITE),
        "privacy evaluation": null(COL_PRIVACY_QA_REANSWER) & present(COL_REWRITTEN_TEXT),
        "quality evaluation": (null(COL_QUALITY_QA_REANSWER) & present(COL_QUALITY_QA) & present(COL_REWRITTEN_TEXT))
        | (null(COL_QUALITY_QA_COMPARE) & present(COL_QUALITY_QA_REANSWER)),
    }
    counts = {step: int(rows.sum()) for step, rows in steps.items() if rows.any()}
    if counts:
        logger.warning(
            "Rows kept without a usable model reply after their retries, by step: %s. "
            "If most rows are listed, check that step's model.",
            ", ".join(f"{step}={count}" for step, count in counts.items()),
        )
