# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Invocation-scoped row accounting and accepted-detection verification.

This module is intentionally private: it is an engine invariant, not a result,
trace, or dataframe API.  Correlations and frozen entity values never leave an
invocation, and the verifier is invalidated before a result is returned.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from anonymizer.engine.constants import COL_FINAL_ENTITIES

if TYPE_CHECKING:
    import pandas as pd


PRIVATE_CORRELATION_COLUMN = "__anonymizer_private_row_correlation__"


class _TerminalOutcome(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class _SafeFailure:
    code: str
    stage: str
    scope: str
    correlation: str | None = None
    retry_owner: str = "anonymizer"
    message: str = "private row verification failed"


class PrivateRowVerificationError(RuntimeError):
    """Sanitized private-engine error; deliberately carries no causal exception."""

    def __init__(self, failure: _SafeFailure) -> None:
        self.failure = failure
        correlation = f" correlation={failure.correlation}" if failure.correlation is not None else ""
        super().__init__(
            f"private_row_verification code={failure.code} stage={failure.stage} "
            f"scope={failure.scope} retry_owner={failure.retry_owner}{correlation}: {failure.message}"
        )


class _InvocationRowVerifier:
    """One-shot verifier for one private engine invocation."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        if PRIVATE_CORRELATION_COLUMN in dataframe.columns:
            raise PrivateRowVerificationError(
                _SafeFailure("private_column_collision", "accept", "invocation", message="reserved private column")
            )
        self._active = True
        self._accepted = tuple(uuid.uuid4().hex for _ in range(len(dataframe)))
        self._frozen: dict[str, str] = {}
        self._outcomes: dict[str, _TerminalOutcome] = {}

    def bind(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        self._require_active()
        bound = dataframe.copy()
        bound[PRIVATE_CORRELATION_COLUMN] = list(self._accepted)
        return bound

    def bind_complete_stage_output(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Attach correlations only to a complete, order-preserving legacy result.

        This keeps older in-process workflow implementations compatible while
        refusing to infer provenance for a dropped, duplicated, or partial
        result. Real engine stages carry the column themselves.
        """
        self._require_active()
        if PRIVATE_CORRELATION_COLUMN in dataframe.columns:
            return dataframe
        expected = tuple(correlation for correlation in self._accepted if correlation not in self._outcomes)
        if len(dataframe) != len(expected):
            raise self._error("correlation_missing", "stage_boundary", "invocation")
        bound = dataframe.copy()
        bound[PRIVATE_CORRELATION_COLUMN] = expected
        return bound

    def freeze_accepted_detections(self, dataframe: pd.DataFrame) -> None:
        self._require_active()
        correlations = self._validate_correlations(dataframe, stage="detection")
        if COL_FINAL_ENTITIES not in dataframe.columns:
            raise self._error("accepted_detection_missing", "detection", "invocation")
        self._frozen = {
            correlation: _stable_digest(value)
            for correlation, value in zip(correlations, dataframe[COL_FINAL_ENTITIES], strict=True)
        }
        self._mark_absent_as_failed(correlations)

    def finish(self, dataframe: pd.DataFrame, *, cancelled: bool = False) -> pd.DataFrame:
        """Verify terminal cardinality and accepted detections, then remove state."""
        self._require_active()
        try:
            correlations = self._validate_correlations(dataframe, stage="result")
            self._mark_absent_as_failed(correlations)
            expected_successes = set(self._accepted) - set(self._outcomes)
            if set(correlations) != expected_successes:
                raise self._error("terminal_row_mismatch", "result", "row")
            if COL_FINAL_ENTITIES in dataframe.columns:
                for correlation, value in zip(correlations, dataframe[COL_FINAL_ENTITIES], strict=True):
                    if self._frozen[correlation] != _stable_digest(value):
                        raise self._error("accepted_detection_tampered", "result", "row", correlation)
            self._outcomes.update({correlation: _TerminalOutcome.SUCCESS for correlation in correlations})
            return dataframe.drop(columns=[PRIVATE_CORRELATION_COLUMN], errors="ignore")
        finally:
            self._active = False
            self._frozen.clear()

    def abort(self, *, cancelled: bool) -> None:
        """Close an interrupted invocation with one terminal outcome per accepted row."""
        if not self._active:
            return
        terminal = _TerminalOutcome.CANCELLED if cancelled else _TerminalOutcome.FAILED
        for correlation in self._accepted:
            self._outcomes.setdefault(correlation, terminal)
        self._active = False
        self._frozen.clear()

    def abort_with_failure(self, *, stage: str, cause: BaseException) -> None:
        """Close an invocation and raise a failure that cannot expose ``cause``."""
        del cause
        self.abort(cancelled=False)
        raise self._error("invocation_failed", stage, "invocation") from None

    def _validate_correlations(self, dataframe: pd.DataFrame, *, stage: str) -> list[str]:
        if PRIVATE_CORRELATION_COLUMN not in dataframe.columns:
            raise self._error("correlation_missing", stage, "invocation")
        correlations = dataframe[PRIVATE_CORRELATION_COLUMN].tolist()
        if not all(isinstance(value, str) and value for value in correlations):
            raise self._error("correlation_invalid", stage, "row")
        if len(set(correlations)) != len(correlations):
            raise self._error("correlation_duplicate", stage, "row")
        unknown = set(correlations) - set(self._accepted)
        if unknown:
            raise self._error("correlation_unknown", stage, "row")
        return correlations

    def _mark_absent_as_failed(self, correlations: list[str]) -> None:
        """Record dropped accepted rows before any invocation-wide terminal state.

        Row failure has precedence over later cancellation or invocation failure.
        A surviving row is never inferred from its contents; only its private
        correlation proves provenance.
        """
        for correlation in set(self._accepted) - set(correlations):
            self._outcomes.setdefault(correlation, _TerminalOutcome.FAILED)

    def _error(self, code: str, stage: str, scope: str, correlation: str | None = None) -> PrivateRowVerificationError:
        return PrivateRowVerificationError(_SafeFailure(code, stage, scope, correlation=correlation))

    def _require_active(self) -> None:
        if not self._active:
            raise PrivateRowVerificationError(_SafeFailure("invocation_closed", "lifecycle", "invocation"))

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private invocation verifier is not serializable")

    def __repr__(self) -> str:
        return "<private invocation row verifier>"


def _stable_digest(value: object) -> str:
    """Hash private detection data without serializing it into a public artifact."""
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    except (TypeError, ValueError):
        encoded = repr(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
