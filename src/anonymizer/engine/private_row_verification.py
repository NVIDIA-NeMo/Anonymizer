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

from anonymizer.engine.constants import COL_FINAL_ENTITIES, COL_TARGET_WORK_ID, COL_TEXT

if TYPE_CHECKING:
    import pandas as pd


PRIVATE_CORRELATION_COLUMN = COL_TARGET_WORK_ID


class _TerminalOutcome(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class _SafeFailure:
    code: str
    stage: str
    scope: str
    retry_owner: str = "anonymizer"
    message: str = "private row verification failed"


class PrivateRowVerificationError(RuntimeError):
    """Sanitized private-engine error; deliberately carries no causal exception."""

    def __init__(self, failure: _SafeFailure) -> None:
        self.failure = failure
        super().__init__(
            f"private_row_verification code={failure.code} stage={failure.stage} "
            f"scope={failure.scope} retry_owner={failure.retry_owner}: {failure.message}"
        )


class _InvocationRowVerifier:
    """One-shot verifier for one private engine invocation."""

    def __init__(self, dataframe: pd.DataFrame, *, correlations: tuple[str, ...] | None = None) -> None:
        if PRIVATE_CORRELATION_COLUMN in dataframe.columns:
            raise PrivateRowVerificationError(
                _SafeFailure("private_column_collision", "accept", "invocation", message="reserved private column")
            )
        accepted = correlations if correlations is not None else tuple(uuid.uuid4().hex for _ in range(len(dataframe)))
        if (
            len(accepted) != len(dataframe)
            or len(set(accepted)) != len(accepted)
            or not all(isinstance(value, str) and value for value in accepted)
        ):
            raise PrivateRowVerificationError(
                _SafeFailure("correlation_invalid", "accept", "invocation", message="invalid private correlations")
            )
        self._active = True
        self._accepted = accepted
        self._legacy_input_order = tuple(_stable_digest(value) for value in dataframe[COL_TEXT])
        self._legacy_identity_is_unambiguous = len(set(self._legacy_input_order)) == len(self._legacy_input_order)
        self._frozen: dict[str, str] = {}
        self._outcomes: dict[str, _TerminalOutcome] = {}
        self._result_order: tuple[str, ...] = ()
        self._result_fingerprints: dict[str, str] = {}

    def bind(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        self._require_active()
        bound = dataframe.copy()
        bound[PRIVATE_CORRELATION_COLUMN] = list(self._accepted)
        return bound

    def bind_complete_stage_output(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Require a stage to preserve correlation or prove exact legacy order.

        Legacy in-process doubles may reconstruct a complete frame without
        unknown passthrough columns. They are accepted only if every accepted
        text fingerprint is unique and the complete fingerprint sequence is
        identical to the accepted sequence. Duplicate texts make a reordered
        legacy frame indistinguishable from the original, so they require the
        private correlation column. Real stages carry that column.
        """
        self._require_active()
        if PRIVATE_CORRELATION_COLUMN in dataframe.columns:
            self._validate_correlations(dataframe, stage="stage_boundary")
            return dataframe
        if (
            COL_TEXT not in dataframe.columns
            or len(dataframe) != len(self._accepted)
            or not self._legacy_identity_is_unambiguous
            or tuple(_stable_digest(value) for value in dataframe[COL_TEXT]) != self._legacy_input_order
        ):
            raise self._error("correlation_missing", "stage_boundary", "invocation")
        bound = dataframe.copy()
        bound[PRIVATE_CORRELATION_COLUMN] = self._accepted
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
            if COL_FINAL_ENTITIES not in dataframe.columns:
                raise self._error("accepted_detection_missing", "result", "invocation")
            for correlation, value in zip(correlations, dataframe[COL_FINAL_ENTITIES], strict=True):
                if self._frozen.get(correlation) != _stable_digest(value):
                    raise self._error("accepted_detection_tampered", "result", "row")
            self._outcomes.update({correlation: _TerminalOutcome.SUCCESS for correlation in correlations})
            self._result_order = tuple(correlations)
            public_result = dataframe.drop(columns=[PRIVATE_CORRELATION_COLUMN], errors="ignore")
            self._result_fingerprints = {
                correlation: _stable_digest(row.to_dict())
                for correlation, (_index, row) in zip(
                    correlations,
                    public_result.iterrows(),
                    strict=True,
                )
            }
            return public_result
        except BaseException:
            # A verifier rejection is an invocation failure for every row that
            # did not already receive a terminal state.  Do this before the
            # verifier is invalidated so the outer sanitizer cannot lose row
            # accounting while translating the error.
            self._complete_remaining(_TerminalOutcome.FAILED)
            raise
        finally:
            self._active = False
            self._frozen.clear()

    def abort(self, *, cancelled: bool) -> None:
        """Close an interrupted invocation with one terminal outcome per accepted row."""
        if not self._active:
            return
        self._complete_remaining(_TerminalOutcome.CANCELLED if cancelled else _TerminalOutcome.FAILED)
        self._active = False
        self._frozen.clear()

    def abort_with_failure(self, *, stage: str, cause: BaseException) -> PrivateRowVerificationError:
        """Close an invocation and return a failure that cannot expose ``cause``.

        The caller raises the returned error after leaving its exception handler.
        This prevents Python from retaining the original exception through the
        otherwise-accessible ``__context__`` attribute.
        """
        del cause
        self.abort(cancelled=False)
        return self._error("invocation_failed", stage, "invocation")

    def take_terminal_outcomes(self) -> tuple[tuple[str, _TerminalOutcome], ...]:
        """Consume the invocation-private correlation accounting exactly once."""
        if self._active:
            raise self._error("invocation_active", "lifecycle", "invocation")
        outcomes = tuple((correlation, self._outcomes[correlation]) for correlation in self._accepted)
        self._accepted = ()
        self._outcomes.clear()
        return outcomes

    def take_result_order(self) -> tuple[str, ...]:
        """Consume verified successful-row correlations in dataframe order."""
        if self._active:
            raise self._error("invocation_active", "lifecycle", "invocation")
        result_order = self._result_order
        self._result_order = ()
        return result_order

    def verify_returned_rows(self, dataframe: pd.DataFrame, result_order: tuple[str, ...]) -> None:
        """Verify token-to-row binding after the backend returns from ``finish``."""
        if self._active:
            raise self._error("invocation_active", "lifecycle", "invocation")
        try:
            if len(result_order) != len(dataframe) or set(result_order) != set(self._result_fingerprints):
                raise self._error("returned_row_mismatch", "return", "invocation")
            for token, (_index, row) in zip(result_order, dataframe.iterrows(), strict=True):
                if self._result_fingerprints.get(token) != _stable_digest(row.to_dict()):
                    raise self._error("returned_row_tampered", "return", "row")
        finally:
            self._result_fingerprints.clear()

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

    def _complete_remaining(self, terminal: _TerminalOutcome) -> None:
        for correlation in self._accepted:
            self._outcomes.setdefault(correlation, terminal)

    def _error(self, code: str, stage: str, scope: str) -> PrivateRowVerificationError:
        return PrivateRowVerificationError(_SafeFailure(code, stage, scope))

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
