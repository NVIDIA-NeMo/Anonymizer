# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for Anonymizer's invocation-private row verification seam."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import COL_TEXT
from anonymizer.engine.private_row_verification import (
    PRIVATE_CORRELATION_COLUMN,
    PrivateRowVerificationError,
    _InvocationRowVerifier,
)
from anonymizer.engine.resolved_input import ResolvedInput
from tests.streaming.structured_trace_prototype import build_synthetic_anonymizer


def test_private_tracking_column_collision_is_rejected_without_exposing_text() -> None:
    secret = "synthetic-secret@example.test"
    frame = pd.DataFrame(
        {
            COL_TEXT: [secret],
            "__anonymizer_private_row_correlation__": ["caller-value"],
        }
    )
    anonymizer = build_synthetic_anonymizer({secret: "email"})
    context = ResolvedInput(frame, requested_text_column="text", resolved_text_column="text")

    with pytest.raises(PrivateRowVerificationError) as exc_info:
        anonymizer._run_internal(
            config=AnonymizerConfig(replace=Redact(), emit_telemetry=False),
            data=AnonymizerInput(source=str(Path(__file__)), text_column="text"),
            context=context,
            preview_num_records=None,
        )

    assert "private_column_collision" in str(exc_info.value)
    assert secret not in str(exc_info.value)


def test_real_local_redact_seam_strips_private_correlation_from_public_result() -> None:
    secret = "synthetic-secret@example.test"
    frame = pd.DataFrame({COL_TEXT: [secret]})
    anonymizer = build_synthetic_anonymizer({secret: "email"})
    context = ResolvedInput(frame, requested_text_column="text", resolved_text_column="text")

    result = anonymizer._run_internal(
        config=AnonymizerConfig(replace=Redact(), emit_telemetry=False),
        data=AnonymizerInput(source=str(Path(__file__)), text_column="text"),
        context=context,
        preview_num_records=None,
    )

    assert PRIVATE_CORRELATION_COLUMN not in result.dataframe
    assert PRIVATE_CORRELATION_COLUMN not in result.trace_dataframe


def _detected_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            COL_TEXT: ["synthetic alpha", "synthetic beta"],
            "final_entities": [{"entities": [{"value": "alpha"}]}, {"entities": []}],
        }
    )


def test_verifier_rejects_reordered_drop_duplicate_and_unknown_rows() -> None:
    base = _detected_frame()
    for transform, expected_code in (
        (lambda frame: frame.iloc[::-1].reset_index(drop=True), None),
        (lambda frame: frame.iloc[:1].copy(), "terminal_row_mismatch"),
        (lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True), "correlation_duplicate"),
        (
            lambda frame: frame.assign(**{PRIVATE_CORRELATION_COLUMN: ["unknown", frame.iloc[1][PRIVATE_CORRELATION_COLUMN]]}),
            "correlation_unknown",
        ),
    ):
        verifier = _InvocationRowVerifier(base)
        bound = verifier.bind(base)
        verifier.freeze_accepted_detections(bound)
        candidate = transform(bound.copy())
        if expected_code is None:
            verified = verifier.finish(candidate)
            assert PRIVATE_CORRELATION_COLUMN not in verified
        else:
            with pytest.raises(PrivateRowVerificationError, match=expected_code):
                verifier.finish(candidate)


def test_verifier_rejects_accepted_detection_tampering_and_closes_without_raw_values() -> None:
    base = _detected_frame()
    verifier = _InvocationRowVerifier(base)
    bound = verifier.bind(base)
    verifier.freeze_accepted_detections(bound)
    tampered = bound.copy()
    tampered.at[0, "final_entities"] = {"entities": [{"value": "different-secret"}]}

    with pytest.raises(PrivateRowVerificationError) as exc_info:
        verifier.finish(tampered)

    assert "accepted_detection_tampered" in str(exc_info.value)
    assert "different-secret" not in str(exc_info.value)
    with pytest.raises(PrivateRowVerificationError, match="invocation_closed"):
        verifier.finish(bound)


def test_verifier_is_nonserializable_and_cancellation_is_terminal() -> None:
    import pickle

    verifier = _InvocationRowVerifier(_detected_frame())
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(verifier)
    verifier.abort(cancelled=True)
    with pytest.raises(PrivateRowVerificationError, match="invocation_closed"):
        verifier.bind(_detected_frame())
