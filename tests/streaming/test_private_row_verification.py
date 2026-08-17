# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for Anonymizer's invocation-private row verification seam."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import COL_FINAL_ENTITIES, COL_TEXT
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult
from anonymizer.engine.private_row_verification import (
    PRIVATE_CORRELATION_COLUMN,
    PrivateRowVerificationError,
    _InvocationRowVerifier,
)
from anonymizer.engine.replace.replace_runner import ReplacementResult
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


def test_real_engine_seam_sanitizes_pipeline_exception() -> None:
    secret = "synthetic-secret@example.test"
    frame = pd.DataFrame({COL_TEXT: [secret]})
    anonymizer = build_synthetic_anonymizer({secret: "email"})
    cast(Any, anonymizer._detection_workflow).run = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError(secret)
    )
    context = ResolvedInput(frame, requested_text_column="text", resolved_text_column="text")

    with pytest.raises(PrivateRowVerificationError) as exc_info:
        anonymizer._run_internal(
            config=AnonymizerConfig(replace=Redact(), emit_telemetry=False),
            data=AnonymizerInput(source=str(Path(__file__)), text_column="text"),
            context=context,
            preview_num_records=None,
        )

    assert "invocation_failed" in str(exc_info.value)
    assert secret not in str(exc_info.value)


@pytest.mark.parametrize(
    ("transform", "raises", "expected_rows"),
    [
        (lambda frame: frame.iloc[::-1].reset_index(drop=True), False, 2),
        (lambda frame: frame.iloc[:1].copy(), False, 1),
        (lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True), True, None),
        (
            lambda frame: frame.assign(
                **{PRIVATE_CORRELATION_COLUMN: ["unknown", frame.iloc[1][PRIVATE_CORRELATION_COLUMN]]}
            ),
            True,
            None,
        ),
    ],
    ids=["reorder", "drop", "duplicate", "unknown"],
)
def test_real_local_engine_seam_accounts_for_correlation_transformations(
    transform: Any, raises: bool, expected_rows: int | None
) -> None:
    secret = "synthetic-secret@example.test"
    frame = pd.DataFrame({COL_TEXT: [secret, "synthetic second"]})
    anonymizer = build_synthetic_anonymizer({secret: "email"})
    original_run = cast(Any, anonymizer._detection_workflow.run)

    def transformed_run(*args: Any, **kwargs: Any) -> EntityDetectionResult:
        result = original_run(*args, **kwargs)
        return EntityDetectionResult(dataframe=transform(result.dataframe), failed_records=result.failed_records)

    cast(Any, anonymizer._detection_workflow).run = transformed_run
    context = ResolvedInput(frame, requested_text_column="text", resolved_text_column="text")
    call = lambda: anonymizer._run_internal(
        config=AnonymizerConfig(replace=Redact(), emit_telemetry=False),
        data=AnonymizerInput(source=str(Path(__file__)), text_column="text"),
        context=context,
        preview_num_records=None,
    )

    if raises:
        with pytest.raises(PrivateRowVerificationError) as exc_info:
            call()
        assert secret not in str(exc_info.value)
    else:
        assert len(call().dataframe) == expected_rows


def test_real_engine_seam_rejects_duplicate_reordered_legacy_detection_output() -> None:
    secret = "synthetic-secret@example.test"
    frame = pd.DataFrame({COL_TEXT: [secret, secret]})
    anonymizer = build_synthetic_anonymizer({secret: "email"})
    original_run = cast(Any, anonymizer._detection_workflow.run)

    def reordered_legacy_run(*args: Any, **kwargs: Any) -> EntityDetectionResult:
        result = original_run(*args, **kwargs)
        legacy = result.dataframe.iloc[::-1].reset_index(drop=True)
        return EntityDetectionResult(
            dataframe=legacy.drop(columns=[PRIVATE_CORRELATION_COLUMN]),
            failed_records=result.failed_records,
        )

    cast(Any, anonymizer._detection_workflow).run = reordered_legacy_run
    context = ResolvedInput(frame, requested_text_column="text", resolved_text_column="text")

    with pytest.raises(PrivateRowVerificationError) as exc_info:
        anonymizer._run_internal(
            config=AnonymizerConfig(replace=Redact(), emit_telemetry=False),
            data=AnonymizerInput(source=str(Path(__file__)), text_column="text"),
            context=context,
            preview_num_records=None,
        )

    assert "invocation_failed" in str(exc_info.value)
    assert secret not in str(exc_info.value)


def test_real_engine_seam_rejects_final_accepted_detection_removal() -> None:
    secret = "synthetic-secret@example.test"
    frame = pd.DataFrame({COL_TEXT: [secret]})
    anonymizer = build_synthetic_anonymizer({secret: "email"})
    original_run = cast(Any, anonymizer._replace_runner.run)

    def remove_accepted_detection(*args: Any, **kwargs: Any) -> ReplacementResult:
        result = original_run(*args, **kwargs)
        return ReplacementResult(
            dataframe=result.dataframe.drop(columns=[COL_FINAL_ENTITIES]),
            failed_records=result.failed_records,
        )

    cast(Any, anonymizer._replace_runner).run = remove_accepted_detection
    context = ResolvedInput(frame, requested_text_column="text", resolved_text_column="text")

    with pytest.raises(PrivateRowVerificationError) as exc_info:
        anonymizer._run_internal(
            config=AnonymizerConfig(replace=Redact(), emit_telemetry=False),
            data=AnonymizerInput(source=str(Path(__file__)), text_column="text"),
            context=context,
            preview_num_records=None,
        )

    assert "invocation_failed" in str(exc_info.value)
    assert secret not in str(exc_info.value)


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
        (lambda frame: frame.iloc[:1].copy(), None),
        (lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True), "correlation_duplicate"),
        (
            lambda frame: frame.assign(
                **{PRIVATE_CORRELATION_COLUMN: ["unknown", frame.iloc[1][PRIVATE_CORRELATION_COLUMN]]}
            ),
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
            if len(candidate) == 1:
                assert len(verified) == 1
                outcomes = [outcome.value for outcome in verifier._outcomes.values()]
                assert outcomes.count("failed") == 1
                assert outcomes.count("success") == 1
        else:
            with pytest.raises(PrivateRowVerificationError, match=expected_code):
                verifier.finish(candidate)


def test_verifier_rejects_a_missing_correlation_instead_of_recovering_from_text() -> None:
    base = _detected_frame()
    verifier = _InvocationRowVerifier(base)
    bound = verifier.bind(base)
    verifier.freeze_accepted_detections(bound)

    with pytest.raises(PrivateRowVerificationError, match="correlation_missing"):
        verifier.finish(bound.drop(columns=[PRIVATE_CORRELATION_COLUMN]))

    assert {outcome.value for outcome in verifier._outcomes.values()} == {"failed"}


def test_stage_output_never_receives_positional_correlation_rebinding() -> None:
    base = _detected_frame()
    verifier = _InvocationRowVerifier(base)
    bound = verifier.bind(base)

    with pytest.raises(PrivateRowVerificationError, match="correlation_missing"):
        verifier.bind_complete_stage_output(bound.iloc[::-1].drop(columns=[PRIVATE_CORRELATION_COLUMN]))


def test_row_failure_precedes_invocation_cancellation() -> None:
    base = _detected_frame()
    verifier = _InvocationRowVerifier(base)
    bound = verifier.bind(base)
    verifier.freeze_accepted_detections(bound.iloc[:1].copy())
    verifier.abort(cancelled=True)

    outcomes = [outcome.value for outcome in verifier._outcomes.values()]
    assert outcomes.count("failed") == 1
    assert outcomes.count("cancelled") == 1


def test_abort_sanitizes_underlying_failure_text() -> None:
    verifier = _InvocationRowVerifier(_detected_frame())
    secret = "provider replied with synthetic-secret and engine-id-8675309"

    with pytest.raises(PrivateRowVerificationError) as exc_info:
        verifier.abort_with_failure(stage="replace", cause=RuntimeError(secret))

    assert "invocation_failed" in str(exc_info.value)
    assert secret not in str(exc_info.value)


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


def test_verifier_requires_frozen_accepted_detection_evidence_at_finish() -> None:
    base = _detected_frame()
    verifier = _InvocationRowVerifier(base)
    bound = verifier.bind(base)
    verifier.freeze_accepted_detections(bound)

    with pytest.raises(PrivateRowVerificationError, match="accepted_detection_missing"):
        verifier.finish(bound.drop(columns=[COL_FINAL_ENTITIES]))


def test_verifier_is_nonserializable_and_cancellation_is_terminal() -> None:
    import pickle

    verifier = _InvocationRowVerifier(_detected_frame())
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(verifier)
    verifier.abort(cancelled=True)
    with pytest.raises(PrivateRowVerificationError, match="invocation_closed"):
        verifier.bind(_detected_frame())


def test_characterization_reports_governed_unavailability_without_zero_measurements() -> None:
    completed = subprocess.run(
        [sys.executable, "tests/streaming/run_internal_characterization.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    arms = {arm["arm"]: arm for arm in report["arms"]}
    blocked = arms["generic_manifest"]

    assert set(blocked) == set(arms["field_per_row"])
    assert blocked["status"] == "blocked"
    assert blocked["availability"] == "governed_unavailable"
    assert blocked["reason_code"] == "source_specific_manifest_not_owned"
    for metric in (
        "input_bytes",
        "output_bytes",
        "rows",
        "targets",
        "provider_calls",
        "elapsed_ms",
        "peak_memory_bytes",
        "raw_copy_count",
        "artifact_delta_bytes",
        "structural_validity",
        "privacy_check",
        "reconstruction_failures",
    ):
        assert blocked[metric] is None
