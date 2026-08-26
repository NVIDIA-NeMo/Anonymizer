# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import logging
import threading
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

import anonymizer.interface._protection as protection_module
from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, Rewrite
from anonymizer.config.replace_strategies import Annotate, Redact
from anonymizer.engine.constants import COL_FINAL_ENTITIES, COL_REPLACED_TEXT, COL_TEXT
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.interface._protection import (
    _BatchFailureCode,
    _Failed,
    _NoAcceptedDetections,
    _PlanRejected,
    _PlanUnsupported,
    _ProtectionApplied,
    _ProtectionBatchError,
    _ProtectionRecord,
    _ProtectionSucceeded,
    _RecordRef,
    _Rejected,
    _TextSegment,
)
from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.interface.cli import main as cli_main
from anonymizer.measurement import MeasurementCollector, measurement_session
from tests.streaming.structured_trace_prototype import build_synthetic_anonymizer


def _record(ref: str, text: str) -> _ProtectionRecord:
    return _ProtectionRecord(_RecordRef(ref), (_TextSegment(text),))


def _flow(*, entities: dict[str, str] | None = None):
    anonymizer = build_synthetic_anonymizer(entities or {})
    plan = anonymizer._compile_protection_plan(AnonymizerConfig(replace=Redact(), emit_telemetry=False))
    return anonymizer, anonymizer._open_protection_flow(plan)


def test_private_redact_applies_and_no_detection_success_is_explicit() -> None:
    _, flow = _flow(entities={"alice@example.test": "email"})
    result = flow.protect((_record("a", "mail alice@example.test"), _record("b", "ordinary text")))

    applied, unchanged = result.outcomes
    assert isinstance(applied, _ProtectionSucceeded)
    assert isinstance(applied.disposition, _ProtectionApplied)
    assert applied.output == "mail [REDACTED_EMAIL]"
    assert isinstance(unchanged, _ProtectionSucceeded)
    assert isinstance(unchanged.disposition, _NoAcceptedDetections)
    assert unchanged.output == "ordinary text"
    assert result.success_count == 2
    assert result.failure_count == 0
    assert not hasattr(result, "trace_dataframe")
    assert "row_token" not in repr(result).lower()


def test_phase5_adds_no_public_context_or_graph_parameters() -> None:
    for entrypoint in (Anonymizer.run, Anonymizer.preview, Anonymizer.evaluate, cli_main.run, cli_main.preview):
        public_parameters = inspect.signature(entrypoint).parameters
        assert "context" not in public_parameters
        assert "graph" not in public_parameters


def test_context_workframe_observations_are_paired_bounded_and_content_free() -> None:
    target_canary = "TARGET-CANARY-alice@example.test"
    source_canary = "SOURCE-CANARY-7843"
    _, flow = _flow(entities={target_canary: "email"})
    collector = MeasurementCollector(run_id="phase5-observation-test")

    with measurement_session(collector):
        result = flow.protect((_record(source_canary, target_canary),))

    assert isinstance(result.outcomes[0], _ProtectionSucceeded)
    records = [record for record in collector.records if record["record_type"] == "context_workframe"]
    boundaries = {record["boundary"] for record in records}
    assert {
        "preflight",
        "capability_recheck",
        "workframe_construction",
        "dispatch",
        "backend_execution",
        "reconciliation",
        "cleanup",
        "release",
    }.issubset(boundaries)
    for boundary in boundaries:
        paired = [record for record in records if record["boundary"] == boundary]
        assert [record["event"] for record in paired] == ["start", "terminal"]
        assert paired[-1]["duration_sec"] >= 0
        assert isinstance(paired[-1]["target_count_bucket"], str)
        assert isinstance(paired[-1]["context_count_bucket"], str)
    rendered = repr(records)
    assert target_canary not in rendered
    assert source_canary not in rendered
    assert "__anonymizer_private_row_correlation__" not in rendered


def test_throwing_context_observer_cannot_change_release() -> None:
    class ThrowingCollector(MeasurementCollector):
        def record(self, record_type: str, **fields: Any) -> None:
            if record_type == "context_workframe":
                raise RuntimeError("observer unavailable")
            super().record(record_type, **fields)

    _, flow = _flow()
    with measurement_session(ThrowingCollector()):
        result = flow.protect((_record("a", "ordinary text"),))

    assert isinstance(result.outcomes[0], _ProtectionSucceeded)


def test_base_exception_from_context_observer_cannot_change_release() -> None:
    class InterruptingCollector(MeasurementCollector):
        def record(self, record_type: str, **fields: Any) -> None:
            if record_type == "context_workframe":
                raise KeyboardInterrupt
            super().record(record_type, **fields)

    _, flow = _flow()
    with measurement_session(InterruptingCollector()):
        result = flow.protect((_record("a", "ordinary text"),))

    assert isinstance(result.outcomes[0], _ProtectionSucceeded)


def test_reentrant_context_observer_is_bounded_and_cannot_change_release() -> None:
    _, flow = _flow()

    class ReentrantCollector(MeasurementCollector):
        entered = False

        def record(self, record_type: str, **fields: Any) -> None:
            if record_type == "context_workframe" and not self.entered:
                self.entered = True
                nested = flow.protect((_record("nested", "ordinary text"),))
                assert isinstance(nested.outcomes[0], _ProtectionSucceeded)
            super().record(record_type, **fields)

    collector = ReentrantCollector()
    with measurement_session(collector):
        result = flow.protect((_record("outer", "ordinary text"),))

    assert isinstance(result.outcomes[0], _ProtectionSucceeded)
    context_records = [record for record in collector.records if record["record_type"] == "context_workframe"]
    assert 0 < len(context_records) <= 16
    assert all(
        sum(record["boundary"] == boundary and record["event"] == event for record in context_records) <= 1
        for boundary in {record["boundary"] for record in context_records}
        for event in ("start", "terminal")
    )


def test_trivial_graph_path_matches_public_run_output(tmp_path: Path) -> None:
    secret = "alice@example.test"
    anonymizer = build_synthetic_anonymizer({secret: "email"})
    config = AnonymizerConfig(replace=Redact(), emit_telemetry=False)
    source = tmp_path / "parity.csv"
    pd.DataFrame({"text": [f"mail {secret}", "ordinary text"]}).to_csv(source, index=False)

    public = anonymizer.run(config=config, data=AnonymizerInput(source=str(source), text_column="text"))
    plan = anonymizer._compile_protection_plan(config)
    private = anonymizer._open_protection_flow(plan).protect(
        (_record("a", f"mail {secret}"), _record("b", "ordinary text"))
    )

    assert [cast(_ProtectionSucceeded, outcome).output for outcome in private.outcomes] == public.dataframe[
        "text_replaced"
    ].tolist()


def test_graph_outcomes_are_rejoined_by_private_datum_identity() -> None:
    secret = "alice@example.test"
    _, flow = _flow(entities={secret: "email"})
    original = flow._runtime.protect

    def reordered(*args: Any, **kwargs: Any):
        result = original(*args, **kwargs)
        return replace(result, outcomes=tuple(reversed(result.outcomes)))

    flow._runtime.protect = reordered
    result = flow.protect((_record("private-a", secret), _record("private-b", "plain")))

    first, second = result.outcomes
    assert isinstance(first, _ProtectionSucceeded)
    assert first.ref.value == "private-a"
    assert first.output == "[REDACTED_EMAIL]"
    assert isinstance(second, _ProtectionSucceeded)
    assert second.ref.value == "private-b"
    assert second.output == "plain"


def test_compilation_is_closed_and_plan_is_content_free_and_immutable() -> None:
    anonymizer = build_synthetic_anonymizer({})
    assert isinstance(anonymizer._compile_protection_plan(AnonymizerConfig(replace=Annotate())), _PlanRejected)
    assert isinstance(anonymizer._compile_protection_plan(AnonymizerConfig(rewrite=Rewrite())), _PlanUnsupported)
    plan = anonymizer._compile_protection_plan(AnonymizerConfig(replace=Redact(), emit_telemetry=False))
    assert "Redact" not in repr(plan)
    with pytest.raises(FrozenInstanceError):
        setattr(plan, "profile", "changed")


def test_plan_snapshot_detects_nested_tampering_and_digest_covers_models() -> None:
    anonymizer = build_synthetic_anonymizer({})
    config = AnonymizerConfig(replace=Redact(), emit_telemetry=False)
    first = anonymizer._compile_protection_plan(config)
    cast(Redact, config.replace).format_template = "<{label}>"
    assert first.invocation.replace_method.format_template == "[REDACTED_{label}]"

    anonymizer._selected_models.detection.entity_detector = "materially-different"
    second = anonymizer._compile_protection_plan(AnonymizerConfig(replace=Redact(), emit_telemetry=False))
    assert first.digest != second.digest
    second.invocation.selected_models.detection.entity_detector = "tampered-after-compile"
    outcome = anonymizer._open_protection_flow(second).protect((_record("a", "text"),)).outcomes[0]
    assert isinstance(outcome, _Failed)


def test_plan_digest_separates_model_config_and_replacement_semantics() -> None:
    anonymizer = build_synthetic_anonymizer({})
    baseline = anonymizer._compile_protection_plan(AnonymizerConfig(replace=Redact(), emit_telemetry=False))
    anonymizer._model_configs[0].model = "materially-different-model"
    different_model = anonymizer._compile_protection_plan(AnonymizerConfig(replace=Redact(), emit_telemetry=False))
    assert baseline.digest != different_model.digest

    rejected_profile = anonymizer._compile_protection_plan(
        AnonymizerConfig(replace=Redact(format_template="<{label}>", normalize_label=True), emit_telemetry=False)
    )
    assert isinstance(rejected_profile, _PlanRejected)


def test_receipt_binds_plan_digest_and_fresh_attempt_identity() -> None:
    _, flow = _flow()
    first = cast(_ProtectionSucceeded, flow.protect((_record("a", "text"),)).outcomes[0]).receipt
    second = cast(_ProtectionSucceeded, flow.protect((_record("a", "text"),)).outcomes[0]).receipt
    assert first.plan_digest == second.plan_digest
    assert first.attempt_id != second.attempt_id


@pytest.mark.parametrize(
    "records, code",
    [
        ((_record("same", "a"), _record("same", "b")), _BatchFailureCode.DUPLICATE_REF),
        ((_record("a", "x" * 32_769),), _BatchFailureCode.RECORD_TOO_LARGE),
        (tuple(_record(str(index), "x" * 32_768) for index in range(33)), _BatchFailureCode.BATCH_TOO_LARGE),
        (tuple(_record(str(index), "x") for index in range(129)), _BatchFailureCode.TOO_MANY_RECORDS),
        ((_ProtectionRecord(_RecordRef("a"), ()),), _BatchFailureCode.UNSUPPORTED_CARDINALITY),
        (
            (_ProtectionRecord(cast(Any, "secret@example.test"), (_TextSegment("text"),)),),
            _BatchFailureCode.MALFORMED_BATCH,
        ),
    ],
)
def test_outer_batch_is_rejected_before_admission(records: object, code: _BatchFailureCode) -> None:
    _, flow = _flow()
    with pytest.raises(_ProtectionBatchError) as exc_info:
        flow.protect(cast(Any, records))
    assert exc_info.value.code is code
    assert repr(exc_info.value) == "<private protection batch error>"


def test_graph_admission_rejects_before_private_execution_context(monkeypatch: pytest.MonkeyPatch) -> None:
    anonymizer, flow = _flow()
    entered = False
    collector = MeasurementCollector()

    @contextmanager
    def effect_spy():
        nonlocal entered
        entered = True
        yield

    monkeypatch.setattr(protection_module, "_trivial_graph", lambda _datums: object())
    monkeypatch.setattr(anonymizer._adapter, "private_execution", effect_spy)

    with measurement_session(collector):
        result = flow.protect((_record("a", "text"),))

    assert isinstance(result.outcomes[0], _Failed)
    assert not entered
    observations = [record for record in collector.records if record["record_type"] == "context_workframe"]
    assert [(record["boundary"], record["event"]) for record in observations] == [
        ("preflight", "start"),
        ("preflight", "terminal"),
    ]


def test_malformed_nested_segment_is_rejected_before_admission() -> None:
    forged = object.__new__(_ProtectionRecord)
    object.__setattr__(forged, "ref", _RecordRef("a"))
    object.__setattr__(forged, "segments", (object(),))

    _, flow = _flow()
    with pytest.raises(_ProtectionBatchError) as exc_info:
        flow.protect((forged,))
    assert exc_info.value.code is _BatchFailureCode.MALFORMED_BATCH
    assert str(exc_info.value) == "private protection batch rejected"
    assert repr(exc_info.value) == "<private protection batch error>"


def test_missing_nested_record_attributes_are_sanitized_batch_rejections() -> None:
    _, flow = _flow()
    missing_record = object.__new__(_ProtectionRecord)
    missing_ref = object.__new__(_RecordRef)
    missing_segment = object.__new__(_TextSegment)
    forged_values = (
        missing_record,
        _ProtectionRecord(missing_ref, (_TextSegment("text"),)),
        _ProtectionRecord(_RecordRef("a"), (missing_segment,)),
    )
    for forged in forged_values:
        with pytest.raises(_ProtectionBatchError, match="private protection batch rejected"):
            flow.protect((forged,))


def test_invalid_ref_is_bounded_and_content_free() -> None:
    secret = "secret@example.test"
    with pytest.raises(ValueError) as exc_info:
        _RecordRef(secret * 30)
    assert secret not in str(exc_info.value)


def test_cancel_before_admission_and_overlap_are_rejections() -> None:
    _, flow = _flow()
    cancelled = flow.protect((_record("a", "text"),), cancelled_before_admission=True)
    assert isinstance(cancelled.outcomes[0], _Rejected)
    assert cancelled.outcomes[0].failure.code.value == "cancelled_before_admission"

    entered = threading.Event()
    release = threading.Event()
    original = flow._runtime.protect

    def blocked(*args: Any, **kwargs: Any):
        entered.set()
        release.wait(timeout=5)
        return original(*args, **kwargs)

    flow._runtime.protect = blocked
    worker = threading.Thread(target=lambda: flow.protect((_record("one", "text"),)))
    worker.start()
    assert entered.wait(timeout=5)
    busy = flow.protect((_record("two", "text"),))
    release.set()
    worker.join(timeout=5)
    assert isinstance(busy.outcomes[0], _Rejected)
    assert busy.outcomes[0].failure.code.value == "busy"


@pytest.mark.parametrize("mode", ["reorder", "drop", "duplicate", "unknown", "tamper"])
def test_adversarial_engine_results_have_exact_safe_terminal_accounting(mode: str) -> None:
    secret = "alice@example.test"
    anonymizer, flow = _flow(entities={secret: "email"})
    original = cast(Any, anonymizer._replace_runner).run

    def altered(*args: Any, **kwargs: Any):
        result = original(*args, **kwargs)
        frame = result.dataframe
        if mode == "reorder":
            frame = frame.iloc[::-1].reset_index(drop=True)
        elif mode == "drop":
            frame = frame.iloc[:1].copy()
        elif mode == "duplicate":
            frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        elif mode == "unknown":
            frame = frame.copy()
            frame.iloc[0, frame.columns.get_loc("__anonymizer_private_row_correlation__")] = "unknown"
        else:
            frame = frame.copy()
            frame.at[0, COL_FINAL_ENTITIES] = {"entities": []}
        return type(result)(dataframe=frame, failed_records=result.failed_records)

    cast(Any, anonymizer._replace_runner).run = altered
    run = flow.protect((_record("a", secret), _record("b", "plain")))
    assert len(run.outcomes) == 2
    assert [outcome.ref.value for outcome in run.outcomes] == ["a", "b"]
    if mode == "reorder":
        assert all(isinstance(outcome, _ProtectionSucceeded) for outcome in run.outcomes)
        first = cast(_ProtectionSucceeded, run.outcomes[0])
        second = cast(_ProtectionSucceeded, run.outcomes[1])
        assert first.output == "[REDACTED_EMAIL]"
        assert second.output == "plain"
    elif mode == "drop":
        assert sum(isinstance(outcome, _Failed) for outcome in run.outcomes) == 1
    else:
        assert all(isinstance(outcome, _Failed) for outcome in run.outcomes)
    assert secret not in repr(run)
    assert "__anonymizer_private_row_correlation__" not in repr(run)


def test_invocation_failure_suppresses_cause_and_emits_no_output() -> None:
    secret = "provider secret alice@example.test"
    anonymizer, flow = _flow()
    cast(Any, anonymizer._detection_workflow).run = lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError(secret))
    run = flow.protect((_record("private-ref", "raw input"),))
    outcome = run.outcomes[0]
    assert isinstance(outcome, _Failed)
    assert not hasattr(outcome, "output")
    assert secret not in repr(outcome)
    assert "private-ref" not in repr(outcome)


def test_private_failure_logs_exclude_backend_exception_and_failed_record_canaries(
    caplog: pytest.LogCaptureFixture,
) -> None:
    backend_canary = "BACKEND-SECRET-alice@example.test"
    reason_canary = "FAILED-REASON-bob@example.test"
    anonymizer, flow = _flow()
    original = cast(Any, anonymizer._replace_runner).run

    def failed_record(*args: Any, **kwargs: Any):
        result = original(*args, **kwargs)
        return type(result)(
            dataframe=result.dataframe,
            failed_records=[FailedRecord(record_id="PRIVATE-ID", step="replace", reason=reason_canary)],
        )

    cast(Any, anonymizer._replace_runner).run = failed_record
    caplog.set_level(logging.DEBUG)
    run = flow.protect((_record("a", "text"),))
    assert isinstance(run.outcomes[0], _Failed)
    rendered = repr(run) + "\n" + "\n".join(record.getMessage() for record in caplog.records)
    assert reason_canary not in rendered
    assert "PRIVATE-ID" not in rendered

    cast(Any, anonymizer._detection_workflow).run = lambda *_a, **_k: (_ for _ in ()).throw(
        RuntimeError(backend_canary)
    )
    flow.protect((_record("b", "raw"),))
    rendered = "\n".join(record.getMessage() for record in caplog.records)
    assert backend_canary not in rendered


def test_malformed_accepted_entities_fail_closed() -> None:
    anonymizer, flow = _flow()
    original = cast(Any, anonymizer._replace_runner).run

    def malformed(*args: Any, **kwargs: Any):
        result = original(*args, **kwargs)
        frame = result.dataframe.copy()
        frame.at[0, COL_FINAL_ENTITIES] = {"entities": object()}
        return type(result)(dataframe=frame, failed_records=result.failed_records)

    cast(Any, anonymizer._replace_runner).run = malformed
    assert isinstance(flow.protect((_record("a", "text"),)).outcomes[0], _Failed)


def test_failure_retry_taxonomy_is_unassigned_and_unknown() -> None:
    _, flow = _flow()
    flow.close()
    failure = cast(_Rejected, flow.protect((_record("a", "text"),)).outcomes[0]).failure
    assert failure.retry_safety.value == "unknown"
    assert failure.retry_owner.value == "unassigned"


def test_release_predicate_prevents_raw_fallback() -> None:
    secret = "alice@example.test"
    anonymizer, flow = _flow(entities={secret: "email"})
    original = cast(Any, anonymizer._replace_runner).run

    def raw_fallback(*args: Any, **kwargs: Any):
        result = original(*args, **kwargs)
        frame = result.dataframe.copy()
        frame[COL_REPLACED_TEXT] = frame[COL_TEXT]
        return type(result)(dataframe=frame, failed_records=result.failed_records)

    cast(Any, anonymizer._replace_runner).run = raw_fallback
    outcome = flow.protect((_record("a", secret),)).outcomes[0]
    assert isinstance(outcome, _Failed)
    assert not hasattr(outcome, "output")


def test_no_accepted_detections_requires_output_unchanged_from_input() -> None:
    anonymizer, flow = _flow()
    original = cast(Any, anonymizer._replace_runner).run

    def modified_without_detections(*args: Any, **kwargs: Any):
        result = original(*args, **kwargs)
        frame = result.dataframe.copy()
        frame[COL_REPLACED_TEXT] = "modified"
        return type(result)(dataframe=frame, failed_records=result.failed_records)

    cast(Any, anonymizer._replace_runner).run = modified_without_detections
    outcome = flow.protect((_record("a", "original"),)).outcomes[0]
    assert isinstance(outcome, _Failed)
    assert not hasattr(outcome, "output")


def test_close_is_idempotent_and_does_not_close_borrowed_anonymizer() -> None:
    anonymizer, flow = _flow()
    flow.close()
    flow.close()
    rejected = flow.protect((_record("a", "text"),))
    assert isinstance(rejected.outcomes[0], _Rejected)
    assert anonymizer.run is not None
