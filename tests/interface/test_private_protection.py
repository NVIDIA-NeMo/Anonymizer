# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import threading
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

import anonymizer.interface._protection as protection_module
from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, Rewrite
from anonymizer.config.replace_strategies import Annotate, Redact
from anonymizer.engine.execution.context_contract import (
    _BackendArtifactClass,
    _ContextBackendCapability,
    _ContextLimits,
    _ContextOrdering,
    _ContextProfile,
    _ContextSchemaVersion,
    _RetentionPosture,
)
from anonymizer.engine.execution.mention_admission import _ValidationDecision, _ValidationDecisionKind
from anonymizer.engine.execution.mention_resolution import _SubjectEvidence
from anonymizer.engine.execution.phase6_runtime import (
    _CandidateProposal,
    _Phase6AugmentationWork,
    _Phase6CandidateWork,
    _Phase6ResolverWork,
    _Phase6ValidationWork,
)
from anonymizer.engine.execution.protection_service import _Phase6RedactProtectionService
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
    entity_map = entities or {}
    anonymizer = build_synthetic_anonymizer(entity_map)
    plan = anonymizer._compile_protection_plan(AnonymizerConfig(replace=Redact(), emit_telemetry=False))
    return anonymizer, protection_module._ProtectionFlow(
        anonymizer,
        plan,
        phase6_backend=_AnchoredPhase6Backend.from_entities(entity_map),
    )


class _AnchoredPhase6Backend:
    def __init__(
        self,
        proposals: dict[str, tuple[_CandidateProposal, ...]],
        *,
        entities: dict[str, str] | None = None,
        fault_stage: str | None = None,
        malformed_stage: str | None = None,
    ) -> None:
        self._proposals = proposals
        self._entities = entities or {}
        self._fault_stage = fault_stage
        self._malformed_stage = malformed_stage
        self.closed = False
        self.calls: list[str] = []

    @classmethod
    def from_entities(cls, entities: dict[str, str]) -> _AnchoredPhase6Backend:
        return cls({}, entities=entities)

    def context_capability(self) -> _ContextBackendCapability:
        return _ContextBackendCapability(
            _ContextProfile.TARGET_CONTEXT_V1,
            _ContextSchemaVersion.V1,
            _ContextLimits(128, 1_048_576, 16_384, 2_097_152),
            True,
            _ContextOrdering.DECLARED,
            (_BackendArtifactClass.CONTEXT_REQUEST,),
            _RetentionPosture.DISABLED,
        )

    def detect(self, work: _Phase6CandidateWork) -> tuple[_CandidateProposal, ...]:
        self._enter("detect")
        if self._malformed_stage == "detect":
            return cast(Any, (object(),))
        explicit = self._proposals.get(work.target.text)
        if explicit is not None:
            return explicit
        proposals: list[_CandidateProposal] = []
        for value, label in self._entities.items():
            start = 0
            while (found := work.target.text.find(value, start)) >= 0:
                proposals.append(_CandidateProposal(found, found + len(value), value, label))
                start = found + len(value)
        return tuple(sorted(proposals, key=lambda proposal: (proposal.start, proposal.end)))

    def augment(self, work: _Phase6AugmentationWork) -> tuple[_CandidateProposal, ...]:
        self._enter("augment")
        return ()

    def validate(self, work: _Phase6ValidationWork) -> tuple[_ValidationDecision, ...]:
        self._enter("validate")
        if self._malformed_stage == "validate":
            return cast(Any, ())
        return tuple(
            _ValidationDecision(candidate.token, _ValidationDecisionKind.KEEP) for candidate in work.candidates
        )

    def resolve(self, work: _Phase6ResolverWork) -> tuple[_SubjectEvidence, ...]:
        self._enter("resolve")
        if self._malformed_stage == "resolve":
            return cast(Any, (object(),))
        return ()

    def close_phase6(self) -> bool:
        self._enter("close")
        self.closed = True
        return self._malformed_stage != "close"

    def _enter(self, stage: str) -> None:
        self.calls.append(stage)
        if self._fault_stage == stage:
            raise RuntimeError("BACKEND-SECRET-alice@example.test")


def test_private_redact_applies_and_no_detection_success_is_explicit() -> None:
    _, flow = _flow(entities={"alice@example.test": "email"})
    result = flow.protect((_record("a", "mail alice@example.test"), _record("b", "ordinary text")))

    applied, unchanged = result.outcomes
    assert isinstance(applied, _ProtectionSucceeded)
    assert isinstance(applied.disposition, _ProtectionApplied)
    assert applied.output == "mail [REDACTED]"
    assert isinstance(unchanged, _ProtectionSucceeded)
    assert isinstance(unchanged.disposition, _NoAcceptedDetections)
    assert unchanged.output == "ordinary text"
    assert result.success_count == 2
    assert result.failure_count == 0
    assert not hasattr(result, "trace_dataframe")
    assert "row_token" not in repr(result).lower()


def test_private_flow_has_an_engine_private_phase6_backend_seam() -> None:
    parameters = inspect.signature(protection_module._ProtectionFlow).parameters

    assert "phase6_backend" in parameters


def test_default_private_flow_selects_phase6_redact_service() -> None:
    anonymizer = build_synthetic_anonymizer({})
    plan = anonymizer._compile_protection_plan(AnonymizerConfig(replace=Redact(), emit_telemetry=False))

    flow = anonymizer._open_protection_flow(plan)

    assert isinstance(flow._runtime, _Phase6RedactProtectionService)


def test_private_flow_executes_phase6_anchor_reconstruction() -> None:
    text = "Alice and Alice"
    anonymizer = build_synthetic_anonymizer({"Alice": "name"})
    plan = anonymizer._compile_protection_plan(AnonymizerConfig(replace=Redact(), emit_telemetry=False))
    backend = _AnchoredPhase6Backend({text: (_CandidateProposal(0, 5, "Alice", "name"),)})
    flow = protection_module._ProtectionFlow(anonymizer, plan, phase6_backend=backend)

    result = flow.protect((_record("a", text),))

    outcome = result.outcomes[0]
    assert isinstance(outcome, _ProtectionSucceeded)
    assert isinstance(outcome.disposition, _ProtectionApplied)
    assert outcome.output == "[REDACTED] and Alice"
    assert backend.closed
    assert "Alice" not in repr(result)


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


def test_private_phase6_and_public_dataframe_redact_keep_their_distinct_compatibility_outputs(tmp_path: Path) -> None:
    secret = "alice@example.test"
    anonymizer = build_synthetic_anonymizer({secret: "email"})
    config = AnonymizerConfig(replace=Redact(), emit_telemetry=False)
    source = tmp_path / "parity.csv"
    pd.DataFrame({"text": [f"mail {secret}", "ordinary text"]}).to_csv(source, index=False)

    public = anonymizer.run(config=config, data=AnonymizerInput(source=str(source), text_column="text"))
    plan = anonymizer._compile_protection_plan(config)
    private = protection_module._ProtectionFlow(
        anonymizer,
        plan,
        phase6_backend=_AnchoredPhase6Backend.from_entities({secret: "email"}),
    ).protect((_record("a", f"mail {secret}"), _record("b", "ordinary text")))

    assert public.dataframe["text_replaced"].tolist() == ["mail [REDACTED_EMAIL]", "ordinary text"]
    assert [cast(_ProtectionSucceeded, outcome).output for outcome in private.outcomes] == [
        "mail [REDACTED]",
        "ordinary text",
    ]


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
    assert first.output == "[REDACTED]"
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


def test_graph_admission_rejects_before_phase6_backend_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    anonymizer = build_synthetic_anonymizer({})
    plan = anonymizer._compile_protection_plan(AnonymizerConfig(replace=Redact(), emit_telemetry=False))
    backend = _AnchoredPhase6Backend.from_entities({})
    flow = protection_module._ProtectionFlow(anonymizer, plan, phase6_backend=backend)
    collector = MeasurementCollector()

    monkeypatch.setattr(protection_module, "_trivial_graph", lambda _datums: object())

    with measurement_session(collector):
        result = flow.protect((_record("a", "text"),))

    assert isinstance(result.outcomes[0], _Failed)
    assert backend.calls == []
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


@pytest.mark.parametrize("stage", ["detect", "augment", "validate", "resolve"])
def test_phase6_backend_faults_have_exact_safe_terminal_accounting(stage: str) -> None:
    secret = "alice@example.test"
    anonymizer = build_synthetic_anonymizer({secret: "email"})
    plan = anonymizer._compile_protection_plan(AnonymizerConfig(replace=Redact(), emit_telemetry=False))
    backend = _AnchoredPhase6Backend.from_entities({secret: "email"})
    backend._fault_stage = stage
    flow = protection_module._ProtectionFlow(anonymizer, plan, phase6_backend=backend)
    run = flow.protect((_record("a", secret), _record("b", "plain")))

    assert len(run.outcomes) == 2
    assert [outcome.ref.value for outcome in run.outcomes] == ["a", "b"]
    assert all(isinstance(outcome, _Failed) for outcome in run.outcomes)
    assert secret not in repr(run)


def test_invocation_failure_suppresses_cause_and_emits_no_output() -> None:
    secret = "provider secret alice@example.test"
    anonymizer = build_synthetic_anonymizer({})
    plan = anonymizer._compile_protection_plan(AnonymizerConfig(replace=Redact(), emit_telemetry=False))
    backend = _AnchoredPhase6Backend({}, fault_stage="detect")
    flow = protection_module._ProtectionFlow(anonymizer, plan, phase6_backend=backend)
    run = flow.protect((_record("private-ref", "raw input"),))
    outcome = run.outcomes[0]
    assert isinstance(outcome, _Failed)
    assert not hasattr(outcome, "output")
    assert secret not in repr(outcome)
    assert "private-ref" not in repr(outcome)


def test_private_failure_logs_exclude_backend_exception_canary(caplog: pytest.LogCaptureFixture) -> None:
    backend_canary = "BACKEND-SECRET-alice@example.test"
    anonymizer = build_synthetic_anonymizer({})
    plan = anonymizer._compile_protection_plan(AnonymizerConfig(replace=Redact(), emit_telemetry=False))
    backend = _AnchoredPhase6Backend({}, fault_stage="detect")
    flow = protection_module._ProtectionFlow(anonymizer, plan, phase6_backend=backend)

    run = flow.protect((_record("a", "text"),))
    assert isinstance(run.outcomes[0], _Failed)
    rendered = repr(run) + "\n" + "\n".join(record.getMessage() for record in caplog.records)
    assert backend_canary not in rendered


@pytest.mark.parametrize("stage", ["detect", "validate", "resolve", "close"])
def test_malformed_phase6_stage_result_fails_closed(stage: str) -> None:
    anonymizer = build_synthetic_anonymizer({"text": "label"})
    plan = anonymizer._compile_protection_plan(AnonymizerConfig(replace=Redact(), emit_telemetry=False))
    backend = _AnchoredPhase6Backend.from_entities({"text": "label"})
    backend._malformed_stage = stage
    flow = protection_module._ProtectionFlow(anonymizer, plan, phase6_backend=backend)

    assert isinstance(flow.protect((_record("a", "text"),)).outcomes[0], _Failed)


def test_failure_retry_taxonomy_is_unassigned_and_unknown() -> None:
    _, flow = _flow()
    flow.close()
    failure = cast(_Rejected, flow.protect((_record("a", "text"),)).outcomes[0]).failure
    assert failure.retry_safety.value == "unknown"
    assert failure.retry_owner.value == "unassigned"


def test_close_is_idempotent_and_does_not_close_borrowed_anonymizer() -> None:
    anonymizer, flow = _flow()
    flow.close()
    flow.close()
    rejected = flow.protect((_record("a", "text"),))
    assert isinstance(rejected.outcomes[0], _Rejected)
    assert anonymizer.run is not None
