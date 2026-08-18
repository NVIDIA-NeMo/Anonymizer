# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private Plan A protection domain and synchronous lifecycle boundary."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import pandas as pd
from data_designer.config.models import ModelConfig

from anonymizer.config.anonymizer_config import AnonymizerConfig
from anonymizer.config.models import ModelSelection
from anonymizer.config.replace_strategies import Annotate, Redact
from anonymizer.engine.constants import COL_FINAL_ENTITIES, COL_REPLACED_TEXT, COL_TEXT
from anonymizer.engine.execution.invocation import _CompiledInvocation
from anonymizer.engine.execution.pandas_runtime import _PandasExecutionResult, _PandasRuntime
from anonymizer.engine.private_row_verification import _InvocationRowVerifier, _TerminalOutcome

if TYPE_CHECKING:
    from anonymizer.interface.anonymizer import Anonymizer

_MAX_REF_BYTES = 256
_MAX_SEGMENT_BYTES = 65_536
_MAX_RECORD_BYTES = 32_768
_MAX_BATCH_BYTES = 1_048_576
_MAX_RECORDS = 128
_CONTRACT_VERSION = "private-protection-v1"
_PROFILE = "redact-release-v1"
_IMPLEMENTATION_VERSION = "pandas-runtime-v1"


class _SafeRepr:
    def __repr__(self) -> str:
        return f"<private {type(self).__name__.strip('_').replace('_', ' ').lower()}>"


@dataclass(frozen=True, slots=True, repr=False)
class _RecordRef(_SafeRepr):
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value or len(self.value.encode("utf-8")) > _MAX_REF_BYTES:
            raise ValueError("record reference is invalid")


@dataclass(frozen=True, slots=True, repr=False)
class _TextSegment(_SafeRepr):
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or len(self.text.encode("utf-8")) > _MAX_SEGMENT_BYTES:
            raise ValueError("text segment is invalid")


@dataclass(frozen=True, slots=True, repr=False)
class _ProtectionRecord(_SafeRepr):
    ref: _RecordRef
    segments: tuple[_TextSegment, ...]


class _CompileCode(str, Enum):
    INVALID = "invalid"
    UNSUPPORTED = "unsupported"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True, repr=False)
class _PlanInvalid(_SafeRepr):
    code: _CompileCode = _CompileCode.INVALID


@dataclass(frozen=True, slots=True, repr=False)
class _PlanUnsupported(_SafeRepr):
    code: _CompileCode = _CompileCode.UNSUPPORTED


@dataclass(frozen=True, slots=True, repr=False)
class _PlanRejected(_SafeRepr):
    code: _CompileCode = _CompileCode.REJECTED


@dataclass(frozen=True, slots=True, repr=False)
class _ProtectionPlan(_SafeRepr):
    profile: str
    digest: str
    invocation: _CompiledInvocation
    max_records: int = _MAX_RECORDS
    max_record_bytes: int = _MAX_RECORD_BYTES
    max_batch_bytes: int = _MAX_BATCH_BYTES

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private protection plan is not serializable")


_CompileResult = _ProtectionPlan | _PlanInvalid | _PlanUnsupported | _PlanRejected


class _BatchFailureCode(str, Enum):
    MALFORMED_BATCH = "malformed_batch"
    DUPLICATE_REF = "duplicate_ref"
    RECORD_TOO_LARGE = "record_too_large"
    BATCH_TOO_LARGE = "batch_too_large"
    TOO_MANY_RECORDS = "too_many_records"
    UNSUPPORTED_CARDINALITY = "unsupported_cardinality"


class _ProtectionBatchError(ValueError):
    def __init__(self, code: _BatchFailureCode) -> None:
        self.code = code
        super().__init__("private protection batch rejected")

    def __repr__(self) -> str:
        return "<private protection batch error>"


class _FailureCode(str, Enum):
    BUSY = "busy"
    CLOSED = "closed"
    CANCELLED_BEFORE_ADMISSION = "cancelled_before_admission"
    INVOCATION_FAILED = "invocation_failed"
    ROW_FAILED = "row_failed"


@dataclass(frozen=True, slots=True, repr=False)
class _SafeFailure(_SafeRepr):
    code: _FailureCode
    stage: str
    scope: str
    retry_safe: bool
    retry_owner: str = "caller"


@dataclass(frozen=True, slots=True, repr=False)
class _ProtectionApplied(_SafeRepr):
    pass


@dataclass(frozen=True, slots=True, repr=False)
class _NoAcceptedDetections(_SafeRepr):
    pass


_SuccessDisposition = _ProtectionApplied | _NoAcceptedDetections


@dataclass(frozen=True, slots=True, repr=False)
class _ProtectionReceipt(_SafeRepr):
    contract_version: str
    profile: str
    implementation_version: str
    terminal_accounting_verified: bool
    accepted_detections_verified: bool


@dataclass(frozen=True, slots=True, repr=False)
class _ProtectionSucceeded(_SafeRepr):
    ref: _RecordRef
    output: str
    disposition: _SuccessDisposition
    receipt: _ProtectionReceipt


@dataclass(frozen=True, slots=True, repr=False)
class _Rejected(_SafeRepr):
    ref: _RecordRef
    failure: _SafeFailure


@dataclass(frozen=True, slots=True, repr=False)
class _Failed(_SafeRepr):
    ref: _RecordRef
    failure: _SafeFailure


@dataclass(frozen=True, slots=True, repr=False)
class _Cancelled(_SafeRepr):
    ref: _RecordRef
    failure: _SafeFailure


_RecordOutcome = _ProtectionSucceeded | _Rejected | _Failed | _Cancelled


@dataclass(frozen=True, slots=True, repr=False)
class _ProtectionRunRecord(_SafeRepr):
    outcomes: tuple[_RecordOutcome, ...]
    contract_version: str = _CONTRACT_VERSION
    implementation_version: str = _IMPLEMENTATION_VERSION

    @property
    def success_count(self) -> int:
        return sum(isinstance(outcome, _ProtectionSucceeded) for outcome in self.outcomes)

    @property
    def failure_count(self) -> int:
        return len(self.outcomes) - self.success_count


@dataclass(frozen=True, slots=True, repr=False)
class _OperationPlan(_SafeRepr):
    records: tuple[_ProtectionRecord, ...]


def _compile_protection_plan(
    config: AnonymizerConfig,
    selected_models: ModelSelection,
    model_configs: list[ModelConfig],
) -> _CompileResult:
    """Purely compile the one release-qualified Plan A profile."""
    if not isinstance(config, AnonymizerConfig):
        return _PlanInvalid()
    if isinstance(config.replace, Annotate):
        return _PlanRejected()
    if not isinstance(config.replace, Redact) or config.rewrite is not None:
        return _PlanUnsupported()
    if config.replace.format_template != "[REDACTED_{label}]" or not config.replace.normalize_label:
        return _PlanRejected()
    invocation = _CompiledInvocation.compile(config, selected_models, model_configs)
    digest_input = (
        f"{_CONTRACT_VERSION}|{_PROFILE}|{invocation.gliner_detection_threshold}|"
        f"{invocation.validation_max_entities_per_call}|{invocation.validation_excerpt_window_chars}|"
        f"{invocation.entity_labels!r}"
    ).encode()
    return _ProtectionPlan(_PROFILE, hashlib.sha256(digest_input).hexdigest(), invocation)


def _build_operation_plan(plan: _ProtectionPlan, records: object) -> _OperationPlan:
    if not isinstance(records, tuple) or not records:
        raise _ProtectionBatchError(_BatchFailureCode.MALFORMED_BATCH)
    if len(records) > plan.max_records:
        raise _ProtectionBatchError(_BatchFailureCode.TOO_MANY_RECORDS)
    if not all(isinstance(record, _ProtectionRecord) for record in records):
        raise _ProtectionBatchError(_BatchFailureCode.MALFORMED_BATCH)
    typed_records = records
    if not all(isinstance(record.ref, _RecordRef) for record in typed_records):
        raise _ProtectionBatchError(_BatchFailureCode.MALFORMED_BATCH)
    refs = [record.ref.value for record in typed_records]
    if len(set(refs)) != len(refs):
        raise _ProtectionBatchError(_BatchFailureCode.DUPLICATE_REF)
    total = 0
    for record in typed_records:
        if len(record.segments) != 1:
            raise _ProtectionBatchError(_BatchFailureCode.UNSUPPORTED_CARDINALITY)
        size = len(record.segments[0].text.encode("utf-8"))
        if size > plan.max_record_bytes:
            raise _ProtectionBatchError(_BatchFailureCode.RECORD_TOO_LARGE)
        total += size
    if total > plan.max_batch_bytes:
        raise _ProtectionBatchError(_BatchFailureCode.BATCH_TOO_LARGE)
    return _OperationPlan(typed_records)


class _ProtectionFlow(_SafeRepr):
    """Reusable synchronous flow with non-waiting whole-invocation admission."""

    def __init__(self, anonymizer: Anonymizer, plan: _ProtectionPlan) -> None:
        self._plan = plan
        self._runtime = _PandasRuntime(
            detection_workflow=anonymizer._detection_workflow,
            replace_runner=anonymizer._replace_runner,
            rewrite_runner=anonymizer._rewrite_runner,
            combined_rewrite_runner=anonymizer._combined_rewrite_runner,
        )
        self._guard = threading.Lock()
        self._state_lock = threading.Lock()
        self._closed = False

    def protect(self, records: object, *, cancelled_before_admission: bool = False) -> _ProtectionRunRecord:
        operation = _build_operation_plan(self._plan, records)
        if cancelled_before_admission:
            return self._reject_all(operation, _FailureCode.CANCELLED_BEFORE_ADMISSION)
        with self._state_lock:
            if self._closed:
                return self._reject_all(operation, _FailureCode.CLOSED)
        if not self._guard.acquire(blocking=False):
            return self._reject_all(operation, _FailureCode.BUSY)
        try:
            with self._state_lock:
                if self._closed:
                    return self._reject_all(operation, _FailureCode.CLOSED)
            return self._execute(operation)
        finally:
            self._guard.release()

    def _execute(self, operation: _OperationPlan) -> _ProtectionRunRecord:
        frame = pd.DataFrame({COL_TEXT: [record.segments[0].text for record in operation.records]})
        verifier = _InvocationRowVerifier(frame)
        bound = verifier.bind(frame)
        try:
            execution = self._runtime.run(
                bound,
                invocation=self._plan.invocation,
                data_summary=None,
                preview_num_records=None,
                verifier=verifier,
            )
        except Exception as cause:
            verifier.abort_with_failure(stage="pipeline", cause=cause)
            terminal = verifier.take_terminal_outcomes()
            del cause
            return self._failed_from_terminal(operation, terminal)

        try:
            return self._materialize_outcomes(operation, execution)
        except Exception as cause:
            del cause
            return self._fail_all(operation)

    def _materialize_outcomes(
        self, operation: _OperationPlan, execution: _PandasExecutionResult
    ) -> _ProtectionRunRecord:
        token_to_row = {
            token: row
            for token, (_, row) in zip(execution.result_row_tokens, execution.dataframe.iterrows(), strict=True)
        }
        receipt = _ProtectionReceipt(_CONTRACT_VERSION, _PROFILE, _IMPLEMENTATION_VERSION, True, True)
        outcomes: list[_RecordOutcome] = []
        for record, (token, status) in zip(operation.records, execution.terminal_outcomes, strict=True):
            if status is not _TerminalOutcome.SUCCESS:
                outcomes.append(_Failed(record.ref, _failure(_FailureCode.ROW_FAILED, "pipeline", "record")))
                continue
            row = token_to_row[token]
            output = row[COL_REPLACED_TEXT]
            if not isinstance(output, str):
                return self._fail_all(operation)
            has_detections = _has_accepted_detections(row[COL_FINAL_ENTITIES])
            if has_detections and not _redact_release_passed(row[COL_FINAL_ENTITIES], output):
                outcomes.append(_Failed(record.ref, _failure(_FailureCode.ROW_FAILED, "release", "record")))
                continue
            if not has_detections and output != record.segments[0].text:
                outcomes.append(_Failed(record.ref, _failure(_FailureCode.ROW_FAILED, "release", "record")))
                continue
            disposition: _SuccessDisposition = _ProtectionApplied() if has_detections else _NoAcceptedDetections()
            outcomes.append(_ProtectionSucceeded(record.ref, output, disposition, receipt))
        return _ProtectionRunRecord(tuple(outcomes))

    def _failed_from_terminal(
        self, operation: _OperationPlan, terminal: tuple[tuple[str, _TerminalOutcome], ...]
    ) -> _ProtectionRunRecord:
        del terminal
        return self._fail_all(operation)

    def _fail_all(self, operation: _OperationPlan) -> _ProtectionRunRecord:
        failure = _failure(_FailureCode.INVOCATION_FAILED, "pipeline", "invocation")
        return _ProtectionRunRecord(tuple(_Failed(record.ref, failure) for record in operation.records))

    def _reject_all(self, operation: _OperationPlan, code: _FailureCode) -> _ProtectionRunRecord:
        failure = _failure(code, "admission", "invocation")
        return _ProtectionRunRecord(tuple(_Rejected(record.ref, failure) for record in operation.records))

    def close(self) -> None:
        """Reject new admission; borrowed Anonymizer resources are untouched."""
        with self._state_lock:
            self._closed = True

    def __enter__(self) -> _ProtectionFlow:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _failure(code: _FailureCode, stage: str, scope: str) -> _SafeFailure:
    return _SafeFailure(code, stage, scope, retry_safe=True)


def _has_accepted_detections(value: object) -> bool:
    if isinstance(value, dict):
        entities = value.get("entities", [])
    else:
        entities = getattr(value, "entities", [])
    return bool(entities)


def _redact_release_passed(value: object, output: str) -> bool:
    """Require every accepted entity value to be absent from released text."""
    if isinstance(value, dict):
        entities = value.get("entities", [])
    else:
        entities = getattr(value, "entities", [])
    for entity in entities:
        raw = entity.get("value") if isinstance(entity, dict) else getattr(entity, "value", None)
        if not isinstance(raw, str) or not raw or raw in output:
            return False
    return True
