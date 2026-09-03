# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private graph-native protection domain and synchronous lifecycle boundary."""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from data_designer.config.models import ModelConfig

from anonymizer.config.anonymizer_config import AnonymizerConfig
from anonymizer.config.models import ModelSelection
from anonymizer.config.replace_strategies import Annotate, Redact, Substitute
from anonymizer.engine.execution.accounting_admission import _AccountingRejected
from anonymizer.engine.execution.accounting_plan import _AccountingLimits
from anonymizer.engine.execution.context_admission import _ContextRejected
from anonymizer.engine.execution.context_contract import (
    _BackendArtifactClass,
    _ContextExecutionContract,
    _ContextLimits,
    _ContextOrdering,
    _ContextProfile,
    _ContextSchemaVersion,
)
from anonymizer.engine.execution.graph import _AtomicGroup, _DatumId, _RewriteGroup, _TextDatum, _trivial_graph
from anonymizer.engine.execution.invocation import _CompiledInvocation
from anonymizer.engine.execution.mention_admission import _MentionLimits
from anonymizer.engine.execution.phase6_ndd_backend import _Phase6NddBackend
from anonymizer.engine.execution.phase6_plan import _Phase6Rejected
from anonymizer.engine.execution.phase6_runtime import _Phase6EffectBackend
from anonymizer.engine.execution.phase7_contract import (
    _is_admitted_phase7_contract,
    _load_phase7_contract,
    _Phase7StableSubstituteContract,
)
from anonymizer.engine.execution.phase7_ndd_backend import _Phase7NddBackend
from anonymizer.engine.execution.phase7_runtime import _Phase7EffectBackend
from anonymizer.engine.execution.phase8_contract import (
    _is_admitted_phase8_contract,
    _load_phase8_contract,
    _Phase8GroupedRewriteContract,
)
from anonymizer.engine.execution.phase8_ndd_backend import _Phase8NddBackend
from anonymizer.engine.execution.phase8_service import _Phase8GroupedRewriteProtectionService
from anonymizer.engine.execution.protection_service import (
    _GraphProtectionFailed,
    _GraphProtectionResult,
    _GraphProtectionSucceeded,
    _Phase6RedactProtectionService,
    _Phase7SubstituteProtectionService,
)

if TYPE_CHECKING:
    from anonymizer.interface.anonymizer import Anonymizer

_MAX_REF_BYTES = 256
_MAX_SEGMENT_BYTES = 65_536
_MAX_RECORD_BYTES = 32_768
_MAX_BATCH_BYTES = 1_048_576
_MAX_RECORDS = 128
_PHASE6_MENTION_LIMITS = _MentionLimits(128, 64, 256, _MAX_RECORD_BYTES)
_PHASE6_MAX_EXPANDED_FRAME_BYTES = 65_536
_CONTRACT_VERSION = "private-protection-v1"
_REDACT_PROFILE = "redact-release-v1"
_SUBSTITUTE_PROFILE = "stable-substitute-v1"
_GROUPED_REWRITE_PROFILE = "grouped-rewrite-v1"
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
    phase7_contract: _Phase7StableSubstituteContract | None = None
    max_records: int = _MAX_RECORDS
    max_record_bytes: int = _MAX_RECORD_BYTES
    max_batch_bytes: int = _MAX_BATCH_BYTES
    phase8_contract: _Phase8GroupedRewriteContract | None = None

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


class _RetrySafety(str, Enum):
    UNKNOWN = "unknown"


class _RetryOwner(str, Enum):
    UNASSIGNED = "unassigned"


@dataclass(frozen=True, slots=True, repr=False)
class _SafeFailure(_SafeRepr):
    code: _FailureCode
    stage: str
    scope: str
    retry_safety: _RetrySafety = _RetrySafety.UNKNOWN
    retry_owner: _RetryOwner = _RetryOwner.UNASSIGNED


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
    plan_digest: str
    attempt_id: str


@dataclass(frozen=True, slots=True, repr=False)
class _Phase7ProtectionReceipt(_SafeRepr):
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
    receipt: _ProtectionReceipt | _Phase7ProtectionReceipt


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
    """Compile one explicitly release-qualified private protection profile."""
    if not isinstance(config, AnonymizerConfig):
        return _PlanInvalid()
    if isinstance(config.replace, Annotate):
        return _PlanRejected()
    if config.rewrite is not None:
        # This is deliberately private graph selection only. The public
        # Rewrite route retains its established workflow.
        if not config.rewrite.strict_entity_protection:
            return _PlanRejected()
        contract = _load_phase8_contract()
        if not isinstance(contract, _Phase8GroupedRewriteContract) or not _is_admitted_phase8_contract(contract):
            return _PlanRejected()
        invocation = _CompiledInvocation.compile(config, selected_models, model_configs)
        phase7_contract = _load_phase7_contract()
        if not isinstance(phase7_contract, _Phase7StableSubstituteContract) or not _is_admitted_phase7_contract(
            phase7_contract
        ):
            return _PlanRejected()
        max_records = min(_MAX_RECORDS, dict(contract.limits)["max_datums_per_invocation"])
        return _ProtectionPlan(
            _GROUPED_REWRITE_PROFILE,
            _plan_fingerprint(invocation, profile=_GROUPED_REWRITE_PROFILE, contract_digest=contract.digest),
            invocation,
            phase7_contract=phase7_contract,
            phase8_contract=contract,
            max_records=max_records,
        )
    if isinstance(config.replace, Redact):
        if config.replace.format_template != "[REDACTED_{label}]" or not config.replace.normalize_label:
            return _PlanRejected()
        invocation = _CompiledInvocation.compile(config, selected_models, model_configs)
        return _ProtectionPlan(
            _REDACT_PROFILE,
            _plan_fingerprint(invocation, profile=_REDACT_PROFILE),
            invocation,
        )
    if isinstance(config.replace, Substitute):
        if config.replace.instructions is not None:
            return _PlanRejected()
        contract = _load_phase7_contract()
        if not isinstance(contract, _Phase7StableSubstituteContract) or not _is_admitted_phase7_contract(contract):
            return _PlanRejected()
        invocation = _CompiledInvocation.compile(config, selected_models, model_configs)
        max_records = min(_MAX_RECORDS, dict(contract.count_limits)["max_scopes_per_invocation"])
        return _ProtectionPlan(
            _SUBSTITUTE_PROFILE,
            _plan_fingerprint(invocation, profile=_SUBSTITUTE_PROFILE, contract_digest=contract.digest),
            invocation,
            contract,
            max_records,
        )
    return _PlanUnsupported()


def _plan_fingerprint(
    invocation: _CompiledInvocation,
    *,
    profile: str,
    contract_digest: str | None = None,
) -> str:
    """Fingerprint the complete allowlisted private profile snapshot."""
    payload = {
        "contract_version": _CONTRACT_VERSION,
        "profile": profile,
        "profile_contract_digest": contract_digest,
        "implementation_version": _IMPLEMENTATION_VERSION,
        "limits": {
            "max_records": _MAX_RECORDS,
            "max_record_bytes": _MAX_RECORD_BYTES,
            "max_batch_bytes": _MAX_BATCH_BYTES,
        },
        "invocation": {
            "model_configs": [model.model_dump(mode="json") for model in invocation.model_configs],
            "selected_models": invocation.selected_models.model_dump(mode="json"),
            "gliner_detection_threshold": invocation.gliner_detection_threshold,
            "validation_max_entities_per_call": invocation.validation_max_entities_per_call,
            "validation_excerpt_window_chars": invocation.validation_excerpt_window_chars,
            "entity_labels": invocation.entity_labels,
            "replace_method": (
                invocation.replace_method.model_dump(mode="json") if invocation.replace_method is not None else None
            ),
            "rewrite": None,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_operation_plan(plan: _ProtectionPlan, records: object) -> _OperationPlan:
    if not isinstance(records, tuple) or not records:
        raise _ProtectionBatchError(_BatchFailureCode.MALFORMED_BATCH)
    if len(records) > plan.max_records:
        raise _ProtectionBatchError(_BatchFailureCode.TOO_MANY_RECORDS)
    if not all(isinstance(record, _ProtectionRecord) for record in records):
        raise _ProtectionBatchError(_BatchFailureCode.MALFORMED_BATCH)
    typed_records = records
    refs: list[str] = []
    for record in typed_records:
        ref = getattr(record, "ref", None)
        value = getattr(ref, "value", None)
        if (
            not isinstance(ref, _RecordRef)
            or not isinstance(value, str)
            or not value
            or len(value.encode("utf-8")) > _MAX_REF_BYTES
        ):
            raise _ProtectionBatchError(_BatchFailureCode.MALFORMED_BATCH)
        refs.append(value)
    if len(set(refs)) != len(refs):
        raise _ProtectionBatchError(_BatchFailureCode.DUPLICATE_REF)
    total = 0
    for record in typed_records:
        segments = getattr(record, "segments", None)
        if not isinstance(segments, tuple):
            raise _ProtectionBatchError(_BatchFailureCode.MALFORMED_BATCH)
        if len(segments) != 1:
            raise _ProtectionBatchError(_BatchFailureCode.UNSUPPORTED_CARDINALITY)
        segment = segments[0]
        if not isinstance(segment, _TextSegment):
            raise _ProtectionBatchError(_BatchFailureCode.MALFORMED_BATCH)
        text = getattr(segment, "text", None)
        if not isinstance(text, str):
            raise _ProtectionBatchError(_BatchFailureCode.MALFORMED_BATCH)
        size = len(text.encode("utf-8"))
        if size > plan.max_record_bytes:
            raise _ProtectionBatchError(_BatchFailureCode.RECORD_TOO_LARGE)
        total += size
    if total > plan.max_batch_bytes:
        raise _ProtectionBatchError(_BatchFailureCode.BATCH_TOO_LARGE)
    return _OperationPlan(typed_records)


class _ProtectionFlow(_SafeRepr):
    """Reusable synchronous flow with non-waiting whole-invocation admission."""

    def __init__(
        self,
        anonymizer: Anonymizer,
        plan: _ProtectionPlan,
        phase6_backend: _Phase6EffectBackend | None = None,
        phase7_backend: _Phase7EffectBackend | None = None,
        phase8_backend: object | None = None,
    ) -> None:
        self._plan = plan
        backend = phase6_backend or _Phase6NddBackend(anonymizer._adapter, plan.invocation)
        if plan.profile == _REDACT_PROFILE and plan.phase7_contract is None:
            self._runtime = _Phase6RedactProtectionService(
                backend,
                mention_limits=_PHASE6_MENTION_LIMITS,
            )
        elif (
            plan.profile in {_SUBSTITUTE_PROFILE, _GROUPED_REWRITE_PROFILE}
            and isinstance(plan.phase7_contract, _Phase7StableSubstituteContract)
            and _is_admitted_phase7_contract(plan.phase7_contract)
        ):
            phase7_runtime = _Phase7SubstituteProtectionService(
                backend,
                (
                    (lambda: phase7_backend)
                    if phase7_backend is not None
                    else (lambda: _Phase7NddBackend(anonymizer._adapter, plan.invocation))
                ),
                mention_limits=_PHASE6_MENTION_LIMITS,
            )
            self._runtime = (
                (phase7_runtime, _Phase8GroupedRewriteProtectionService(), phase8_backend)
                if plan.profile == _GROUPED_REWRITE_PROFILE
                else phase7_runtime
            )
        else:
            raise ValueError("private protection plan is not executable")
        self._guard = threading.Lock()
        self._state_lock = threading.Lock()
        self._closed = False
        self._adapter = anonymizer._adapter

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
        contract_digest = (
            self._plan.phase8_contract.digest
            if self._plan.phase8_contract is not None
            else self._plan.phase7_contract.digest
            if self._plan.phase7_contract is not None
            else None
        )
        if (
            _plan_fingerprint(
                self._plan.invocation,
                profile=self._plan.profile,
                contract_digest=contract_digest,
            )
            != self._plan.digest
        ):
            return self._fail_all(operation)
        datums = tuple(
            _TextDatum(_DatumId(f"datum-{index}"), record.segments[0].text)
            for index, record in enumerate(operation.records)
        )
        graph = _trivial_graph(datums)
        if self._plan.profile == _GROUPED_REWRITE_PROFILE:
            graph = type(graph)(
                graph.datums,
                graph.links,
                graph.context_scopes,
                graph.coherence_scopes,
                (_AtomicGroup(tuple(datum.id for datum in datums)),),
                graph.dependencies,
                (_RewriteGroup(tuple(datum.id for datum in datums)),),
            )
        context_limits = (
            _ContextLimits(
                max_context_members_per_target=2,
                max_context_bytes_per_target=4_096,
                max_total_context_references=4,
                max_expanded_frame_bytes=65_536,
            )
            if self._plan.profile == _GROUPED_REWRITE_PROFILE
            else _ContextLimits(
                max_context_members_per_target=0,
                max_context_bytes_per_target=0,
                max_total_context_references=0,
                max_expanded_frame_bytes=_PHASE6_MAX_EXPANDED_FRAME_BYTES,
            )
        )
        try:
            admission_runtime = self._runtime[0] if isinstance(self._runtime, tuple) else self._runtime
            admitted = admission_runtime.admit_context(
                graph,
                accounting_limits=_AccountingLimits(
                    max_datums=self._plan.max_records,
                    max_datum_bytes=self._plan.max_record_bytes,
                    max_graph_bytes=self._plan.max_batch_bytes,
                    max_stages=8,
                ),
                contract=_ContextExecutionContract(
                    profile=_ContextProfile.TARGET_CONTEXT_V1,
                    schema_version=_ContextSchemaVersion.V1,
                    limits=context_limits,
                    allow_target_as_context=False,
                    ordering=_ContextOrdering.DECLARED,
                    required_artifacts=(_BackendArtifactClass.CONTEXT_REQUEST,),
                ),
            )
            if isinstance(admitted, (_AccountingRejected, _ContextRejected, _Phase6Rejected)):
                return self._fail_all(operation)
            with self._adapter.private_execution():
                if isinstance(self._runtime, tuple):
                    phase7_runtime, phase8_service, injected_backend = self._runtime
                    contract = self._plan.phase7_contract
                    if not isinstance(contract, _Phase7StableSubstituteContract):
                        return self._fail_all(operation)
                    predecessor = phase7_runtime.execute_successor(admitted, contract=contract)
                    if predecessor is None:
                        return self._fail_all(operation)
                    backend = injected_backend or _Phase8NddBackend(self._adapter, self._plan.invocation)
                    execution = phase8_service.run_from_phase7_successor_with_backend(
                        graph, predecessor, backend, self._plan.invocation
                    )
                    execution = _phase8_as_graph_result(execution, tuple(datum.id for datum in datums))
                elif isinstance(self._runtime, _Phase7SubstituteProtectionService):
                    contract = self._plan.phase7_contract
                    if not isinstance(contract, _Phase7StableSubstituteContract):
                        return self._fail_all(operation)
                    execution = self._runtime.protect(admitted, contract=contract)
                else:
                    execution = self._runtime.protect(
                        admitted,
                        invocation=self._plan.invocation,
                    )
        except Exception:
            return self._fail_all(operation)

        try:
            return self._materialize_outcomes(operation, execution)
        except Exception as cause:
            del cause
            return self._fail_all(operation)

    def _materialize_outcomes(
        self,
        operation: _OperationPlan,
        execution: _GraphProtectionResult,
    ) -> _ProtectionRunRecord:
        expected_ids = tuple(_DatumId(f"datum-{index}") for index in range(len(operation.records)))
        outcome_by_id = {}
        for graph_outcome in execution.outcomes:
            datum_id = getattr(graph_outcome, "datum_id", None)
            if not isinstance(datum_id, _DatumId) or datum_id in outcome_by_id:
                return self._fail_all(operation)
            outcome_by_id[datum_id] = graph_outcome
        if set(outcome_by_id) != set(expected_ids):
            return self._fail_all(operation)
        if self._plan.profile == _SUBSTITUTE_PROFILE:
            receipt: _ProtectionReceipt | _Phase7ProtectionReceipt = _Phase7ProtectionReceipt(
                _CONTRACT_VERSION,
                self._plan.profile,
                _IMPLEMENTATION_VERSION,
                True,
                True,
            )
        else:
            receipt = _ProtectionReceipt(
                _CONTRACT_VERSION,
                self._plan.profile,
                _IMPLEMENTATION_VERSION,
                True,
                True,
                self._plan.digest,
                secrets.token_hex(16),
            )
        outcomes: list[_RecordOutcome] = []
        for record, datum_id in zip(operation.records, expected_ids, strict=True):
            graph_outcome = outcome_by_id[datum_id]
            if isinstance(graph_outcome, _GraphProtectionFailed):
                code = (
                    _FailureCode.INVOCATION_FAILED if graph_outcome.scope == "invocation" else _FailureCode.ROW_FAILED
                )
                scope = "record" if graph_outcome.scope == "datum" else graph_outcome.scope
                outcomes.append(_Failed(record.ref, _failure(code, graph_outcome.stage, scope)))
            elif isinstance(graph_outcome, _GraphProtectionSucceeded):
                disposition: _SuccessDisposition
                disposition = _ProtectionApplied() if graph_outcome.applied else _NoAcceptedDetections()
                outcomes.append(_ProtectionSucceeded(record.ref, graph_outcome.output, disposition, receipt))
            else:
                return self._fail_all(operation)
        return _ProtectionRunRecord(tuple(outcomes))

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


def _phase8_as_graph_result(execution: object, expected_ids: tuple[_DatumId, ...]) -> _GraphProtectionResult:
    """Materialize only the complete Phase 8 release projection.

    Phase 7 candidates deliberately never cross this boundary: a datum is
    successful only when Phase 8's own cleanup-qualified Phase 4 reduction
    released a replacement for its exact private key.
    """
    released = getattr(execution, "released", None)
    cleanup_verified = getattr(execution, "cleanup_verified", False)
    global_embargo = getattr(execution, "global_embargo", True)
    if not isinstance(released, tuple) or not cleanup_verified or global_embargo:
        return _phase8_failed_graph_result(expected_ids)
    values = {
        datum_id: value for datum_id, value in released if isinstance(datum_id, _DatumId) and isinstance(value, str)
    }
    if len(values) != len(released):
        return _phase8_failed_graph_result(expected_ids)
    return _GraphProtectionResult(
        tuple(
            _GraphProtectionSucceeded(datum_id, values[datum_id], True)
            if datum_id in values
            else _GraphProtectionFailed(datum_id, "phase8", "datum")
            for datum_id in expected_ids
        )
    )


def _phase8_failed_graph_result(expected_ids: tuple[_DatumId, ...]) -> _GraphProtectionResult:
    return _GraphProtectionResult(
        tuple(_GraphProtectionFailed(datum_id, "phase8", "datum") for datum_id in expected_ids)
    )


def _failure(code: _FailureCode, stage: str, scope: str) -> _SafeFailure:
    return _SafeFailure(code, stage, scope)
