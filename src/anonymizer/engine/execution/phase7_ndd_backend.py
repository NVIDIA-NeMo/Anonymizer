# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded NDD candidate proposals for private Phase 7 Substitute scopes."""

from __future__ import annotations

import json
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TypeGuard, TypeVar

import pandas as pd
from data_designer.config.column_configs import LLMStructuredColumnConfig
from pydantic import BaseModel, ConfigDict, StrictStr

from anonymizer.engine.constants import (
    COL_ATTEMPT_ID,
    COL_PHASE7_CANDIDATE_BUNDLE,
    COL_PHASE7_CANDIDATE_REQUEST,
    COL_PHASE7_INVOCATION_ID,
    COL_TARGET_WORK_ID,
    COL_TASK_ID,
    _jinja,
)
from anonymizer.engine.execution.accounting_evidence import (
    _AttemptId,
    _Dispatch,
    _InvocationId,
    _RowToken,
)
from anonymizer.engine.execution.accounting_plan import _ScopeTaskSubject, _TaskKey
from anonymizer.engine.execution.invocation import _CompiledInvocation
from anonymizer.engine.execution.phase7_admission import (
    _is_admitted_scope_manifest,
    _ReplacementSlotId,
    _ScopeManifest,
)
from anonymizer.engine.execution.phase7_contract import (
    _is_admitted_phase7_contract,
    _Phase7StableSubstituteContract,
)
from anonymizer.engine.execution.phase7_validation import (
    _CandidateAssignment,
    _index_scope_sources,
    _ScopeSourceIndex,
)
from anonymizer.engine.ndd.adapter import (
    RECORD_ID_COLUMN,
    FailedRecord,
    NddAdapter,
    WorkflowRunResult,
    _FailedRowEvidence,
)
from anonymizer.engine.ndd.model_loader import resolve_model_alias

T = TypeVar("T", bound=BaseModel)


class _PrivatePhase7NddValue:
    def __repr__(self) -> str:
        return f"<private {type(self).__name__.strip('_').replace('_', ' ').lower()}>"

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private Phase 7 NDD values are not serializable")


class _Phase7NddStatus(str, Enum):
    NO_WORK = "no_work"
    CANDIDATE = "candidate"
    TASK_FAILED = "task_failed"
    INVOCATION_INCONSISTENT = "invocation_inconsistent"


class _Phase7NddReason(str, Enum):
    BACKEND_FAILED = "backend_failed"
    EVIDENCE_UNATTRIBUTABLE = "evidence_unattributable"
    LIMIT_EXCEEDED = "limit_exceeded"
    CONTRACT_INVALID = "contract_invalid"
    PHASE6_HANDOFF_MISMATCH = "phase6_handoff_mismatch"


@dataclass(frozen=True, slots=True, repr=False)
class _Phase7NddResult(_PrivatePhase7NddValue):
    status: _Phase7NddStatus
    assignments: tuple[_CandidateAssignment, ...] = ()
    reason: _Phase7NddReason | None = None


class _WireAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot_token: StrictStr
    value: StrictStr


class _WireBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assignments: list[_WireAssignment]


@dataclass(frozen=True, slots=True, repr=False)
class _SlotBinding(_PrivatePhase7NddValue):
    token: str
    slot_id: _ReplacementSlotId


@dataclass(frozen=True, slots=True, repr=False)
class _CandidateWorkframe(_PrivatePhase7NddValue):
    dataframe: pd.DataFrame
    dispatch: _Dispatch
    task_token: str
    slots: tuple[_SlotBinding, ...]


def _default_identity() -> str:
    return secrets.token_hex(16)


class _Phase7NddBackend(_PrivatePhase7NddValue):
    """Propose and structurally reconcile one complete scope bundle."""

    def __init__(
        self,
        adapter: NddAdapter,
        invocation: _CompiledInvocation,
        *,
        identity_factory: Callable[[], str] = _default_identity,
    ) -> None:
        self._adapter = adapter
        self._invocation = invocation
        self._identity_factory = identity_factory

    def propose_scope(
        self,
        manifest: object,
        handoffs: object,
        contract: object,
        dispatch: object,
    ) -> _Phase7NddResult:
        prepared = _prepare_scope(manifest, handoffs, contract)
        if isinstance(prepared, _Phase7NddResult):
            return prepared
        admitted_manifest, admitted_contract, sources = prepared
        if not admitted_manifest.slots:
            if dispatch is not None:
                return _inconsistent(_Phase7NddReason.EVIDENCE_UNATTRIBUTABLE)
            return _Phase7NddResult(_Phase7NddStatus.NO_WORK)
        if not _valid_dispatch(dispatch):
            return _inconsistent(_Phase7NddReason.EVIDENCE_UNATTRIBUTABLE)
        if not isinstance(self._invocation, _CompiledInvocation):
            return _inconsistent(_Phase7NddReason.CONTRACT_INVALID)
        workframe = _lower_candidate_workframe(
            admitted_manifest,
            sources,
            admitted_contract,
            dispatch,
            identity_factory=self._identity_factory,
        )
        if workframe is None:
            return _inconsistent(_Phase7NddReason.LIMIT_EXCEEDED)
        workflow = self._run_candidate_workflow(workframe)
        if isinstance(workflow, _Phase7NddResult):
            return workflow
        return _reconcile_candidate_result(workframe, workflow)

    def _run_candidate_workflow(
        self,
        workframe: _CandidateWorkframe,
    ) -> WorkflowRunResult | _Phase7NddResult:
        try:
            with self._adapter.private_execution():
                return self._adapter.run_workflow(
                    workframe.dataframe,
                    model_configs=list(self._invocation.model_configs),
                    columns=[_candidate_column(self._invocation)],
                    workflow_name="phase7-candidate-planning",
                )
        except Exception as error:
            del error
            return _Phase7NddResult(
                _Phase7NddStatus.TASK_FAILED,
                reason=_Phase7NddReason.BACKEND_FAILED,
            )


def _prepare_scope(
    manifest: object,
    handoffs: object,
    contract: object,
) -> tuple[_ScopeManifest, _Phase7StableSubstituteContract, _ScopeSourceIndex] | _Phase7NddResult:
    if not isinstance(contract, _Phase7StableSubstituteContract) or not _is_admitted_phase7_contract(contract):
        return _inconsistent(_Phase7NddReason.CONTRACT_INVALID)
    if not isinstance(manifest, _ScopeManifest) or not _is_admitted_scope_manifest(manifest):
        return _inconsistent(_Phase7NddReason.CONTRACT_INVALID)
    if not isinstance(handoffs, tuple):
        return _inconsistent(_Phase7NddReason.PHASE6_HANDOFF_MISMATCH)
    sources = _index_scope_sources(manifest, handoffs)
    if sources is None:
        return _inconsistent(_Phase7NddReason.PHASE6_HANDOFF_MISMATCH)
    return manifest, contract, sources


def _candidate_column(invocation: _CompiledInvocation) -> LLMStructuredColumnConfig:
    return LLMStructuredColumnConfig(
        name=COL_PHASE7_CANDIDATE_BUNDLE,
        prompt="""Propose exactly one complete candidate assignment for this private scope.
Request: """
        + _jinja(COL_PHASE7_CANDIDATE_REQUEST)
        + """
Return every opaque slot_token exactly once. Values are proposals and will be validated separately.""",
        model_alias=resolve_model_alias(
            "replacement_generator",
            invocation.selected_models.replace,
        ),
        output_format=_WireBundle,
    )


def _valid_dispatch(value: object) -> TypeGuard[_Dispatch]:
    return (
        isinstance(value, _Dispatch)
        and isinstance(value.invocation_id, _InvocationId)
        and type(value.invocation_id.value) is str
        and bool(value.invocation_id.value)
        and isinstance(value.task, _TaskKey)
        and isinstance(value.task.subject, _ScopeTaskSubject)
        and isinstance(value.attempt_id, _AttemptId)
        and type(value.attempt_id.value) is str
        and bool(value.attempt_id.value)
        and isinstance(value.row_token, _RowToken)
        and type(value.row_token.value) is str
        and bool(value.row_token.value)
    )


def _lower_candidate_workframe(
    manifest: _ScopeManifest,
    sources: _ScopeSourceIndex,
    contract: _Phase7StableSubstituteContract,
    dispatch: _Dispatch,
    *,
    identity_factory: Callable[[], str],
) -> _CandidateWorkframe | None:
    try:
        used = {
            dispatch.invocation_id.value,
            dispatch.attempt_id.value,
            dispatch.row_token.value,
        }
        task_token = _claim_identity(identity_factory(), used)
        bindings = tuple(_SlotBinding(_claim_identity(identity_factory(), used), slot.id) for slot in manifest.slots)
        request = _candidate_request(manifest, sources, bindings)
        encoded_request = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
        dataframe = _candidate_dataframe(
            dispatch,
            task_token,
            encoded_request,
            max_bytes=dict(contract.byte_limits)["max_workframe_bytes_per_scope"],
        )
        if dataframe is None:
            return None
    except (KeyError, StopIteration, TypeError, UnicodeEncodeError, ValueError):
        return None
    return _CandidateWorkframe(dataframe, dispatch, task_token, bindings)


def _candidate_request(
    manifest: _ScopeManifest,
    sources: _ScopeSourceIndex,
    bindings: tuple[_SlotBinding, ...],
) -> dict[str, object]:
    wire_token = {binding.slot_id: binding.token for binding in bindings}
    mention_by_id = dict(sources.mentions)
    return {
        "schema_version": "phase7-workframe/v1",
        "slots": [
            {
                "slot_token": binding.token,
                "role": slot.role,
                "format": slot.format,
                "mask": slot.mask,
                "source_values": [mention_by_id[mention_id].source_slice for mention_id in slot.mention_ids],
            }
            for slot, binding in zip(manifest.slots, bindings, strict=True)
        ],
        "required_distinct_pairs": [
            {"left": wire_token[pair.left], "right": wire_token[pair.right]} for pair in manifest.required_pairs
        ],
        "relations": [
            {
                "version": relation.version,
                "upstream": [wire_token[token] for token in relation.upstream],
                "downstream": wire_token[relation.downstream],
            }
            for relation in manifest.relations
        ],
    }


def _candidate_dataframe(
    dispatch: _Dispatch,
    task_token: str,
    request: str,
    *,
    max_bytes: int,
) -> pd.DataFrame | None:
    row = {
        COL_TARGET_WORK_ID: dispatch.row_token.value,
        COL_PHASE7_INVOCATION_ID: dispatch.invocation_id.value,
        COL_TASK_ID: task_token,
        COL_ATTEMPT_ID: dispatch.attempt_id.value,
        COL_PHASE7_CANDIDATE_REQUEST: request,
        RECORD_ID_COLUMN: dispatch.row_token.value,
    }
    encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > max_bytes:
        return None
    return pd.DataFrame(
        [row],
        columns=pd.Index(
            [
                COL_TARGET_WORK_ID,
                COL_PHASE7_INVOCATION_ID,
                COL_TASK_ID,
                COL_ATTEMPT_ID,
                COL_PHASE7_CANDIDATE_REQUEST,
                RECORD_ID_COLUMN,
            ]
        ),
    )


def _claim_identity(value: object, used: set[str]) -> str:
    if type(value) is not str or not value or value in used:
        raise TypeError
    used.add(value)
    return value


def _reconcile_candidate_result(
    workframe: _CandidateWorkframe,
    result: object,
) -> _Phase7NddResult:
    if not isinstance(result, WorkflowRunResult):
        return _inconsistent(_Phase7NddReason.EVIDENCE_UNATTRIBUTABLE)
    if not isinstance(result.failed_records, list) or not isinstance(result.failed_row_evidence, tuple):
        return _inconsistent(_Phase7NddReason.EVIDENCE_UNATTRIBUTABLE)
    if result.failed_records:
        return _reconcile_failed_result(workframe, result)
    if result.failed_row_evidence:
        return _inconsistent(_Phase7NddReason.EVIDENCE_UNATTRIBUTABLE)
    dataframe = result.dataframe
    if not isinstance(dataframe, pd.DataFrame) or len(dataframe) != 1:
        return _inconsistent(_Phase7NddReason.EVIDENCE_UNATTRIBUTABLE)
    expected_columns = {
        COL_TARGET_WORK_ID: workframe.dispatch.row_token.value,
        COL_PHASE7_INVOCATION_ID: workframe.dispatch.invocation_id.value,
        COL_TASK_ID: workframe.task_token,
        COL_ATTEMPT_ID: workframe.dispatch.attempt_id.value,
    }
    if not _has_exact_correlations(dataframe, expected_columns):
        return _inconsistent(_Phase7NddReason.EVIDENCE_UNATTRIBUTABLE)
    if COL_PHASE7_CANDIDATE_BUNDLE not in dataframe.columns:
        return _inconsistent(_Phase7NddReason.EVIDENCE_UNATTRIBUTABLE)
    bundle = _coerce_model(dataframe.iloc[0][COL_PHASE7_CANDIDATE_BUNDLE], _WireBundle)
    if bundle is None:
        return _inconsistent(_Phase7NddReason.EVIDENCE_UNATTRIBUTABLE)
    binding_by_token = {binding.token: binding.slot_id for binding in workframe.slots}
    observed = tuple(item.slot_token for item in bundle.assignments)
    if len(set(observed)) != len(observed) or set(observed) != set(binding_by_token):
        return _inconsistent(_Phase7NddReason.EVIDENCE_UNATTRIBUTABLE)
    value_by_token = {item.slot_token: item.value for item in bundle.assignments}
    assignments = tuple(
        _CandidateAssignment(binding.slot_id, value_by_token[binding.token]) for binding in workframe.slots
    )
    return _Phase7NddResult(_Phase7NddStatus.CANDIDATE, assignments)


def _has_exact_correlations(dataframe: pd.DataFrame, expected: dict[str, str]) -> bool:
    for column, expected_value in expected.items():
        if dataframe.columns.tolist().count(column) != 1:
            return False
        observed = dataframe.iloc[0][column]
        if type(observed) is not str or observed != expected_value:
            return False
    return True


def _reconcile_failed_result(
    workframe: _CandidateWorkframe,
    result: WorkflowRunResult,
) -> _Phase7NddResult:
    dataframe = result.dataframe
    if not isinstance(dataframe, pd.DataFrame) or not dataframe.empty:
        return _inconsistent(_Phase7NddReason.EVIDENCE_UNATTRIBUTABLE)
    if len(result.failed_records) != 1 or len(result.failed_row_evidence) != 1:
        return _inconsistent(_Phase7NddReason.EVIDENCE_UNATTRIBUTABLE)
    evidence = result.failed_row_evidence[0]
    record = result.failed_records[0]
    if (
        not isinstance(evidence, _FailedRowEvidence)
        or not isinstance(record, FailedRecord)
        or type(evidence.row_token) is not str
        or evidence.row_token != workframe.dispatch.row_token.value
        or evidence.record is not record
    ):
        return _inconsistent(_Phase7NddReason.EVIDENCE_UNATTRIBUTABLE)
    return _Phase7NddResult(
        _Phase7NddStatus.TASK_FAILED,
        reason=_Phase7NddReason.BACKEND_FAILED,
    )


def _coerce_model(raw: object, model: type[T]) -> T | None:
    try:
        if isinstance(raw, model):
            return raw
        if isinstance(raw, str):
            raw = json.loads(raw)
        return model.model_validate(raw)
    except Exception as error:
        del error
        return None


def _inconsistent(reason: _Phase7NddReason) -> _Phase7NddResult:
    return _Phase7NddResult(_Phase7NddStatus.INVOCATION_INCONSISTENT, reason=reason)
