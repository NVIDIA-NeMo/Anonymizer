# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private target/context framing, reconciliation, and cleanup."""

from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from anonymizer.engine.constants import (
    COL_ATTEMPT_ID,
    COL_CONTEXT_BINDING_ID,
    COL_CONTEXT_ORDINAL,
    COL_CONTEXT_OWNER_WORK_ID,
    COL_CONTEXT_TEXT,
    COL_TARGET_WORK_ID,
    COL_TASK_ID,
    COL_TEXT,
)
from anonymizer.engine.execution.accounting_evidence import _Dispatch
from anonymizer.engine.execution.accounting_plan import _TaskKey
from anonymizer.engine.execution.context_admission import (
    _CompiledContextBinding,
    _CompiledContextProjection,
    _ContextPlan,
    _is_admitted_context_plan,
)
from anonymizer.engine.execution.context_contract import _BackendArtifactClass, _ContextSchemaVersion
from anonymizer.engine.execution.graph import _DatumId


class _PrivateWorkframeValue:
    def __repr__(self) -> str:
        return f"<private {type(self).__name__.strip('_').replace('_', ' ').lower()}>"

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private context workframe values are not serializable")


@dataclass(frozen=True, slots=True, repr=False)
class _TargetWorkId(_PrivateWorkframeValue):
    value: str


@dataclass(frozen=True, slots=True, repr=False)
class _ContextBindingId(_PrivateWorkframeValue):
    value: str


@dataclass(frozen=True, slots=True, repr=False)
class _ContextPayloadToken(_PrivateWorkframeValue):
    value: str


class _ContextPayload(str):
    _token: _ContextPayloadToken

    def __new__(cls, text: str, token: _ContextPayloadToken) -> _ContextPayload:
        value = super().__new__(cls, text)
        value._token = token
        return value

    @property
    def token(self) -> _ContextPayloadToken:
        return self._token

    def __repr__(self) -> str:
        return "<private context payload>"

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private context payload is not serializable")


_ExpectedBinding = tuple[
    _ContextBindingId,
    _TargetWorkId,
    int,
    _CompiledContextBinding,
    _ContextPayloadToken,
]


@dataclass(frozen=True, slots=True, repr=False)
class _BackendArtifactId(_PrivateWorkframeValue):
    value: str


@dataclass(frozen=True, slots=True, repr=False)
class _ContextBindingEvidence(_PrivateWorkframeValue):
    binding_id: _ContextBindingId
    owner_target_work_id: _TargetWorkId
    ordinal: int
    payload_token: _ContextPayloadToken


@dataclass(frozen=True, slots=True, repr=False)
class _BackendClosureAttestation(_PrivateWorkframeValue):
    artifact_id: _BackendArtifactId
    artifact_class: _BackendArtifactClass
    closed: bool
    schema_version: _ContextSchemaVersion = _ContextSchemaVersion.V1


class _ContextReconciliationStatus(str, Enum):
    VERIFIED = "verified"
    LOCAL_INVALID = "local_invalid"
    GLOBAL_INVALID = "global_invalid"


class _ContextBindingFault(str, Enum):
    MISSING = "missing"
    DUPLICATE = "duplicate"
    CONTRADICTORY = "contradictory"


class _ContextCleanupStatus(str, Enum):
    VERIFIED = "verified"
    FAILED = "failed"
    UNCONFIRMED = "unconfirmed"


@dataclass(frozen=True, slots=True, repr=False)
class _ContextReconciliation(_PrivateWorkframeValue):
    status: _ContextReconciliationStatus
    affected_tasks: tuple[_TaskKey, ...] = ()
    faults: tuple[tuple[_TaskKey, _ContextBindingFault], ...] = ()


@dataclass(frozen=True, slots=True, repr=False)
class _ContextCleanup(_PrivateWorkframeValue):
    status: _ContextCleanupStatus


class _WorkframeStateError(RuntimeError):
    def __init__(self, message: str = "private context workframe state violation") -> None:
        super().__init__(message)

    def __repr__(self) -> str:
        return "<private context workframe error>"


class _WorkframeClosedError(_WorkframeStateError):
    pass


class _WorkframeConstructionError(_WorkframeStateError):
    pass


def _default_identity() -> str:
    return secrets.token_hex(16)


class _ContextWorkframes(_PrivateWorkframeValue):
    """Owned bounded frames and maps for one dispatched target frontier."""

    def __init__(
        self,
        *,
        target_frame: pd.DataFrame,
        context_frame: pd.DataFrame,
        tasks: tuple[_TaskKey, ...],
        expected: tuple[_ExpectedBinding, ...],
        artifact_id: _BackendArtifactId,
        required_artifacts: tuple[_BackendArtifactClass, ...],
        schema_version: _ContextSchemaVersion,
    ) -> None:
        self._target_frame = target_frame
        self._context_frame = context_frame
        self._tasks = tasks
        self._expected = expected
        self._artifact_id: _BackendArtifactId | None = artifact_id
        self._required_artifacts = required_artifacts
        self._schema_version = schema_version
        self._closed = False
        self._reconciled = False
        self._dispatches_bound = False

    @property
    def target_frame(self) -> pd.DataFrame:
        if self._closed:
            return self._target_frame.iloc[0:0].copy()
        return self._target_frame.copy()

    @property
    def context_frame(self) -> pd.DataFrame:
        if self._closed:
            return self._context_frame.iloc[0:0].copy()
        return self._context_frame.copy()

    @property
    def tasks(self) -> tuple[_TaskKey, ...]:
        if self._closed:
            return ()
        return self._tasks

    @property
    def artifact_id(self) -> _BackendArtifactId:
        self._require_active()
        if self._artifact_id is None:
            raise _WorkframeClosedError
        return self._artifact_id

    def expected_bindings(self) -> tuple[tuple[_ContextBindingId, _TargetWorkId, int], ...]:
        self._require_active()
        return tuple((binding_id, owner, ordinal) for binding_id, owner, ordinal, _binding, _digest in self._expected)

    def target_work_ids(self) -> tuple[_TargetWorkId, ...]:
        self._require_active()
        return tuple(_TargetWorkId(value) for value in self._target_frame[COL_TARGET_WORK_ID])

    def bind_dispatches(self, dispatches: tuple[_Dispatch, ...]) -> None:
        """Attach the exact accepted phase-4 identities before backend invocation."""
        self._require_active()
        target_work_ids = self.target_work_ids()
        if (
            self._dispatches_bound
            or self._reconciled
            or len(dispatches) != len(self._tasks)
            or len(dispatches) != len(target_work_ids)
            or len({dispatch.attempt_id for dispatch in dispatches}) != len(dispatches)
            or any(
                dispatch.task != task or dispatch.row_token.value != work_id.value
                for dispatch, task, work_id in zip(dispatches, self._tasks, target_work_ids, strict=True)
            )
        ):
            raise _WorkframeStateError
        self._target_frame.loc[:, COL_ATTEMPT_ID] = [dispatch.attempt_id for dispatch in dispatches]
        self._dispatches_bound = True

    def reconcile(self, evidence: object) -> _ContextReconciliation:
        self._require_active()
        if self._reconciled:
            raise _WorkframeStateError
        self._reconciled = True
        if not self._dispatches_bound:
            return _ContextReconciliation(_ContextReconciliationStatus.GLOBAL_INVALID)
        if not isinstance(evidence, tuple) or not all(isinstance(item, _ContextBindingEvidence) for item in evidence):
            return _ContextReconciliation(_ContextReconciliationStatus.GLOBAL_INVALID)
        if not all(_valid_binding_evidence(item) for item in evidence):
            return _ContextReconciliation(_ContextReconciliationStatus.GLOBAL_INVALID)
        expected_by_id = {
            binding_id: (owner, ordinal, binding, payload_token)
            for binding_id, owner, ordinal, binding, payload_token in self._expected
        }
        observed_ids = tuple(item.binding_id for item in evidence)
        if len(expected_by_id) != len(self._expected) or any(
            binding_id not in expected_by_id for binding_id in observed_ids
        ):
            return _ContextReconciliation(_ContextReconciliationStatus.GLOBAL_INVALID)
        faults: dict[_TaskKey, _ContextBindingFault] = {}
        for item in evidence:
            owner, ordinal, binding, payload_token = expected_by_id[item.binding_id]
            if item.owner_target_work_id != owner:
                return _ContextReconciliation(_ContextReconciliationStatus.GLOBAL_INVALID)
            if type(item.ordinal) is not int:
                return _ContextReconciliation(_ContextReconciliationStatus.GLOBAL_INVALID)
            if item.ordinal != ordinal:
                faults[binding.owner_task] = _ContextBindingFault.CONTRADICTORY
            if item.payload_token != payload_token:
                faults[binding.owner_task] = _ContextBindingFault.CONTRADICTORY
        for binding_id, (_owner, _ordinal, binding, _payload_token) in expected_by_id.items():
            count = observed_ids.count(binding_id)
            if count == 0:
                faults.setdefault(binding.owner_task, _ContextBindingFault.MISSING)
            elif count > 1:
                faults.setdefault(binding.owner_task, _ContextBindingFault.DUPLICATE)
        if faults:
            ordered = tuple(task for task in self._tasks if task in faults)
            return _ContextReconciliation(
                _ContextReconciliationStatus.LOCAL_INVALID,
                ordered,
                tuple((task, faults[task]) for task in ordered),
            )
        return _ContextReconciliation(_ContextReconciliationStatus.VERIFIED)

    def close(self, attestations: object) -> _ContextCleanup:
        self._require_active()
        try:
            if not isinstance(attestations, tuple) or not all(
                isinstance(item, _BackendClosureAttestation) for item in attestations
            ):
                return _ContextCleanup(_ContextCleanupStatus.UNCONFIRMED)
            if len(attestations) != 1:
                return _ContextCleanup(_ContextCleanupStatus.UNCONFIRMED)
            attestation = attestations[0]
            if (
                attestation.artifact_id != self._artifact_id
                or self._required_artifacts != (attestation.artifact_class,)
                or attestation.schema_version is not self._schema_version
                or type(attestation.closed) is not bool
            ):
                return _ContextCleanup(_ContextCleanupStatus.UNCONFIRMED)
            if not self._dispatches_bound or not self._reconciled:
                return _ContextCleanup(_ContextCleanupStatus.UNCONFIRMED)
            status = _ContextCleanupStatus.VERIFIED if attestation.closed else _ContextCleanupStatus.FAILED
            return _ContextCleanup(status)
        finally:
            self._discard_owned_state()

    def discard_before_dispatch(self) -> None:
        """Close only owned frames after dispatch could not commit a backend handoff."""
        self._require_active()
        if self._reconciled:
            raise _WorkframeStateError
        self._discard_owned_state()

    def contain_discard_failure(self) -> None:
        """Make owned state inaccessible after a failed pre-dispatch discard."""
        self._closed = True
        self._target_frame = self._target_frame.iloc[0:0].copy()
        self._context_frame = self._context_frame.iloc[0:0].copy()
        self._tasks = ()
        self._expected = ()
        self._artifact_id = None
        self._required_artifacts = ()

    def _discard_owned_state(self) -> None:
        self._target_frame = self._target_frame.iloc[0:0].copy()
        self._context_frame = self._context_frame.iloc[0:0].copy()
        self._tasks = ()
        self._expected = ()
        self._artifact_id = None
        self._required_artifacts = ()
        self._closed = True

    def _require_active(self) -> None:
        if self._closed:
            raise _WorkframeClosedError


def _lower_context_workframes(
    plan: _ContextPlan,
    tasks: tuple[_TaskKey, ...],
    *,
    target_work_ids: tuple[str, ...] | None = None,
    identity_factory: Callable[[], str] = _default_identity,
) -> _ContextWorkframes:
    """Lower a ready target frontier from the sealed compiled snapshot only."""
    if not _is_admitted_context_plan(plan):
        raise _WorkframeConstructionError
    target_work_ids = _resolve_target_work_ids(tasks, target_work_ids, identity_factory)
    projection_by_task = {projection.owner_task: projection for projection in plan.projections}
    if len(set(tasks)) != len(tasks) or any(task not in projection_by_task for task in tasks):
        raise _WorkframeConstructionError
    target_ids = tuple(_TargetWorkId(value) for value in target_work_ids)
    target_text = {datum.id: datum.text for datum in plan.accounting.datums}
    context_text = {datum.id: datum.text for datum in (*plan.accounting.datums, *plan.context_only_datums)}
    expected, context_rows, artifact_value = _lower_binding_rows(
        tasks,
        target_ids,
        projection_by_task,
        context_text,
        used=set(target_work_ids),
        identity_factory=identity_factory,
    )
    target_frame = pd.DataFrame(
        (
            {
                COL_TARGET_WORK_ID: owner.value,
                COL_TASK_ID: task,
                COL_ATTEMPT_ID: None,
                COL_TEXT: target_text[task.datum_id],
            }
            for task, owner in zip(tasks, target_ids, strict=True)
        ),
        columns=pd.Index([COL_TARGET_WORK_ID, COL_TASK_ID, COL_ATTEMPT_ID, COL_TEXT]),
    )
    context_frame = pd.DataFrame(
        context_rows,
        columns=pd.Index([COL_CONTEXT_BINDING_ID, COL_CONTEXT_OWNER_WORK_ID, COL_CONTEXT_ORDINAL, COL_CONTEXT_TEXT]),
    )
    return _ContextWorkframes(
        target_frame=target_frame,
        context_frame=context_frame,
        tasks=tasks,
        expected=expected,
        artifact_id=_BackendArtifactId(artifact_value),
        required_artifacts=plan.contract.required_artifacts,
        schema_version=plan.contract.schema_version,
    )


def _resolve_target_work_ids(
    tasks: tuple[_TaskKey, ...],
    supplied: tuple[str, ...] | None,
    identity_factory: Callable[[], str],
) -> tuple[str, ...]:
    try:
        values = supplied if supplied is not None else tuple(identity_factory() for _task in tasks)
    except (StopIteration, TypeError):
        raise _WorkframeConstructionError from None
    if (
        len(tasks) != len(values)
        or len(set(values)) != len(values)
        or not all(isinstance(value, str) and value for value in values)
    ):
        raise _WorkframeConstructionError
    return values


def _lower_binding_rows(
    tasks: tuple[_TaskKey, ...],
    target_ids: tuple[_TargetWorkId, ...],
    projection_by_task: dict[_TaskKey, _CompiledContextProjection],
    context_text: dict[_DatumId, str],
    *,
    used: set[str],
    identity_factory: Callable[[], str],
) -> tuple[tuple[_ExpectedBinding, ...], list[dict[str, object]], str]:
    expected: list[_ExpectedBinding] = []
    rows: list[dict[str, object]] = []
    try:
        for task, owner in zip(tasks, target_ids, strict=True):
            for binding in projection_by_task[task].bindings:
                binding_id = _ContextBindingId(_claim_work_identity(identity_factory(), used))
                text = context_text[binding.datum_id]
                payload_token = _ContextPayloadToken(binding_id.value)
                payload = _ContextPayload(text, payload_token)
                expected.append((binding_id, owner, binding.ordinal, binding, payload_token))
                rows.append(
                    {
                        COL_CONTEXT_BINDING_ID: binding_id.value,
                        COL_CONTEXT_OWNER_WORK_ID: owner.value,
                        COL_CONTEXT_ORDINAL: binding.ordinal,
                        COL_CONTEXT_TEXT: payload,
                    }
                )
        artifact_value = _claim_work_identity(identity_factory(), used)
    except (KeyError, StopIteration, TypeError):
        raise _WorkframeConstructionError from None
    return tuple(expected), rows, artifact_value


def _claim_work_identity(value: object, used: set[str]) -> str:
    if not isinstance(value, str) or not value or value in used:
        raise _WorkframeConstructionError
    used.add(value)
    return value


def _make_context_binding_evidence(
    binding_id: str,
    owner_target_work_id: str,
    ordinal: int,
    text: object,
) -> _ContextBindingEvidence:
    """Create typed consumption evidence over the exact context-row payload."""
    if not isinstance(text, _ContextPayload):
        raise _WorkframeStateError
    private_binding_id = _ContextBindingId(binding_id)
    private_owner = _TargetWorkId(owner_target_work_id)
    return _ContextBindingEvidence(
        private_binding_id,
        private_owner,
        ordinal,
        text.token,
    )


def _valid_binding_evidence(value: _ContextBindingEvidence) -> bool:
    return (
        isinstance(value.binding_id, _ContextBindingId)
        and isinstance(value.binding_id.value, str)
        and bool(value.binding_id.value)
        and isinstance(value.owner_target_work_id, _TargetWorkId)
        and isinstance(value.owner_target_work_id.value, str)
        and bool(value.owner_target_work_id.value)
        and type(value.ordinal) is int
        and isinstance(value.payload_token, _ContextPayloadToken)
        and isinstance(value.payload_token.value, str)
        and bool(value.payload_token.value)
    )
