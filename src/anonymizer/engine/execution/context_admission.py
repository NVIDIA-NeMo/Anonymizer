# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure admission boundary for private target and bounded-context plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias, TypeGuard

from anonymizer.engine.execution.accounting_admission import (
    _AccountingAdmissionCode,
    _AccountingRejected,
    _compile_accounting_plan,
)
from anonymizer.engine.execution.accounting_plan import (
    _AccountingLimits,
    _AccountingPlan,
    _is_admitted_accounting_plan,
    _TaskKey,
)
from anonymizer.engine.execution.context_contract import (
    _BackendArtifactClass,
    _capability_satisfies,
    _ContextBackendCapability,
    _ContextExecutionContract,
    _ContextLimits,
    _ContextOrdering,
    _ContextProfile,
    _ContextSchemaVersion,
    _RetentionPosture,
    _valid_context_limits,
)
from anonymizer.engine.execution.graph import (
    _ContextScope,
    _DatumId,
    _DatumPurpose,
    _ProtectionGraph,
    _TextDatum,
)


class _PrivateContextPlanValue:
    def __repr__(self) -> str:
        return f"<private {type(self).__name__.strip('_').replace('_', ' ').lower()}>"

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private context plan values are not serializable")


class _ContextAdmissionCode(str, Enum):
    MALFORMED_GRAPH = "malformed_graph"
    INVALID_DATUM_PURPOSE = "invalid_datum_purpose"
    MISSING_CONTEXT_SCOPE = "missing_context_scope"
    DUPLICATE_CONTEXT_SCOPE = "duplicate_context_scope"
    UNKNOWN_CONTEXT_TARGET = "unknown_context_target"
    UNKNOWN_CONTEXT_DATUM = "unknown_context_datum"
    CONTEXT_ONLY_TARGET = "context_only_target"
    ORPHAN_CONTEXT_DATUM = "orphan_context_datum"
    SELF_CONTEXT = "self_context"
    DUPLICATE_CONTEXT_MEMBER = "duplicate_context_member"
    TARGET_CONTEXT_DISABLED = "target_context_disabled"
    CONTEXT_MEMBERS_EXCEEDED = "context_members_exceeded"
    CONTEXT_BYTES_EXCEEDED = "context_bytes_exceeded"
    TOTAL_CONTEXT_REFERENCES_EXCEEDED = "total_context_references_exceeded"
    EXPANDED_FRAME_BYTES_EXCEEDED = "expanded_frame_bytes_exceeded"
    UNSUPPORTED_CONTEXT_CONTRACT = "unsupported_context_contract"
    BACKEND_INCOMPATIBLE = "backend_incompatible"


_ContextRejectionCode: TypeAlias = _AccountingAdmissionCode | _ContextAdmissionCode


@dataclass(frozen=True, slots=True, repr=False)
class _ContextRejected(_PrivateContextPlanValue):
    code: _ContextRejectionCode


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class _ContextScopeKey(_PrivateContextPlanValue):
    """Compiler-issued graph-scoped identity for one admitted context scope."""


@dataclass(frozen=True, slots=True, repr=False)
class _CompiledContextBinding(_PrivateContextPlanValue):
    owner_task: _TaskKey
    scope: _ContextScopeKey
    ordinal: int
    datum_id: _DatumId


@dataclass(frozen=True, slots=True, repr=False)
class _CompiledContextProjection(_PrivateContextPlanValue):
    owner_task: _TaskKey
    target_datum_id: _DatumId
    scope: _ContextScopeKey
    context_datum_ids: tuple[_DatumId, ...]
    bindings: tuple[_CompiledContextBinding, ...]
    context_bytes: int


@dataclass(frozen=True, slots=True, repr=False)
class _ContextPlanProof(_PrivateContextPlanValue):
    seal: object = field(compare=False)
    snapshot: tuple[object, ...]


_CONTEXT_ADMISSION_SEAL = object()


@dataclass(frozen=True, slots=True, repr=False)
class _ContextPlan(_PrivateContextPlanValue):
    accounting: _AccountingPlan
    context_only_datums: tuple[_TextDatum, ...]
    projections: tuple[_CompiledContextProjection, ...]
    contract: _ContextExecutionContract
    preflight_capability: _ContextBackendCapability
    _proof: _ContextPlanProof | None = field(default=None, compare=False)


_ContextAdmissionResult: TypeAlias = _ContextPlan | _ContextRejected


def _compile_context_plan(
    graph: object,
    *,
    accounting_limits: _AccountingLimits,
    contract: object,
    capability: object,
    stages: tuple[str, ...] = ("protect",),
) -> _ContextAdmissionResult:
    """Compile one detached target/context projection before invocation effects."""
    rejected_or_datums = _compile_all_datums(graph, accounting_limits)
    if isinstance(rejected_or_datums, _ContextRejected):
        return rejected_or_datums
    if not isinstance(graph, _ProtectionGraph):
        return _ContextRejected(_ContextAdmissionCode.MALFORMED_GRAPH)
    datums = rejected_or_datums
    target_datums = tuple(datum for datum in datums if datum.purpose is _DatumPurpose.TARGET)
    context_datums = tuple(datum for datum in datums if datum.purpose is _DatumPurpose.CONTEXT_ONLY)
    if not target_datums:
        return _ContextRejected(_AccountingAdmissionCode.MALFORMED_GRAPH)
    target_ids = frozenset(datum.id for datum in target_datums)
    projected = _ProtectionGraph(
        datums=target_datums,
        links=graph.links,
        context_scopes=tuple(_ContextScope(datum.id) for datum in target_datums),
        coherence_scopes=graph.coherence_scopes,
        atomic_groups=graph.atomic_groups,
        dependencies=graph.dependencies,
    )
    accounting = _compile_accounting_plan(projected, limits=accounting_limits, stages=stages)
    if isinstance(accounting, _AccountingRejected):
        return _ContextRejected(accounting.code)
    if not _valid_contract(contract):
        return _ContextRejected(_ContextAdmissionCode.UNSUPPORTED_CONTEXT_CONTRACT)
    scopes_or_rejected = _compile_scopes(graph.context_scopes, datums, target_ids, context_datums, contract)
    if isinstance(scopes_or_rejected, _ContextRejected):
        return scopes_or_rejected
    scopes = scopes_or_rejected
    limit_rejection = _check_context_limits(scopes, datums, target_datums, contract.limits)
    if limit_rejection is not None:
        return limit_rejection
    if not isinstance(capability, _ContextBackendCapability) or not _capability_satisfies(capability, contract):
        return _ContextRejected(_ContextAdmissionCode.BACKEND_INCOMPATIBLE)
    projections = _materialize_projections(accounting, scopes, datums)
    return _admit_context_plan(accounting, context_datums, projections, contract, capability)


def _is_admitted_context_plan(value: object) -> bool:
    if not isinstance(value, _ContextPlan) or value._proof is None:
        return False
    return (
        value._proof.seal is _CONTEXT_ADMISSION_SEAL
        and _is_admitted_accounting_plan(value.accounting)
        and value._proof.snapshot == _context_plan_snapshot(value)
    )


def _compile_all_datums(
    graph: object,
    limits: _AccountingLimits,
) -> tuple[_TextDatum, ...] | _ContextRejected:
    if not isinstance(graph, _ProtectionGraph) or not isinstance(getattr(graph, "datums", None), tuple):
        return _ContextRejected(_ContextAdmissionCode.MALFORMED_GRAPH)
    if not graph.datums:
        return _ContextRejected(_AccountingAdmissionCode.MALFORMED_GRAPH)
    if len(graph.datums) > limits.max_datums:
        return _ContextRejected(_AccountingAdmissionCode.TOO_MANY_DATUMS)
    detached: list[_TextDatum] = []
    seen: set[str] = set()
    total_bytes = 0
    for candidate in graph.datums:
        if not isinstance(candidate, _TextDatum):
            return _ContextRejected(_ContextAdmissionCode.MALFORMED_GRAPH)
        datum_id = getattr(candidate, "id", None)
        value = getattr(datum_id, "value", None)
        text = getattr(candidate, "text", None)
        purpose = getattr(candidate, "purpose", None)
        if not isinstance(datum_id, _DatumId) or not isinstance(value, str) or not value or not isinstance(text, str):
            return _ContextRejected(_ContextAdmissionCode.MALFORMED_GRAPH)
        if purpose not in {_DatumPurpose.TARGET, _DatumPurpose.CONTEXT_ONLY}:
            return _ContextRejected(_ContextAdmissionCode.INVALID_DATUM_PURPOSE)
        try:
            id_bytes = len(value.encode("utf-8"))
            text_bytes = len(text.encode("utf-8"))
        except UnicodeEncodeError:
            return _ContextRejected(_ContextAdmissionCode.MALFORMED_GRAPH)
        if id_bytes > limits.max_id_bytes:
            return _ContextRejected(_AccountingAdmissionCode.MALFORMED_GRAPH)
        if text_bytes > limits.max_datum_bytes:
            return _ContextRejected(_AccountingAdmissionCode.DATUM_TOO_LARGE)
        if value in seen:
            return _ContextRejected(_AccountingAdmissionCode.DUPLICATE_DATUM_ID)
        seen.add(value)
        total_bytes += text_bytes
        detached.append(_TextDatum(_DatumId(value), text, purpose))
    if total_bytes > limits.max_graph_bytes:
        return _ContextRejected(_AccountingAdmissionCode.GRAPH_TOO_LARGE)
    return tuple(detached)


def _compile_scopes(
    source: object,
    datums: tuple[_TextDatum, ...],
    target_ids: frozenset[_DatumId],
    context_only_datums: tuple[_TextDatum, ...],
    contract: _ContextExecutionContract,
) -> tuple[tuple[_DatumId, tuple[_DatumId, ...]], ...] | _ContextRejected:
    if not isinstance(source, tuple) or not all(isinstance(scope, _ContextScope) for scope in source):
        return _ContextRejected(_ContextAdmissionCode.MALFORMED_GRAPH)
    by_value = {datum.id.value: datum.id for datum in datums}
    target_values = {datum_id.value for datum_id in target_ids}
    observed_targets: set[str] = set()
    referenced_context_only: set[str] = set()
    compiled: list[tuple[_DatumId, tuple[_DatumId, ...]]] = []
    for scope in source:
        scope_result = _compile_scope(scope, by_value, target_values, observed_targets, contract)
        if isinstance(scope_result, _ContextRejected):
            return scope_result
        target_id, member_ids, context_values = scope_result
        compiled.append((target_id, member_ids))
        referenced_context_only.update(context_values)
    if observed_targets != target_values:
        return _ContextRejected(_ContextAdmissionCode.MISSING_CONTEXT_SCOPE)
    if referenced_context_only != {datum.id.value for datum in context_only_datums}:
        return _ContextRejected(_ContextAdmissionCode.ORPHAN_CONTEXT_DATUM)
    position = {datum.id.value: index for index, datum in enumerate(datums)}
    compiled.sort(key=lambda item: position[item[0].value])
    return tuple(compiled)


def _compile_scope(
    scope: _ContextScope,
    by_value: dict[str, _DatumId],
    target_values: set[str],
    observed_targets: set[str],
    contract: _ContextExecutionContract,
) -> tuple[_DatumId, tuple[_DatumId, ...], set[str]] | _ContextRejected:
    target_value = getattr(getattr(scope, "target", None), "value", None)
    members = getattr(scope, "context", None)
    if not isinstance(target_value, str) or not isinstance(members, tuple):
        return _ContextRejected(_ContextAdmissionCode.MALFORMED_GRAPH)
    if target_value not in by_value:
        return _ContextRejected(_ContextAdmissionCode.UNKNOWN_CONTEXT_TARGET)
    if target_value not in target_values:
        return _ContextRejected(_ContextAdmissionCode.CONTEXT_ONLY_TARGET)
    if target_value in observed_targets:
        return _ContextRejected(_ContextAdmissionCode.DUPLICATE_CONTEXT_SCOPE)
    observed_targets.add(target_value)
    member_values = _compile_scope_members(members, target_value, by_value)
    if isinstance(member_values, _ContextRejected):
        return member_values
    if not contract.allow_target_as_context and any(value in target_values for value in member_values):
        return _ContextRejected(_ContextAdmissionCode.TARGET_CONTEXT_DISABLED)
    context_values = {value for value in member_values if value not in target_values}
    return by_value[target_value], tuple(by_value[value] for value in member_values), context_values


def _compile_scope_members(
    members: tuple[object, ...],
    target_value: str,
    by_value: dict[str, _DatumId],
) -> tuple[str, ...] | _ContextRejected:
    values: list[str] = []
    for member in members:
        value = getattr(member, "value", None)
        if not isinstance(member, _DatumId) or not isinstance(value, str):
            return _ContextRejected(_ContextAdmissionCode.MALFORMED_GRAPH)
        if value not in by_value:
            return _ContextRejected(_ContextAdmissionCode.UNKNOWN_CONTEXT_DATUM)
        if value == target_value:
            return _ContextRejected(_ContextAdmissionCode.SELF_CONTEXT)
        values.append(value)
    if len(set(values)) != len(values):
        return _ContextRejected(_ContextAdmissionCode.DUPLICATE_CONTEXT_MEMBER)
    return tuple(values)


def _check_context_limits(
    scopes: tuple[tuple[_DatumId, tuple[_DatumId, ...]], ...],
    datums: tuple[_TextDatum, ...],
    target_datums: tuple[_TextDatum, ...],
    limits: _ContextLimits,
) -> _ContextRejected | None:
    if not _valid_context_limits(limits):
        return _ContextRejected(_ContextAdmissionCode.UNSUPPORTED_CONTEXT_CONTRACT)
    text_by_id = {datum.id: datum.text for datum in datums}
    total_references = 0
    expanded_bytes = sum(len(datum.text.encode("utf-8")) for datum in target_datums)
    for _target_id, members in scopes:
        if len(members) > limits.max_context_members_per_target:
            return _ContextRejected(_ContextAdmissionCode.CONTEXT_MEMBERS_EXCEEDED)
        context_bytes = sum(len(text_by_id[member].encode("utf-8")) for member in members)
        if context_bytes > limits.max_context_bytes_per_target:
            return _ContextRejected(_ContextAdmissionCode.CONTEXT_BYTES_EXCEEDED)
        total_references += len(members)
        expanded_bytes += context_bytes
    if total_references > limits.max_total_context_references:
        return _ContextRejected(_ContextAdmissionCode.TOTAL_CONTEXT_REFERENCES_EXCEEDED)
    if expanded_bytes > limits.max_expanded_frame_bytes:
        return _ContextRejected(_ContextAdmissionCode.EXPANDED_FRAME_BYTES_EXCEEDED)
    return None


def _valid_contract(contract: object) -> TypeGuard[_ContextExecutionContract]:
    return (
        isinstance(contract, _ContextExecutionContract)
        and contract.profile is _ContextProfile.TARGET_CONTEXT_V1
        and contract.schema_version is _ContextSchemaVersion.V1
        and _valid_context_limits(contract.limits)
        and type(contract.allow_target_as_context) is bool
        and contract.ordering is _ContextOrdering.DECLARED
        and contract.retention is _RetentionPosture.DISABLED
        and isinstance(contract.required_artifacts, tuple)
        and contract.required_artifacts == (_BackendArtifactClass.CONTEXT_REQUEST,)
    )


def _materialize_projections(
    accounting: _AccountingPlan,
    scopes: tuple[tuple[_DatumId, tuple[_DatumId, ...]], ...],
    datums: tuple[_TextDatum, ...],
) -> tuple[_CompiledContextProjection, ...]:
    member_by_target = dict(scopes)
    text_by_id = {datum.id: datum.text for datum in datums}
    final_stage = accounting.stages[-1]
    projections: list[_CompiledContextProjection] = []
    for target in accounting.datums:
        task = _TaskKey(final_stage, target.id)
        scope = _ContextScopeKey()
        members = member_by_target[target.id]
        bindings = tuple(
            _CompiledContextBinding(task, scope, ordinal, datum_id) for ordinal, datum_id in enumerate(members)
        )
        projections.append(
            _CompiledContextProjection(
                task,
                target.id,
                scope,
                members,
                bindings,
                sum(len(text_by_id[member].encode("utf-8")) for member in members),
            )
        )
    return tuple(projections)


def _admit_context_plan(
    accounting: _AccountingPlan,
    context_only_datums: tuple[_TextDatum, ...],
    projections: tuple[_CompiledContextProjection, ...],
    contract: _ContextExecutionContract,
    capability: _ContextBackendCapability,
) -> _ContextPlan:
    values = (accounting, context_only_datums, projections, contract, capability)
    candidate = _ContextPlan(*values)
    snapshot = _context_plan_snapshot(candidate)
    if snapshot is None:
        raise TypeError("private context plan admission failed")
    return _ContextPlan(*values, _ContextPlanProof(_CONTEXT_ADMISSION_SEAL, snapshot))


def _context_plan_snapshot(plan: _ContextPlan) -> tuple[object, ...] | None:
    try:
        return (
            plan.accounting._proof,
            tuple((datum.id.value, datum.text, datum.purpose.value) for datum in plan.context_only_datums),
            _projection_snapshot(plan.projections),
            _contract_snapshot(plan.contract),
            _capability_snapshot(plan.preflight_capability),
        )
    except (AttributeError, TypeError):
        return None


def _projection_snapshot(projections: tuple[_CompiledContextProjection, ...]) -> tuple[object, ...]:
    return tuple(
        (
            projection.owner_task.stage.value,
            projection.target_datum_id.value,
            projection.scope,
            tuple(member.value for member in projection.context_datum_ids),
            tuple(
                (
                    binding.owner_task.stage.value,
                    binding.owner_task.datum_id.value,
                    binding.scope,
                    binding.ordinal,
                    binding.datum_id.value,
                )
                for binding in projection.bindings
            ),
            projection.context_bytes,
        )
        for projection in projections
    )


def _contract_snapshot(contract: _ContextExecutionContract) -> tuple[object, ...]:
    limits = contract.limits
    return (
        contract.profile.value,
        contract.schema_version.value,
        limits.max_context_members_per_target,
        limits.max_context_bytes_per_target,
        limits.max_total_context_references,
        limits.max_expanded_frame_bytes,
        contract.allow_target_as_context,
        contract.ordering.value,
        tuple(value.value for value in contract.required_artifacts),
        contract.retention.value,
    )


def _capability_snapshot(capability: _ContextBackendCapability) -> tuple[object, ...]:
    limits = capability.limits
    return (
        capability.profile.value,
        capability.schema_version.value,
        limits.max_context_members_per_target,
        limits.max_context_bytes_per_target,
        limits.max_total_context_references,
        limits.max_expanded_frame_bytes,
        capability.allow_target_as_context,
        capability.ordering.value,
        tuple(value.value for value in capability.artifact_classes),
        capability.retention.value,
    )
