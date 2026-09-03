# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Redact verification and compatibility projection over terminal accounting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, assert_never

import pandas as pd

from anonymizer.engine.constants import COL_FINAL_ENTITIES, COL_REPLACED_TEXT, COL_REPLACEMENT_APPLICATION
from anonymizer.engine.execution.accounting_admission import (
    _AccountingAdmissionResult,
    _compile_accounting_plan,
)
from anonymizer.engine.execution.accounting_outcomes import (
    _CauseCode,
    _DatumBlocked,
    _DatumCancelled,
    _DatumFailed,
    _DatumInconsistent,
    _DatumLost,
    _DatumOutcome,
    _DatumQualified,
    _GroupReleased,
    _InvocationCompleted,
)
from anonymizer.engine.execution.accounting_plan import _AccountingLimits, _AccountingPlan
from anonymizer.engine.execution.context_admission import (
    _compile_context_plan,
    _ContextAdmissionResult,
    _ContextPlan,
)
from anonymizer.engine.execution.context_contract import _ContextExecutionContract, _snapshot_context_capability
from anonymizer.engine.execution.context_observations import _observe_context_boundary
from anonymizer.engine.execution.graph import _DatumId, _DatumPurpose, _TextDatum
from anonymizer.engine.execution.graph_runtime import _AccountingGraphExecution
from anonymizer.engine.execution.invocation import _CompiledInvocation
from anonymizer.engine.execution.mention_admission import _MentionLimits
from anonymizer.engine.execution.phase6_plan import (
    _compile_phase6_plan,
    _Phase6Plan,
    _Phase6PlanRejectionCode,
    _Phase6ProfileVersion,
    _Phase6Rejected,
)
from anonymizer.engine.execution.phase6_runtime import (
    _Phase6Candidate,
    _Phase6EffectBackend,
    _Phase6Execution,
    _Phase6Runtime,
)
from anonymizer.engine.execution.phase7_admission import (
    _compile_phase7_plan,
    _Phase7Declarations,
    _Phase7Plan,
)
from anonymizer.engine.execution.phase7_contract import _Phase7StableSubstituteContract
from anonymizer.engine.execution.phase7_runtime import (
    _Phase7EffectBackend,
    _Phase7Execution,
    _Phase7Runtime,
)
from anonymizer.engine.execution.redact_patches import _VerifiedDatum


class _PrivateProtectionValue:
    def __repr__(self) -> str:
        return f"<private {type(self).__name__.strip('_').replace('_', ' ').lower()}>"

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private protection results are not serializable")


@dataclass(frozen=True, slots=True, repr=False)
class _RedactCandidate(_PrivateProtectionValue):
    output: str
    applied: bool
    release_qualified: bool


@dataclass(frozen=True, slots=True, repr=False)
class _GraphProtectionSucceeded(_PrivateProtectionValue):
    datum_id: _DatumId
    output: str
    applied: bool


@dataclass(frozen=True, slots=True, repr=False)
class _GraphProtectionFailed(_PrivateProtectionValue):
    datum_id: _DatumId
    stage: str
    scope: str


_GraphProtectionOutcome = _GraphProtectionSucceeded | _GraphProtectionFailed


@dataclass(frozen=True, slots=True, repr=False)
class _GraphProtectionResult(_PrivateProtectionValue):
    outcomes: tuple[_GraphProtectionOutcome, ...]


class _GraphRuntimeBackend(Protocol):
    def run(
        self,
        plan: _AccountingPlan | _ContextPlan,
        *,
        invocation: _CompiledInvocation,
        data_summary: str | None,
        preview_num_records: int | None,
        hydrate: Callable[[_TextDatum, pd.Series], _RedactCandidate],
        datum_release_predicate: Callable[[_DatumId, _RedactCandidate], bool],
    ) -> _AccountingGraphExecution[_RedactCandidate]: ...


class _RedactProtectionService:
    """Protect text datums and project accounting into the compatibility result."""

    def __init__(self, runtime: _GraphRuntimeBackend) -> None:
        self._runtime = runtime

    @staticmethod
    def admit(
        graph: object,
        *,
        limits: _AccountingLimits,
    ) -> _AccountingAdmissionResult:
        """Compile the complete graph before any execution context opens."""
        return _compile_accounting_plan(graph, limits=limits)

    def admit_context(
        self,
        graph: object,
        *,
        accounting_limits: _AccountingLimits,
        contract: _ContextExecutionContract,
    ) -> _ContextAdmissionResult:
        """Compile context framing against the selected backend's preflight snapshot."""
        target_count, context_count = _preflight_observation_counts(graph)
        with _observe_context_boundary(
            "preflight",
            target_count=target_count,
            context_count=context_count,
        ) as observation:
            capability = _snapshot_context_capability(self._runtime)
            result = _compile_context_plan(
                graph,
                accounting_limits=accounting_limits,
                contract=contract,
                capability=capability,
            )
            if isinstance(result, _ContextPlan):
                observation.outcome = "admitted"
            else:
                observation.outcome = "rejected"
                observation.reason = result.code.value
            return result

    def protect(
        self,
        plan: _AccountingPlan | _ContextPlan,
        *,
        invocation: _CompiledInvocation,
    ) -> _GraphProtectionResult:
        execution = self._runtime.run(
            plan,
            invocation=invocation,
            data_summary=None,
            preview_num_records=None,
            hydrate=_hydrate_redact_candidate,
            datum_release_predicate=lambda _datum_id, candidate: candidate.release_qualified,
        )
        return _materialize(execution)


class _Phase6RedactProtectionService:
    """Compile and execute the selected private Phase 6 Redact profile."""

    def __init__(self, backend: _Phase6EffectBackend, *, mention_limits: _MentionLimits) -> None:
        self._backend = backend
        self._mention_limits = mention_limits

    def admit_context(
        self,
        graph: object,
        *,
        accounting_limits: _AccountingLimits,
        contract: _ContextExecutionContract,
    ) -> _Phase6Plan | _Phase6Rejected:
        target_count, context_count = _preflight_observation_counts(graph)
        with _observe_context_boundary(
            "preflight",
            target_count=target_count,
            context_count=context_count,
        ) as observation:
            capability = _snapshot_context_capability(self._backend)
            if capability is None:
                result: _Phase6Plan | _Phase6Rejected = _Phase6Rejected(_Phase6PlanRejectionCode.INVALID_PROFILE)
            else:
                result = _compile_phase6_plan(
                    graph,
                    accounting_limits=accounting_limits,
                    context_contract=contract,
                    capability=capability,
                    mention_limits=self._mention_limits,
                )
            if isinstance(result, _Phase6Plan):
                observation.outcome = "admitted"
            else:
                observation.outcome = "rejected"
                observation.reason = result.code.value
            return result

    def protect(
        self,
        plan: _Phase6Plan,
        *,
        invocation: _CompiledInvocation,
    ) -> _GraphProtectionResult:
        del invocation
        return _materialize_phase6(plan, _Phase6Runtime(self._backend).run(plan))


class _Phase7SubstituteProtectionService:
    """Compile, plan, apply, and qualify the private stable-Substitute profile."""

    def __init__(
        self,
        phase6_backend: _Phase6EffectBackend,
        phase7_backend_factory: Callable[[], _Phase7EffectBackend],
        *,
        mention_limits: _MentionLimits,
    ) -> None:
        self._phase6_backend = phase6_backend
        self._phase7_backend_factory = phase7_backend_factory
        self._mention_limits = mention_limits

    def admit_context(
        self,
        graph: object,
        *,
        accounting_limits: _AccountingLimits,
        contract: _ContextExecutionContract,
    ) -> _Phase6Plan | _Phase6Rejected:
        target_count, context_count = _preflight_observation_counts(graph)
        with _observe_context_boundary(
            "preflight",
            target_count=target_count,
            context_count=context_count,
        ) as observation:
            capability = _snapshot_context_capability(self._phase6_backend)
            if capability is None:
                result: _Phase6Plan | _Phase6Rejected = _Phase6Rejected(_Phase6PlanRejectionCode.INVALID_PROFILE)
            else:
                result = _compile_phase6_plan(
                    graph,
                    accounting_limits=accounting_limits,
                    context_contract=contract,
                    capability=capability,
                    mention_limits=self._mention_limits,
                    profile_version=_Phase6ProfileVersion.SUBSTITUTE_V1,
                )
            if isinstance(result, _Phase6Plan):
                observation.outcome = "admitted"
            else:
                observation.outcome = "rejected"
                observation.reason = result.code.value
            return result

    def protect(
        self,
        plan: _Phase6Plan,
        *,
        contract: _Phase7StableSubstituteContract,
    ) -> _GraphProtectionResult:
        execution = self.execute(plan, contract=contract)
        return (
            _materialize_phase7(plan, execution)
            if isinstance(execution, _Phase7Execution)
            else _fail_all(tuple(datum.id for datum in plan.accounting.datums))
        )

    def execute(
        self,
        plan: _Phase6Plan,
        *,
        contract: _Phase7StableSubstituteContract,
    ) -> _Phase7Execution | None:
        """Return the released Phase 7 handoff for a private successor only."""
        phase6 = _Phase6Runtime(self._phase6_backend).run(plan)
        phase7 = _compile_phase7_plan(
            plan,
            phase6.handoffs,
            _Phase7Declarations(plan.coherence_scopes),
            contract,
        )
        if not isinstance(phase7, _Phase7Plan):
            return None
        return _Phase7Runtime(self._phase7_backend_factory()).run(plan, phase6, phase7, contract)


def _preflight_observation_counts(graph: object) -> tuple[int, int]:
    """Best-effort total counts for telemetry, kept outside admission semantics."""
    try:
        datums = getattr(graph, "datums", ())
        scopes = getattr(graph, "context_scopes", ())
    except BaseException:
        return 0, 0
    target_count = 0
    context_count = 0
    if isinstance(datums, tuple):
        for datum in datums:
            try:
                target_count += getattr(datum, "purpose", None) is _DatumPurpose.TARGET
            except BaseException:
                continue
    if isinstance(scopes, tuple):
        for scope in scopes:
            try:
                context = getattr(scope, "context", ())
            except BaseException:
                continue
            if isinstance(context, tuple):
                context_count += len(context)
    return target_count, context_count


def _hydrate_redact_candidate(datum: _TextDatum, row: pd.Series) -> _RedactCandidate:
    output = row[COL_REPLACED_TEXT]
    if not isinstance(output, str):
        raise TypeError("private redact result is malformed")
    valid_entities, has_detections = _accepted_detection_state(row[COL_FINAL_ENTITIES])
    release_qualified = (
        valid_entities
        and _redact_release_passed(
            row[COL_FINAL_ENTITIES],
            row[COL_REPLACEMENT_APPLICATION],
            datum.text,
            output,
        )
        and (has_detections or output == datum.text)
    )
    return _RedactCandidate(output, has_detections, release_qualified)


def _materialize(execution: _AccountingGraphExecution[_RedactCandidate]) -> _GraphProtectionResult:
    if not isinstance(execution.accounting.invocation, _InvocationCompleted):
        return _fail_all(tuple(datum.id for datum in execution.plan.datums))
    released = {
        datum_id: candidate
        for group in execution.accounting.groups
        if isinstance(group, _GroupReleased)
        for datum_id, candidate in group.outputs
    }
    datum_outcomes = {outcome.datum_id: outcome for outcome in execution.accounting.datums}
    return _GraphProtectionResult(
        tuple(
            _materialize_datum(datum.id, released.get(datum.id), datum_outcomes[datum.id])
            for datum in execution.plan.datums
        )
    )


def _materialize_phase6(plan: _Phase6Plan, execution: _Phase6Execution) -> _GraphProtectionResult:
    datum_ids = tuple(datum.id for datum in plan.accounting.datums)
    if not isinstance(execution.accounting.invocation, _InvocationCompleted):
        return _fail_all(datum_ids)
    released = {datum.datum_id: datum for datum in execution.released}
    datum_outcomes = {outcome.datum_id: outcome for outcome in execution.accounting.datums}
    return _GraphProtectionResult(
        tuple(
            _materialize_phase6_datum(datum_id, released.get(datum_id), datum_outcomes[datum_id])
            for datum_id in datum_ids
        )
    )


def _materialize_phase7(plan: _Phase6Plan, execution: _Phase7Execution) -> _GraphProtectionResult:
    datum_ids = tuple(datum.id for datum in plan.accounting.datums)
    if not isinstance(execution.phase4.accounting.invocation, _InvocationCompleted) or execution.phase4.global_embargo:
        return _fail_all(datum_ids)
    released = {datum.datum_id: datum for datum in execution.released}
    if len(released) != len(execution.released):
        return _fail_all(datum_ids)
    return _GraphProtectionResult(
        tuple(
            _GraphProtectionSucceeded(datum_id, released[datum_id].output, released[datum_id].applied)
            if datum_id in released
            else _GraphProtectionFailed(datum_id, "planning", "datum")
            for datum_id in datum_ids
        )
    )


def _materialize_phase6_datum(
    datum_id: _DatumId,
    candidate: _VerifiedDatum | None,
    outcome: _DatumOutcome[_Phase6Candidate],
) -> _GraphProtectionOutcome:
    if candidate is not None:
        return _GraphProtectionSucceeded(datum_id, candidate.output, candidate.applied)
    match outcome:
        case _DatumFailed(causes=causes) if any(cause.code is _CauseCode.RELEASE_PREDICATE_FAILED for cause in causes):
            return _GraphProtectionFailed(datum_id, "release", "datum")
        case _DatumQualified():
            return _GraphProtectionFailed(datum_id, "release", "group")
        case _DatumFailed() | _DatumBlocked() | _DatumCancelled() | _DatumLost() | _DatumInconsistent():
            return _GraphProtectionFailed(datum_id, "pipeline", "datum")
        case unreachable:
            assert_never(unreachable)


def _materialize_datum(
    datum_id: _DatumId,
    candidate: _RedactCandidate | None,
    outcome: _DatumOutcome[_RedactCandidate],
) -> _GraphProtectionOutcome:
    if candidate is not None:
        return _GraphProtectionSucceeded(datum_id, candidate.output, candidate.applied)
    match outcome:
        case _DatumFailed(causes=causes) if any(cause.code is _CauseCode.RELEASE_PREDICATE_FAILED for cause in causes):
            return _GraphProtectionFailed(datum_id, "release", "datum")
        case _DatumQualified():
            return _GraphProtectionFailed(datum_id, "release", "group")
        case _DatumFailed() | _DatumBlocked() | _DatumCancelled() | _DatumLost() | _DatumInconsistent():
            return _GraphProtectionFailed(datum_id, "pipeline", "datum")
        case unreachable:
            assert_never(unreachable)


def _fail_all(datum_ids: tuple[_DatumId, ...]) -> _GraphProtectionResult:
    return _GraphProtectionResult(
        tuple(_GraphProtectionFailed(datum_id, "pipeline", "invocation") for datum_id in datum_ids)
    )


def _accepted_detection_state(value: object) -> tuple[bool, bool]:
    if isinstance(value, dict):
        if "entities" not in value:
            return False, False
        entities = value["entities"]
    else:
        entities = getattr(value, "entities", None)
    if not isinstance(entities, (list, tuple)):
        return False, False
    return True, bool(entities)


def _redact_release_passed(value: object, application: object, input_text: str, output: str) -> bool:
    """Require complete Redact accounting and removal of authoritative source spans."""
    if isinstance(value, dict):
        entities = value.get("entities", [])
    else:
        entities = getattr(value, "entities", [])
    if not isinstance(entities, (list, tuple)) or not _replacement_application_passed(application, len(entities)):
        return False
    spans: list[tuple[int, int, str]] = []
    for entity in entities:
        raw = entity.get("value") if isinstance(entity, dict) else getattr(entity, "value", None)
        label = entity.get("label") if isinstance(entity, dict) else getattr(entity, "label", None)
        start = entity.get("start_position") if isinstance(entity, dict) else getattr(entity, "start_position", None)
        end = entity.get("end_position") if isinstance(entity, dict) else getattr(entity, "end_position", None)
        if (
            not isinstance(raw, str)
            or not isinstance(label, str)
            or not label
            or type(start) is not int
            or type(end) is not int
            or start < 0
            or end <= start
            or end > len(input_text)
            or input_text[start:end] != raw
        ):
            return False
        spans.append((start, end, input_text[start:end]))
    previous_end = 0
    for start, end, source_slice in sorted(spans):
        if start < previous_end or source_slice in output:
            return False
        previous_end = end
    return True


def _replacement_application_passed(value: object, entity_count: int) -> bool:
    if not isinstance(value, dict):
        return False
    expected_keys = {
        "targeted_span_count",
        "applied_span_count",
        "skipped_span_count",
        "skipped_span_label_counts",
    }
    if set(value) != expected_keys:
        return False
    targeted = value["targeted_span_count"]
    applied = value["applied_span_count"]
    skipped = value["skipped_span_count"]
    skipped_by_label = value["skipped_span_label_counts"]
    if (
        type(targeted) is not int
        or type(applied) is not int
        or type(skipped) is not int
        or targeted < 0
        or applied < 0
        or skipped < 0
        or not isinstance(skipped_by_label, dict)
    ):
        return False
    if any(
        not isinstance(label, str) or not label or type(count) is not int or count <= 0
        for label, count in skipped_by_label.items()
    ):
        return False
    return targeted == entity_count and applied == targeted and skipped == 0 and not skipped_by_label
