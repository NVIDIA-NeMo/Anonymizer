# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private-only composition seam for the Phase 8 grouped profile."""

from __future__ import annotations

import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import cast

from anonymizer.engine.execution.accounting_plan import (
    _AccountingPlan,
    _admit_accounting_plan,
    _AtomicGroupKey,
    _CompiledAtomicGroup,
    _CompiledDependency,
)
from anonymizer.engine.execution.accounting_release import _qualify_release
from anonymizer.engine.execution.graph import _DatumId, _DatumPurpose, _TextDatum
from anonymizer.engine.execution.phase7_application import _AppliedDatum
from anonymizer.engine.execution.phase7_runtime import _Phase7Execution
from anonymizer.engine.execution.phase8_admission import (
    _compile_group_operation_plan,
    _compile_phase8_plan,
    _Phase8GroupOperationPlan,
    _Phase8Plan,
)
from anonymizer.engine.execution.phase8_contract import _load_phase8_contract
from anonymizer.engine.execution.phase8_ndd_backend import _Phase8Operation
from anonymizer.engine.execution.phase8_runtime import (
    _GroupFailed,
    _GroupInconsistent,
    _GroupLost,
    _GroupSucceeded,
    _Phase8FaultKind,
    _Phase8GroupOutcome,
    _Phase8InvocationLedger,
    _Phase8LifecycleExecution,
    _Phase8OperationFault,
    _Phase8Reason,
    _run_group_operation,
)
from anonymizer.engine.execution.phase8_successor import _is_admitted_phase8_successor, _Phase8SuccessorHandoff
from anonymizer.engine.execution.phase8_validation import (
    _evaluate_metrics,
    _Phase8Metric,
    _validate_complete_revisions,
)


class _Phase8GroupedRewriteProtectionService:
    """Deliberately not wired into the public Rewrite selector."""

    def run_group(
        self,
        members: tuple[object, ...],
        baselines: dict[object, str],
        *,
        analyze: Callable[[], tuple[bool, bool]],
        rewrite: Callable[[dict[object, str]], dict[object, str]],
        evaluate: Callable[[dict[object, str]], _Phase8Metric],
        repair: Callable[[dict[object, str], int], dict[object, str]],
        max_repairs: int,
    ) -> tuple[tuple[object, str], ...] | None:
        """Return sealed keyed candidates only after a whole group succeeds."""
        outcome = _run_group_operation(
            members,
            baselines,
            analyze=analyze,
            rewrite=rewrite,
            evaluate=evaluate,
            repair=repair,
            max_repairs=max_repairs,
        )
        if outcome.state != "succeeded" or outcome.revisions is None:
            return None
        return tuple((member, outcome.revisions[member]) for member in members)

    def run_lifecycle(
        self,
        *,
        groups: tuple[tuple[object, ...], ...],
        atomic_groups: tuple[tuple[object, ...], ...],
        dependencies: tuple[tuple[object, object], ...],
        phase7_released: tuple[tuple[object, str], ...],
        phase7_cleanup_verified: bool,
        phase7_global_embargo: bool,
        operations: tuple[_GroupOperation, ...],
    ) -> _Phase8LifecycleExecution:
        """Consume only a clean released Phase 7 handoff and reduce via Phase 4.

        This is deliberately a private graph-service seam.  It has no row or
        public result representation: each operation receives the entire
        declared group and its exact released baselines, then Phase 4 withholds
        atomic/dependent cells before any candidate is returned.
        """
        early = _lifecycle_preflight(
            groups,
            atomic_groups,
            dependencies,
            operations,
            phase7_released,
            phase7_cleanup_verified,
            phase7_global_embargo,
        )
        if isinstance(early, _Phase8LifecycleExecution):
            return early
        baselines = early
        candidates, states, invocation_inconsistent = _run_operations(groups, operations, baselines)
        if invocation_inconsistent:
            candidates.clear()
            baselines.clear()
            return _terminal((), tuple(states), True, False)

        qualified = _phase4_released(groups, atomic_groups, dependencies, states)
        released = tuple(
            (member, candidates[member]) for members in groups for member in members if member in qualified
        )
        # First cleanup attestation: no candidate-bearing ledger or baseline
        # index survives the Phase 4 reduction.  Clear before constructing the
        # terminal result, then retain only the copied release cells.
        candidates.clear()
        baselines.clear()
        cleanup_verified = not candidates and not baselines
        if not cleanup_verified:
            return _terminal((), tuple(states), True, False)
        # Second attestation is represented by construction: ``released`` is
        # the only value copied past the reduction and every member is unique.
        if len({member for member, _ in released}) != len(released):
            return _terminal((), tuple(states), True, False)
        return _terminal(released, tuple(states), False, True)

    def run_from_phase7_execution(
        self,
        *,
        groups: tuple[tuple[object, ...], ...],
        atomic_groups: tuple[tuple[object, ...], ...],
        dependencies: tuple[tuple[object, object], ...],
        phase7: _Phase7Execution,
        operations: tuple[_GroupOperation, ...],
    ) -> _Phase8LifecycleExecution:
        """Import exactly the Phase 7 release-qualified baseline handoff.

        This adapter intentionally accepts no provisional Phase 7 material.
        A malformed released cell, unverified Phase 7 cleanup, or Phase 7
        embargo is terminal before a Phase 8 operation can be called.
        """
        released = phase7.released
        if not all(isinstance(value, _AppliedDatum) for value in released):
            return _terminal((), tuple("inconsistent" for _ in groups), True, False)
        return self.run_lifecycle(
            groups=groups,
            atomic_groups=atomic_groups,
            dependencies=dependencies,
            phase7_released=tuple((value.datum_id, value.output) for value in released),
            phase7_cleanup_verified=phase7.cleanup.verified,
            phase7_global_embargo=phase7.phase4.global_embargo,
            # Every baseline-ready private group is analyzed.  A baseline-only
            # shortcut cannot establish the frozen analysis-derived zero route.
            operations=operations,
        )

    def run_from_phase7_execution_with_backend(
        self, graph: object, phase7: _Phase7Execution, backend: object, invocation: object | None = None
    ) -> _Phase8LifecycleExecution:
        """Execute the admitted complete groups through the sole NDD boundary.

        Requests use fresh opaque wire tokens and are reconciled back to the
        compiler-issued datum keys before any candidate reaches Phase 4.
        """
        max_repairs = _phase8_max_repairs(invocation)
        plan = _compile_phase8_plan(graph, max_repairs=max_repairs)
        if not isinstance(plan, _Phase8Plan):
            return _terminal((), (), True, False)
        groups = tuple(manifest.members for manifest in plan.groups)
        atomic_groups = tuple(getattr(group, "members", ()) for group in getattr(graph, "atomic_groups", ()))
        dependencies = tuple(
            (getattr(edge, "prerequisite", None), getattr(edge, "dependent", None))
            for edge in getattr(graph, "dependencies", ())
        )
        original_by_datum = {
            getattr(datum, "id"): getattr(datum, "text")
            for datum in getattr(graph, "datums", ())
            if isinstance(getattr(datum, "text", None), str)
        }
        applied_by_datum = {value.datum_id: value.applied for value in phase7.released}
        registry = _Phase8WireRegistry()
        operations = tuple(
            _backend_group_operation(
                backend,
                _phase8_group_input(
                    members, original_by_datum, cast(Mapping[object, bool], applied_by_datum), invocation
                ),
                registry,
                operation_plan=manifest.operations,
            )
            for manifest, members in zip(plan.groups, groups, strict=True)
        )
        return self.run_lifecycle(
            groups=groups,
            atomic_groups=atomic_groups,
            dependencies=dependencies,
            phase7_released=tuple((value.datum_id, value.output) for value in phase7.released),
            phase7_cleanup_verified=phase7.cleanup.verified,
            phase7_global_embargo=phase7.phase4.global_embargo,
            operations=operations,
        )

    def run_from_phase7_successor_with_backend(
        self,
        graph: object,
        predecessor: _Phase8SuccessorHandoff,
        backend: object,
        invocation: object | None = None,
    ) -> _Phase8LifecycleExecution:
        """Consume only the sealed predecessor held by the Phase 7 owner."""
        if not _is_admitted_phase8_successor(predecessor):
            return _terminal((), (), True, False)
        phase7 = predecessor.phase7_execution
        max_repairs = _phase8_max_repairs(invocation)
        plan = _compile_phase8_plan(graph, max_repairs=max_repairs)
        if not isinstance(plan, _Phase8Plan):
            return _terminal((), (), True, False)
        groups = tuple(manifest.members for manifest in plan.groups)
        atomic_groups = tuple(getattr(group, "members", ()) for group in getattr(graph, "atomic_groups", ()))
        dependencies = tuple(
            (getattr(edge, "prerequisite", None), getattr(edge, "dependent", None))
            for edge in getattr(graph, "dependencies", ())
        )
        registry = _Phase8WireRegistry()
        operations = tuple(
            _backend_group_operation(
                backend,
                _phase8_group_input_from_successor(members, predecessor, invocation),
                registry,
                operation_plan=manifest.operations,
            )
            for manifest, members in zip(plan.groups, groups, strict=True)
        )
        return self.run_lifecycle(
            groups=groups,
            atomic_groups=atomic_groups,
            dependencies=dependencies,
            phase7_released=tuple((value.datum_id, value.output) for value in phase7.released),
            phase7_cleanup_verified=phase7.cleanup.verified,
            phase7_global_embargo=phase7.phase4.global_embargo,
            operations=operations,
        )


_GroupOperation = Callable[
    [tuple[object, ...], dict[object, str]],
    _Phase8GroupOutcome | tuple[tuple[object, str], ...] | None,
]


class _Phase8InvocationInconsistent(_Phase8OperationFault):
    """Provider evidence cannot safely be assigned to one complete group."""

    def __init__(self, code: _Phase8Reason) -> None:
        super().__init__(_Phase8FaultKind.INCONSISTENT, code, invocation_global=True)


def _phase8_max_repairs(invocation: object | None) -> int:
    rewrite = getattr(invocation, "rewrite", None)
    evaluation = getattr(rewrite, "evaluation", None)
    value = getattr(evaluation, "max_repair_iterations", None)
    return (
        value
        if type(value) is int
        else dict(getattr(_load_phase8_contract(), "limits", ())).get("max_repair_iterations", 0)
    )


@dataclass(frozen=True, slots=True, repr=False)
class _Phase8WireGroup:
    """Invocation-private correlations for one complete group operation."""

    group_token: str
    member_tokens: tuple[str, ...]
    context_tokens: tuple[str, ...] = ()
    mention_tokens: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class _Phase8ObligationId:
    """Stable compiler-private obligation identity; wire tokens are never stable."""

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private Phase 8 obligation identities are not serializable")


@dataclass(frozen=True, slots=True, repr=False)
class _Phase8Obligation:
    """Accepted analysis capability with private source lineage."""

    id: _Phase8ObligationId
    statement: str
    kind: str | None = None
    importance: str | None = None
    sensitivity: str | None = None
    source_members: tuple[object, ...] = ()
    source_mentions: tuple[object, ...] = ()

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private Phase 8 obligations are not serializable")


@dataclass(frozen=True, slots=True, repr=False)
class _Phase8WiredObligations:
    """Fresh stage-local tokens bound to accepted stable obligations."""

    privacy: tuple[tuple[str, _Phase8Obligation], ...]
    utility: tuple[tuple[str, _Phase8Obligation], ...]

    def request(self) -> dict[str, list[dict[str, object]]]:
        return {
            "privacy_obligations": [
                {
                    "obligation_token": token,
                    "statement": item.statement,
                    "kind": item.kind,
                    "sensitivity": item.sensitivity,
                }
                for token, item in self.privacy
            ],
            "utility_obligations": [
                {
                    "obligation_token": token,
                    "statement": item.statement,
                    "importance": item.importance,
                }
                for token, item in self.utility
            ],
        }


@dataclass(frozen=True, slots=True, repr=False)
class _Phase8AcceptedMention:
    """A Phase 6 accepted mention bound to its compiler-owned target."""

    owner: object
    identity: object
    start: int
    end: int
    text: str
    label: str
    source: str


@dataclass(frozen=True, slots=True, repr=False)
class _Phase8ContextProjection:
    """One admitted Phase 5 binding; it is never a Phase 8 member."""

    owner: object
    datum_binding: object
    ordinal: int
    text: str


@dataclass(frozen=True, slots=True, repr=False)
class _Phase8GroupInput:
    """Phase-8-owned provenance needed to admit the conditional zero route."""

    originals: dict[object, str]
    phase7_applied: dict[object, bool]
    accepted_mentions: tuple[_Phase8AcceptedMention, ...] = ()
    context_projections: tuple[_Phase8ContextProjection, ...] = ()
    privacy_goal: dict[str, str] | None = None
    strict_entity_protection: bool = True
    max_repairs: int = 3
    repair_threshold: float = 1.0
    repair_any_high_leak: bool = True
    utility_floor: float = 0.5


@dataclass(slots=True, repr=False)
class _Phase8WireRegistry:
    """Tracks only this invocation's private capabilities for stale-token detection."""

    issued: set[str] = field(default_factory=set)

    def new(self) -> str:
        token = _opaque_token()
        self.issued.add(token)
        return token


def _phase8_group_input(
    members: tuple[object, ...],
    originals: Mapping[object, str],
    applied: Mapping[object, bool],
    invocation: object | None,
) -> _Phase8GroupInput:
    """Detach the compiler-owned Phase 8 authority before any provider call."""
    rewrite = getattr(invocation, "rewrite", None)
    evaluation = getattr(rewrite, "evaluation", None)
    goal = getattr(rewrite, "privacy_goal", None)
    dumped = getattr(goal, "model_dump", None)
    privacy_goal = dumped() if callable(dumped) else None
    return _Phase8GroupInput(
        {member: originals[member] for member in members if member in originals},
        {member: applied[member] for member in members if member in applied},
        privacy_goal=privacy_goal if isinstance(privacy_goal, dict) else None,
        strict_entity_protection=getattr(rewrite, "strict_entity_protection", False) is True,
        max_repairs=getattr(evaluation, "max_repair_iterations", 3),
        repair_threshold=getattr(evaluation, "repair_threshold", 1.0),
        repair_any_high_leak=getattr(evaluation, "repair_any_high_leak", True),
        utility_floor=getattr(evaluation, "flag_utility_below", 0.5),
    )


def _phase8_group_input_from_successor(
    members: tuple[object, ...],
    predecessor: _Phase8SuccessorHandoff,
    invocation: object | None,
) -> _Phase8GroupInput | None:
    """Project authoritative Phase 5/6 values from one authenticated predecessor."""
    if not _is_admitted_phase8_successor(predecessor):
        return None
    phase6 = predecessor.phase6_plan
    originals = {datum.id: datum.text for datum in (*phase6.accounting.datums, *phase6.context.context_only_datums)}
    released = predecessor.phase7_execution.released
    applied = {value.datum_id: value.applied for value in released}
    if set(members) - set(originals) or set(members) - set(applied):
        return None
    mentions: list[_Phase8AcceptedMention] = []
    for handoff in predecessor.phase6_execution.handoffs:
        for resolved in handoff.resolved.mentions:
            mention = resolved.mention
            if mention.target_datum_id in members:
                mentions.append(
                    _Phase8AcceptedMention(
                        mention.target_datum_id,
                        mention.id,
                        mention.start,
                        mention.end,
                        mention.source_slice,
                        mention.detector_label,
                        mention.provenance.value,
                    )
                )
    contexts: list[_Phase8ContextProjection] = []
    for projection in phase6.context.projections:
        if projection.target_datum_id not in members:
            continue
        for binding in projection.bindings:
            text = originals.get(binding.datum_id)
            if text is None or binding.owner_task != projection.owner_task:
                return None
            contexts.append(
                _Phase8ContextProjection(projection.target_datum_id, binding.datum_id, binding.ordinal, text)
            )
    base = _phase8_group_input(
        members,
        cast(Mapping[object, str], originals),
        cast(Mapping[object, bool], applied),
        invocation,
    )
    return _Phase8GroupInput(
        base.originals,
        base.phase7_applied,
        tuple(mentions),
        tuple(contexts),
        base.privacy_goal,
        base.strict_entity_protection,
        base.max_repairs,
        base.repair_threshold,
        base.repair_any_high_leak,
        base.utility_floor,
    )


def _opaque_token() -> str:
    """Allocate an unguessable correlation capability, never a positional label."""
    return secrets.token_urlsafe(24)


def _backend_group_operation(
    backend: object,
    group_input: _Phase8GroupInput | None = None,
    registry: _Phase8WireRegistry | None = None,
    *,
    operation_plan: _Phase8GroupOperationPlan | None = None,
) -> Callable[[tuple[object, ...], dict[object, str]], _Phase8GroupOutcome]:
    registry = registry or _Phase8WireRegistry()
    limits = dict(getattr(_load_phase8_contract(), "limits", ()))
    if operation_plan is None:
        operation_plan = _compile_group_operation_plan(
            group_input.max_repairs if group_input is not None else limits.get("max_repair_iterations", 0),
            limits.get("max_repair_iterations", 0),
        )
    privacy: list[_Phase8Obligation] | None = None
    utility: list[_Phase8Obligation] | None = None
    completed: _Phase8GroupOutcome | None = None

    def analyze(members: tuple[object, ...], baselines: dict[object, str]) -> tuple[bool, bool]:
        nonlocal privacy, utility
        if not _valid_group_input(members, baselines, group_input):
            raise _Phase8OperationFault(_Phase8FaultKind.FAILED, _Phase8Reason.INVALID_GROUP_INPUT)
        wire = _new_wire(members, group_input, registry, include_mentions=True)
        tokens = wire.member_tokens
        request = _operation_request(_Phase8Operation.ANALYZE, wire, members, baselines, group_input)
        analysis = _dispatch(backend, _Phase8Operation.ANALYZE, request)
        if not _reconcile_common(analysis, "analyzed_member_tokens", tokens, wire.context_tokens, registry):
            raise _Phase8OperationFault(_Phase8FaultKind.INCONSISTENT, _Phase8Reason.ANALYSIS_RECONCILIATION)
        mention_owners = (
            dict(zip(wire.mention_tokens, (item.owner for item in group_input.accepted_mentions), strict=True))
            if group_input
            else {}
        )
        mention_identities = (
            dict(zip(wire.mention_tokens, (item.identity for item in group_input.accepted_mentions), strict=True))
            if group_input
            else {}
        )
        member_owners = dict(zip(tokens, members, strict=True))
        obligations = _admit_obligations(
            analysis, tokens, wire.mention_tokens, mention_owners, mention_identities, member_owners, registry
        )
        if obligations is None:
            raise _Phase8OperationFault(_Phase8FaultKind.FAILED, _Phase8Reason.ANALYSIS_INVALID)
        privacy, utility = obligations
        return not privacy, _zero_route_admitted(members, baselines, group_input)

    def rewrite(members: tuple[object, ...], baselines: dict[object, str]) -> dict[object, str]:
        if privacy is None or utility is None:
            raise _Phase8OperationFault(_Phase8FaultKind.INCONSISTENT, _Phase8Reason.ANALYSIS_STATE_MISSING)
        rewrite_wire = _new_wire(members, group_input, registry)
        rewrite_tokens = rewrite_wire.member_tokens
        rewrite_map = dict(zip(rewrite_tokens, members, strict=True))
        rewrite_obligations = _wire_obligations(privacy, utility, registry)
        active_request = {
            **_operation_request(_Phase8Operation.REWRITE, rewrite_wire, members, baselines, group_input),
            **rewrite_obligations.request(),
        }
        revision = _dispatch(backend, _Phase8Operation.REWRITE, active_request)
        if not _reconcile_common(revision, None, rewrite_tokens, rewrite_wire.context_tokens, registry):
            raise _Phase8OperationFault(_Phase8FaultKind.INCONSISTENT, _Phase8Reason.REWRITE_RECONCILIATION)
        current = _revisions(revision, rewrite_map, registry)
        if current is None:
            raise _Phase8OperationFault(_Phase8FaultKind.INCONSISTENT, _Phase8Reason.REWRITE_MEMBERS)
        return current

    def evaluate(
        members: tuple[object, ...], baselines: dict[object, str], current: dict[object, str]
    ) -> _Phase8Metric:
        if privacy is None or utility is None or group_input is None:
            raise _Phase8OperationFault(_Phase8FaultKind.INCONSISTENT, _Phase8Reason.ANALYSIS_STATE_MISSING)
        evaluation_wire = _new_wire(members, group_input, registry)
        evaluation_tokens = evaluation_wire.member_tokens
        evaluation_map = dict(zip(evaluation_tokens, members, strict=True))
        evaluation_obligations = _wire_obligations(privacy, utility, registry)
        evaluation_request = {
            **_operation_request(_Phase8Operation.EVALUATE, evaluation_wire, members, baselines, group_input),
            **evaluation_obligations.request(),
            "revisions": _revision_request(current, evaluation_map),
        }
        evaluation = _dispatch(backend, _Phase8Operation.EVALUATE, evaluation_request)
        metric = _metric(
            evaluation,
            evaluation_tokens,
            evaluation_wire.context_tokens,
            evaluation_obligations,
            registry,
            repair_any_high=group_input.repair_any_high_leak,
            repair_threshold=group_input.repair_threshold,
            utility_floor=group_input.utility_floor,
        )
        if metric is None:
            raise _Phase8OperationFault(_Phase8FaultKind.INCONSISTENT, _Phase8Reason.EVALUATION_RECONCILIATION)
        return metric

    def repair(
        members: tuple[object, ...], baselines: dict[object, str], current: dict[object, str], repair_round: int
    ) -> dict[object, str]:
        if privacy is None or utility is None:
            raise _Phase8OperationFault(_Phase8FaultKind.INCONSISTENT, _Phase8Reason.ANALYSIS_STATE_MISSING)
        repair_wire = _new_wire(members, group_input, registry)
        repair_tokens = repair_wire.member_tokens
        repair_map = dict(zip(repair_tokens, members, strict=True))
        repair_obligations = _wire_obligations(privacy, utility, registry)
        repair_request = {
            **_operation_request(_Phase8Operation.REPAIR, repair_wire, members, baselines, group_input),
            **repair_obligations.request(),
            "revisions": _revision_request(current, repair_map),
            "repair_round": repair_round,
        }
        repaired = _dispatch(backend, _Phase8Operation.REPAIR, repair_request)
        if not _reconcile_common(repaired, None, repair_tokens, repair_wire.context_tokens, registry):
            raise _Phase8OperationFault(_Phase8FaultKind.INCONSISTENT, _Phase8Reason.REPAIR_RECONCILIATION)
        revised = _revisions(repaired, repair_map, registry)
        if revised is None:
            raise _Phase8OperationFault(_Phase8FaultKind.INCONSISTENT, _Phase8Reason.REPAIR_MEMBERS)
        return revised

    def operation(members: tuple[object, ...], baselines: dict[object, str]) -> _Phase8GroupOutcome:
        nonlocal completed
        if completed is not None:
            raise _Phase8InvocationInconsistent(_Phase8Reason.GROUP_OPERATION_REUSED)
        max_repairs = operation_plan.max_repairs if operation_plan is not None else -1
        completed = _run_group_operation(
            members,
            baselines,
            analyze=lambda: analyze(members, baselines),
            rewrite=lambda current: rewrite(members, current),
            evaluate=lambda current: evaluate(members, baselines, current),
            repair=lambda current, repair_round: repair(members, baselines, current, repair_round),
            max_repairs=max_repairs,
            operation_plan=operation_plan,
        )
        return completed

    return operation


def _zero_route_admitted(
    members: tuple[object, ...], baselines: dict[object, str], group_input: _Phase8GroupInput | None
) -> bool:
    """Prove every frozen guard; absence of provenance is a rejection."""
    return (
        group_input is not None
        and not group_input.accepted_mentions
        and set(group_input.originals) == set(members)
        and set(group_input.phase7_applied) == set(members)
        and all(group_input.phase7_applied[member] is False for member in members)
        and all(baselines[member] == group_input.originals[member] for member in members)
    )


def _new_wire(
    members: tuple[object, ...],
    group_input: _Phase8GroupInput | None,
    registry: _Phase8WireRegistry,
    *,
    include_mentions: bool = False,
) -> _Phase8WireGroup:
    """Allocate attempt-local capabilities; they must never span a stage."""
    context_count = len(group_input.context_projections) if group_input is not None else 0
    mention_count = len(group_input.accepted_mentions) if group_input is not None and include_mentions else 0
    return _Phase8WireGroup(
        registry.new(),
        tuple(registry.new() for _ in members),
        tuple(registry.new() for _ in range(context_count)),
        tuple(registry.new() for _ in range(mention_count)),
    )


def _operation_request(
    operation: _Phase8Operation,
    wire: _Phase8WireGroup,
    members: tuple[object, ...],
    baselines: dict[object, str],
    group_input: _Phase8GroupInput | None,
) -> dict[str, object]:
    """Build the frozen, operation-specific allowlisted workframe."""
    originals = group_input.originals if group_input is not None else {}
    records: list[dict[str, object]] = [
        {"member_token": token, "original_text": originals.get(member), "phase7_baseline": baselines[member]}
        for member, token in zip(members, wire.member_tokens, strict=True)
    ]
    request: dict[str, object] = {
        "schema_version": "phase8-group-workframe/v1",
        "privacy_goal": group_input.privacy_goal if group_input is not None else None,
        "members": records,
        "context_bindings": _context_request(wire, members, group_input),
    }
    if operation is _Phase8Operation.ANALYZE:
        for record, mentions in zip(records, _mentions_by_owner(wire, members, group_input), strict=True):
            record["accepted_mentions"] = mentions
        request["strict_entity_protection"] = group_input.strict_entity_protection if group_input is not None else None
    elif operation is _Phase8Operation.REWRITE:
        request["strict_entity_protection"] = group_input.strict_entity_protection if group_input is not None else None
    elif operation is _Phase8Operation.EVALUATE:
        for record in records:
            record.pop("original_text")
            record.pop("phase7_baseline")
    else:
        request["strict_entity_protection"] = group_input.strict_entity_protection if group_input is not None else None
    return request


def _context_request(
    wire: _Phase8WireGroup, members: tuple[object, ...], group_input: _Phase8GroupInput | None
) -> list[dict[str, object]]:
    if group_input is None:
        return []
    member_tokens = dict(zip(members, wire.member_tokens, strict=True))
    return [
        {
            "binding_token": token,
            "owner_member_token": member_tokens[item.owner],
            "ordinal": item.ordinal,
            "text": item.text,
        }
        for item, token in zip(group_input.context_projections, wire.context_tokens, strict=True)
    ]


def _mentions_by_owner(
    wire: _Phase8WireGroup, members: tuple[object, ...], group_input: _Phase8GroupInput | None
) -> list[list[dict[str, object]]]:
    if group_input is None or not wire.mention_tokens:
        return [[] for _ in members]
    member_tokens = dict(zip(members, wire.member_tokens, strict=True))
    by_owner: dict[object, list[dict[str, object]]] = {member: [] for member in members}
    for item, token in zip(group_input.accepted_mentions, wire.mention_tokens, strict=True):
        by_owner[item.owner].append(
            {
                "mention_token": token,
                "owner_member_token": member_tokens[item.owner],
                "start": item.start,
                "end": item.end,
                "text": item.text,
                "label": item.label,
                "source": item.source,
            }
        )
    return [by_owner[member] for member in members]


def _valid_group_input(
    members: tuple[object, ...], baselines: dict[object, str], group_input: _Phase8GroupInput | None
) -> bool:
    if group_input is None:
        return True
    if set(group_input.originals) != set(members) or set(group_input.phase7_applied) != set(members):
        return False
    if not group_input.strict_entity_protection or (
        group_input.privacy_goal is not None and set(group_input.privacy_goal) != {"protect", "preserve"}
    ):
        return False
    mentions = group_input.accepted_mentions
    contexts = group_input.context_projections
    limits = dict(getattr(_load_phase8_contract(), "limits", ()))
    return (
        all(
            item.owner in baselines
            and type(item.start) is int
            and type(item.end) is int
            and 0 <= item.start < item.end <= len(group_input.originals[item.owner])
            and group_input.originals[item.owner][item.start : item.end] == item.text
            and item.label
            and item.source
            for item in mentions
        )
        and len({item.identity for item in mentions}) == len(mentions)
        and len(mentions) <= limits.get("max_accepted_mentions_per_group", 0)
        and all(len(item.text.encode("utf-8")) <= limits.get("max_member_text_utf8_bytes", 0) for item in mentions)
        and all(
            item.owner in baselines and type(item.ordinal) is int and item.ordinal >= 0 and isinstance(item.text, str)
            for item in contexts
        )
        and len({(item.owner, item.ordinal) for item in contexts}) == len(contexts)
        and all(len(item.text.encode("utf-8")) <= limits.get("max_context_fragment_utf8_bytes", 0) for item in contexts)
        and len(contexts) <= limits.get("max_context_bindings_per_group", 0)
        and sum(len(item.text.encode("utf-8")) for item in contexts)
        <= limits.get("max_all_context_utf8_bytes_per_group", 0)
    )


def _revision_request(current: dict[object, str], token_to_member: dict[str, object]) -> list[dict[str, str]]:
    return [{"member_token": token, "text": current[member]} for token, member in token_to_member.items()]


def _reconcile_common(
    payload: object,
    member_field: str | None,
    members: tuple[str, ...],
    contexts: tuple[str, ...],
    registry: _Phase8WireRegistry,
) -> bool:
    return (member_field is None or _exact_tokens(payload, member_field, members, registry)) and _exact_tokens(
        payload, "consumed_context_binding_tokens", contexts, registry
    )


def _admit_obligations(
    payload: object,
    members: tuple[str, ...],
    mentions: tuple[str, ...],
    mention_owners: dict[str, object],
    mention_identities: dict[str, object],
    member_owners: dict[str, object],
    registry: _Phase8WireRegistry,
) -> tuple[list[_Phase8Obligation], list[_Phase8Obligation]] | None:
    """Allocate obligation tokens locally after exact provenance validation."""
    admitted: list[list[_Phase8Obligation]] = []
    covered_mentions: set[str] = set()
    for obligation_field in ("privacy_obligations", "utility_obligations"):
        raw = _field(payload, obligation_field)
        limits = dict(getattr(_load_phase8_contract(), "limits", ()))
        limit_name = (
            "max_privacy_obligations_per_group"
            if obligation_field == "privacy_obligations"
            else "max_utility_obligations_per_group"
        )
        if not isinstance(raw, list) or len(raw) > limits.get(limit_name, 0):
            return None
        values: list[_Phase8Obligation] = []
        for value in raw:
            statement = _field(value, "statement")
            if (
                not isinstance(statement, str)
                or not statement
                or len(statement.encode()) > limits.get("max_obligation_statement_utf8_bytes", 0)
            ):
                return None
            if obligation_field == "privacy_obligations":
                owners = _field(value, "source_member_tokens", [])
                kind = _field(value, "kind")
                sensitivity = _field(value, "sensitivity")
                mention_tokens = _field(value, "source_mention_tokens", [])
                _raise_if_retired(owners, set(members), registry)
                _raise_if_retired(mention_tokens, set(mentions), registry)
                if (
                    not isinstance(owners, list)
                    or not owners
                    or len(owners) != len(set(owners))
                    or not set(owners) <= set(members)
                    or kind not in {"direct", "latent", "combination"}
                    or sensitivity not in {"high", "medium", "low"}
                    or not isinstance(mention_tokens, list)
                    or len(mention_tokens) != len(set(mention_tokens))
                    or not set(mention_tokens) <= set(mentions)
                    or any(
                        mention_owners[token] not in {member_owners[token] for token in owners}
                        for token in mention_tokens
                    )
                    or (kind == "direct" and not mention_tokens)
                ):
                    return None
                assert isinstance(kind, str) and isinstance(sensitivity, str)
                covered_mentions.update(mention_tokens)
                values.append(
                    _Phase8Obligation(
                        _Phase8ObligationId(),
                        statement,
                        kind,
                        None,
                        sensitivity,
                        tuple(member_owners[token] for token in owners),
                        tuple(mention_identities[token] for token in mention_tokens),
                    )
                )
            else:
                importance = _field(value, "importance")
                if importance not in {"critical", "important"}:
                    return None
                assert isinstance(importance, str)
                values.append(_Phase8Obligation(_Phase8ObligationId(), statement, None, importance))
        admitted.append(values)
    if covered_mentions != set(mentions):
        return None
    return admitted[0], admitted[1]


def _wire_obligations(
    privacy: list[_Phase8Obligation], utility: list[_Phase8Obligation], registry: _Phase8WireRegistry
) -> _Phase8WiredObligations:
    """Lower stable capabilities to fresh stage-local tokens, omitting source wires."""
    return _Phase8WiredObligations(
        tuple((registry.new(), item) for item in privacy),
        tuple((registry.new(), item) for item in utility),
    )


def _dispatch(backend: object, operation: _Phase8Operation, request: dict[str, object]) -> object:
    method = getattr(backend, "run_operation", None)
    if not callable(method):
        raise _Phase8OperationFault(_Phase8FaultKind.FAILED, _Phase8Reason.BACKEND_UNAVAILABLE)
    try:
        result = method(operation, request)
    except Exception:
        raise _Phase8OperationFault(_Phase8FaultKind.LOST, _Phase8Reason.TRANSPORT_LOST) from None
    if getattr(result, "operation", None) is not operation:
        raise _Phase8InvocationInconsistent(_Phase8Reason.OPERATION_CORRELATION_MISMATCH)
    if getattr(result, "failed", True):
        if getattr(result, "failure_kind", None) == "invocation_inconsistent":
            raise _Phase8InvocationInconsistent(_Phase8Reason.UNATTRIBUTABLE_PROVIDER_FAILURE)
        raise _Phase8OperationFault(_Phase8FaultKind.FAILED, _Phase8Reason.BACKEND_FAILURE)
    return getattr(result, "payload", None)


def _field(payload: object, name: str, default: object = None) -> object:
    if isinstance(payload, dict):
        return payload.get(name, default)
    return getattr(payload, name, default)


def _exact_tokens(payload: object, name: str, expected: tuple[str, ...], registry: _Phase8WireRegistry) -> bool:
    observed = _field(payload, name)
    if isinstance(observed, list):
        _raise_if_retired(observed, set(expected), registry)
    return _exact_token_list(observed, expected)


def _raise_if_retired(observed: object, current: set[str], registry: _Phase8WireRegistry) -> None:
    if isinstance(observed, list) and any(
        isinstance(token, str) and token in registry.issued - current for token in observed
    ):
        raise _Phase8InvocationInconsistent(_Phase8Reason.RETIRED_CORRELATION_TOKEN)


def _exact_token_list(observed: object, expected: tuple[str, ...]) -> bool:
    return (
        isinstance(observed, list)
        and len(observed) == len(expected)
        and set(observed) == set(expected)
        and len(set(observed)) == len(observed)
    )


def _revisions(
    payload: object, token_to_member: dict[str, object], registry: _Phase8WireRegistry
) -> dict[object, str] | None:
    revisions = _field(payload, "revisions")
    if not isinstance(revisions, list):
        return None
    observed = [_field(revision, "member_token") for revision in revisions]
    _raise_if_retired(observed, set(token_to_member), registry)
    if not _exact_token_list(observed, tuple(token_to_member)):
        return None
    result: dict[object, str] = {}
    limits = dict(getattr(_load_phase8_contract(), "limits", ()))
    total_bytes = 0
    for revision in revisions:
        token, text = _field(revision, "member_token"), _field(revision, "text")
        if not isinstance(token, str) or token not in token_to_member or not isinstance(text, str):
            raise _Phase8OperationFault(_Phase8FaultKind.FAILED, _Phase8Reason.REVISION_INVALID)
        encoded_bytes = len(text.encode("utf-8"))
        total_bytes += encoded_bytes
        if encoded_bytes > limits.get("max_revision_text_utf8_bytes_per_member", 0):
            raise _Phase8OperationFault(_Phase8FaultKind.FAILED, _Phase8Reason.REVISION_LIMIT)
        member = token_to_member[token]
        result[member] = text
    if total_bytes > limits.get("max_all_member_text_utf8_bytes_per_group", 0):
        raise _Phase8OperationFault(_Phase8FaultKind.FAILED, _Phase8Reason.REVISION_LIMIT)
    return result


def _metric(
    payload: object,
    tokens: tuple[str, ...],
    contexts: tuple[str, ...],
    obligations: _Phase8WiredObligations,
    registry: _Phase8WireRegistry,
    *,
    repair_any_high: bool,
    repair_threshold: float,
    utility_floor: float,
) -> _Phase8Metric | None:
    if payload is None or not _reconcile_common(payload, "evaluated_member_tokens", tokens, contexts, registry):
        return None
    privacy = _field(payload, "privacy_answers", [])
    utility = _field(payload, "utility_answers", [])
    privacy_by_token = dict(obligations.privacy)
    utility_by_token = dict(obligations.utility)
    if (
        not isinstance(privacy, list)
        or not isinstance(utility, list)
        or not _exact_answer_tokens(privacy, tuple(privacy_by_token), registry)
        or not _exact_answer_tokens(utility, tuple(utility_by_token), registry)
    ):
        return None
    try:
        privacy_values = tuple(_privacy_answer(answer, privacy_by_token) for answer in privacy)
        utility_values = tuple(_utility_answer(answer, utility_by_token) for answer in utility)
    except (TypeError, ValueError):
        raise _Phase8OperationFault(_Phase8FaultKind.FAILED, _Phase8Reason.EVALUATION_INVALID) from None
    metric = _evaluate_metrics(
        privacy_values,
        utility_values,
        repair_any_high=repair_any_high,
        repair_threshold=repair_threshold,
        utility_floor=utility_floor,
    )
    if metric is None:
        raise _Phase8OperationFault(_Phase8FaultKind.FAILED, _Phase8Reason.EVALUATION_INVALID)
    return metric


def _exact_answer_tokens(answers: list[object], expected: tuple[str, ...], registry: _Phase8WireRegistry) -> bool:
    tokens = [_field(answer, "obligation_token") for answer in answers]
    _raise_if_retired(tokens, set(expected), registry)
    return _exact_token_list(tokens, expected)


def _privacy_answer(answer: object, obligations: dict[str, _Phase8Obligation]) -> tuple[str, float, bool]:
    token, confidence, leaked = (
        _field(answer, "obligation_token"),
        _field(answer, "confidence"),
        _field(answer, "deducible"),
    )
    obligation = obligations.get(token) if isinstance(token, str) else None
    sensitivity = obligation.sensitivity if obligation is not None else None
    if (
        sensitivity not in {"high", "medium", "low"}
        or not isinstance(confidence, (int, float))
        or leaked
        not in {
            "yes",
            "no",
        }
    ):
        raise ValueError
    assert isinstance(sensitivity, str)
    return sensitivity, float(confidence), leaked == "yes"


def _utility_answer(answer: object, obligations: dict[str, _Phase8Obligation]) -> tuple[int, float]:
    token, score = _field(answer, "obligation_token"), _field(answer, "preservation_score")
    obligation = obligations.get(token) if isinstance(token, str) else None
    importance = obligation.importance if obligation is not None else None
    if importance not in {"critical", "important"} or not isinstance(score, (int, float)):
        raise ValueError
    return (2 if importance == "critical" else 1), float(score)


def _terminal(
    released: tuple[tuple[object, str], ...], states: tuple[str, ...], global_embargo: bool, cleanup_verified: bool
) -> _Phase8LifecycleExecution:
    return _Phase8LifecycleExecution(released, states, global_embargo, cleanup_verified)


def _lifecycle_preflight(
    groups: tuple[tuple[object, ...], ...],
    atomic_groups: tuple[tuple[object, ...], ...],
    dependencies: tuple[tuple[object, object], ...],
    operations: tuple[_GroupOperation, ...],
    phase7_released: tuple[tuple[object, str], ...],
    phase7_cleanup_verified: bool,
    phase7_global_embargo: bool,
) -> dict[object, str] | _Phase8LifecycleExecution:
    if not _valid_declarations(groups, atomic_groups, dependencies, operations):
        return _terminal((), (), True, False)
    if not phase7_cleanup_verified or phase7_global_embargo:
        return _terminal((), tuple("blocked" for _ in groups), True, False)
    baselines = _exact_baseline_index(phase7_released)
    return baselines if baselines is not None else _terminal((), tuple("inconsistent" for _ in groups), True, False)


def _run_operations(
    groups: tuple[tuple[object, ...], ...], operations: tuple[_GroupOperation, ...], baselines: dict[object, str]
) -> tuple[dict[object, str], list[str], bool]:
    candidates: dict[object, str] = {}
    states: list[str] = []
    invocation = _Phase8InvocationLedger()
    for members, operation in zip(groups, operations, strict=True):
        group_baselines = {member: baselines[member] for member in members if member in baselines}
        try:
            result = operation(members, group_baselines)
        except _Phase8InvocationInconsistent:
            states.append("inconsistent")
            invocation.admit(_GroupInconsistent(_Phase8Reason.INVOCATION_INCONSISTENT, True))
            states.extend("blocked" for _ in groups[len(states) :])
            break
        except Exception:
            states.append("lost")
            invocation.admit(_GroupLost(_Phase8Reason.TRANSPORT_LOST))
            states.extend("blocked" for _ in groups[len(states) :])
            break
        if isinstance(result, _Phase8GroupOutcome):
            terminal = result.terminal
            if isinstance(terminal, _GroupSucceeded) and not _validate_complete_revisions(members, result.revisions):
                terminal = _GroupInconsistent(_Phase8Reason.CANDIDATE_RECONCILIATION)
            states.append(result.state)
            if terminal is not result.terminal:
                states[-1] = "inconsistent"
            can_continue = invocation.admit(terminal)
            if isinstance(terminal, _GroupSucceeded) and isinstance(result.revisions, dict):
                candidates.update((member, result.revisions[member]) for member in members)
            if not can_continue:
                candidates.clear()
                states.extend("blocked" for _ in groups[len(states) :])
                break
            continue
        if not _complete_candidate(members, result):
            states.append("failed")
            invocation.admit(_GroupFailed(_Phase8Reason.INCOMPLETE_GROUP))
            continue
        candidates.update(cast(tuple[tuple[object, str], ...], result))
        states.append("succeeded")
        invocation.admit(_GroupSucceeded())
    return candidates, states, invocation.global_embargo


def _valid_declarations(
    groups: tuple[tuple[object, ...], ...],
    atomic_groups: tuple[tuple[object, ...], ...],
    dependencies: tuple[tuple[object, object], ...],
    operations: tuple[_GroupOperation, ...],
) -> bool:
    if len(groups) == 0 or len(groups) != len(operations):
        return False
    members = tuple(member for group in groups for member in group)
    if not members or len(set(members)) != len(members) or any(not group for group in groups):
        return False
    if any(not atomic or not set(atomic) <= set(members) for atomic in atomic_groups):
        return False
    if set(member for atomic in atomic_groups for member in atomic) != set(members):
        return False
    if any(sum(member in atomic for atomic in atomic_groups) != 1 for member in members):
        return False
    return all(
        isinstance(edge, tuple) and len(edge) == 2 and set(edge) <= set(members) for edge in dependencies
    ) and all(callable(operation) for operation in operations)


def _exact_baseline_index(values: object) -> dict[object, str] | None:
    if not isinstance(values, tuple) or any(not isinstance(item, tuple) or len(item) != 2 for item in values):
        return None
    try:
        result = {member: text for member, text in values}
    except TypeError:
        return None
    return result if len(result) == len(values) and all(isinstance(text, str) for _, text in values) else None


def _complete_candidate(members: tuple[object, ...], result: object) -> bool:
    return (
        isinstance(result, tuple)
        and len(result) == len(members)
        and all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], str) for item in result)
        and {item[0] for item in result} == set(members)
        and len({item[0] for item in result}) == len(members)
    )


def _phase4_released(
    groups: tuple[tuple[object, ...], ...],
    atomic_groups: tuple[tuple[object, ...], ...],
    dependencies: tuple[tuple[object, object], ...],
    states: list[str],
) -> set[object]:
    """Delegate atomic/dependency withholding to the Phase 4 fixed point."""
    members = tuple(member for group in groups for member in group)
    if not all(isinstance(member, _DatumId) for member in members):
        return set()
    datums = tuple(_TextDatum(member, "", _DatumPurpose.TARGET) for member in members if isinstance(member, _DatumId))
    if not all(isinstance(member, _DatumId) for group in atomic_groups for member in group) or not all(
        isinstance(item, _DatumId) for edge in dependencies for item in edge
    ):
        return set()
    try:
        plan: _AccountingPlan = _admit_accounting_plan(
            datums,
            (),
            (),
            tuple(
                _CompiledDependency(cast(_DatumId, prerequisite), cast(_DatumId, dependent))
                for prerequisite, dependent in dependencies
            ),
            tuple(
                _CompiledAtomicGroup(_AtomicGroupKey(), cast(tuple[_DatumId, ...], group)) for group in atomic_groups
            ),
            tuple(datum.id for datum in datums),
        )
    except (TypeError, ValueError):
        return set()
    locally_qualified = frozenset(
        member
        for group, state in zip(groups, states, strict=True)
        if state == "succeeded"
        for member in group
        if isinstance(member, _DatumId)
    )
    return set(_qualify_release(plan, locally_qualified).release_eligible)
