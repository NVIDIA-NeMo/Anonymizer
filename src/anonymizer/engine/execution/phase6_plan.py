# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sealed compiler output for the private Phase 6 Redact profile."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

from anonymizer.engine.execution.accounting_admission import _AccountingAdmissionCode
from anonymizer.engine.execution.accounting_plan import (
    _AccountingLimits,
    _AccountingPlan,
    _is_admitted_accounting_plan,
    _TaskKey,
    _TaskPredecessor,
)
from anonymizer.engine.execution.context_admission import (
    _compile_context_plan,
    _ContextAdmissionCode,
    _ContextPlan,
    _ContextRejected,
    _is_admitted_context_plan,
)
from anonymizer.engine.execution.context_contract import (
    _ContextBackendCapability,
    _ContextExecutionContract,
)
from anonymizer.engine.execution.mention_admission import (
    _MentionLimits,
    _MentionTarget,
    _MentionTargetToken,
)
from anonymizer.engine.execution.mention_resolution import _ResolverScope
from anonymizer.engine.execution.role_policy import (
    _is_admitted_policy,
    _load_redact_role_policy,
    _RolePolicy,
    _RolePolicyRejected,
)


class _PrivatePhase6PlanValue:
    def __repr__(self) -> str:
        return f"<private {type(self).__name__.strip('_').replace('_', ' ').lower()}>"

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private Phase 6 plan values are not serializable")


class _Phase6ProfileVersion(str, Enum):
    REDACT_V1 = "phase6-redact-graph/v1"


class _Phase6PlanRejectionCode(str, Enum):
    INVALID_PROFILE = "invalid_profile"


_Phase6RejectionCode: TypeAlias = _AccountingAdmissionCode | _ContextAdmissionCode | _Phase6PlanRejectionCode


@dataclass(frozen=True, slots=True, repr=False)
class _Phase6Rejected(_PrivatePhase6PlanValue):
    code: _Phase6RejectionCode


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class _Phase6ComponentKey(_PrivatePhase6PlanValue):
    """Compiler-issued identity for one target-context resolution component."""


@dataclass(frozen=True, slots=True, repr=False)
class _Phase6Component(_PrivatePhase6PlanValue):
    key: _Phase6ComponentKey
    target_tokens: tuple[_MentionTargetToken, ...]


@dataclass(frozen=True, slots=True, repr=False)
class _Phase6PlanProof(_PrivatePhase6PlanValue):
    seal: object = field(compare=False)
    snapshot: tuple[object, ...]


@dataclass(frozen=True, slots=True, repr=False)
class _Phase6Plan(_PrivatePhase6PlanValue):
    accounting: _AccountingPlan
    context: _ContextPlan
    targets: tuple[_MentionTarget, ...]
    resolver_scopes: tuple[_ResolverScope, ...]
    components: tuple[_Phase6Component, ...]
    mention_limits: _MentionLimits
    role_policy: _RolePolicy
    profile_version: _Phase6ProfileVersion
    _proof: _Phase6PlanProof | None = field(default=None, compare=False)


_PHASE6_PLAN_SEAL = object()
_PHASE6_STAGES = (
    "detect",
    "augment",
    "validate",
    "finalize",
    "resolve",
    "classify",
    "transform",
    "verify",
)


def _compile_phase6_plan(
    graph: object,
    *,
    accounting_limits: _AccountingLimits,
    context_contract: _ContextExecutionContract,
    capability: _ContextBackendCapability,
    mention_limits: _MentionLimits,
) -> _Phase6Plan | _Phase6Rejected:
    """Compile target-only mention and resolver identities before invocation effects."""
    context = _compile_context_plan(
        graph,
        accounting_limits=accounting_limits,
        contract=context_contract,
        capability=capability,
        stages=_PHASE6_STAGES,
    )
    if isinstance(context, _ContextRejected):
        return _Phase6Rejected(context.code)
    if not _valid_mention_limits(mention_limits):
        return _Phase6Rejected(_Phase6PlanRejectionCode.INVALID_PROFILE)
    policy = _load_redact_role_policy()
    if isinstance(policy, _RolePolicyRejected):
        return _Phase6Rejected(_Phase6PlanRejectionCode.INVALID_PROFILE)
    return _materialize_phase6_plan(context, mention_limits, policy)


def _materialize_phase6_plan(
    context: _ContextPlan,
    mention_limits: _MentionLimits,
    policy: _RolePolicy,
) -> _Phase6Plan | _Phase6Rejected:
    targets = tuple(_MentionTarget(_MentionTargetToken(), datum.id, datum.text) for datum in context.accounting.datums)
    scopes = _compile_resolver_scopes(context, targets)
    components = _compile_components(targets, scopes)
    predecessors = _compile_predecessors(context.accounting, targets, scopes, components)
    accounting = context.accounting.with_task_predecessors(predecessors)
    values = (
        accounting,
        context,
        targets,
        scopes,
        components,
        mention_limits,
        policy,
        _Phase6ProfileVersion.REDACT_V1,
    )
    candidate = _Phase6Plan(*values)
    snapshot = _phase6_plan_snapshot(candidate)
    if snapshot is None:
        return _Phase6Rejected(_Phase6PlanRejectionCode.INVALID_PROFILE)
    return _Phase6Plan(*values, _Phase6PlanProof(_PHASE6_PLAN_SEAL, snapshot))


def _compile_resolver_scopes(
    context: _ContextPlan,
    targets: tuple[_MentionTarget, ...],
) -> tuple[_ResolverScope, ...]:
    token_by_datum = {target.datum_id: target.token for target in targets}
    projection_by_target = {projection.target_datum_id: projection for projection in context.projections}
    return tuple(
        _ResolverScope(
            target.token,
            (
                target.token,
                *tuple(
                    token_by_datum[datum_id]
                    for datum_id in projection_by_target[target.datum_id].context_datum_ids
                    if datum_id in token_by_datum and datum_id != target.datum_id
                ),
            ),
        )
        for target in targets
    )


def _is_admitted_phase6_plan(value: object) -> bool:
    if not isinstance(value, _Phase6Plan) or value._proof is None:
        return False
    return (
        value._proof.seal is _PHASE6_PLAN_SEAL
        and _is_admitted_accounting_plan(value.accounting)
        and _is_admitted_context_plan(value.context)
        and _is_admitted_policy(value.role_policy)
        and value._proof.snapshot == _phase6_plan_snapshot(value)
    )


def _compile_components(
    targets: tuple[_MentionTarget, ...],
    scopes: tuple[_ResolverScope, ...],
) -> tuple[_Phase6Component, ...]:
    parents = {target.token: target.token for target in targets}

    def find(token: _MentionTargetToken) -> _MentionTargetToken:
        while parents[token] is not token:
            token = parents[token]
        return token

    for scope in scopes:
        owner_root = find(scope.owner)
        for eligible in scope.eligible_targets:
            eligible_root = find(eligible)
            if owner_root is not eligible_root:
                parents[eligible_root] = owner_root
    grouped: dict[_MentionTargetToken, list[_MentionTargetToken]] = {}
    for target in targets:
        grouped.setdefault(find(target.token), []).append(target.token)
    return tuple(_Phase6Component(_Phase6ComponentKey(), tuple(members)) for members in grouped.values())


def _compile_predecessors(
    accounting: _AccountingPlan,
    targets: tuple[_MentionTarget, ...],
    scopes: tuple[_ResolverScope, ...],
    components: tuple[_Phase6Component, ...],
) -> tuple[_TaskPredecessor, ...]:
    datum_by_token = {target.token: target.datum_id for target in targets}
    stage_by_value = {stage.value: stage for stage in accounting.stages}
    edges: list[_TaskPredecessor] = []
    for scope in scopes:
        resolve = _TaskKey(stage_by_value["resolve"], datum_by_token[scope.owner])
        for token in scope.eligible_targets:
            if token is scope.owner:
                continue
            edges.append(
                _TaskPredecessor(
                    _TaskKey(stage_by_value["finalize"], datum_by_token[token]),
                    resolve,
                )
            )
    return tuple(edges)


def _phase6_plan_snapshot(plan: _Phase6Plan) -> tuple[object, ...] | None:
    try:
        return (
            plan.accounting._proof,
            plan.context._proof,
            tuple((target.token, target.datum_id.value, target.text) for target in plan.targets),
            tuple((scope.owner, scope.eligible_targets) for scope in plan.resolver_scopes),
            tuple((component.key, component.target_tokens) for component in plan.components),
            (
                plan.mention_limits.max_candidates_per_target,
                plan.mention_limits.max_mentions_per_target,
                plan.mention_limits.max_label_bytes,
                plan.mention_limits.max_source_slice_bytes,
            ),
            plan.role_policy._proof,
            plan.profile_version.value,
        )
    except (AttributeError, TypeError):
        return None


def _valid_mention_limits(limits: object) -> bool:
    return isinstance(limits, _MentionLimits) and all(
        type(value) is int and value > 0
        for value in (
            limits.max_candidates_per_target,
            limits.max_mentions_per_target,
            limits.max_label_bytes,
            limits.max_source_slice_bytes,
        )
    )
