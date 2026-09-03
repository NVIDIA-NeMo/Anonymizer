# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private-only composition seam for the Phase 8 grouped profile."""

from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass
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
from anonymizer.engine.execution.phase8_admission import _compile_phase8_plan, _Phase8Plan
from anonymizer.engine.execution.phase8_contract import _load_phase8_contract
from anonymizer.engine.execution.phase8_ndd_backend import _Phase8Operation
from anonymizer.engine.execution.phase8_runtime import _Phase8LifecycleExecution, _run_group_operation
from anonymizer.engine.execution.phase8_validation import _evaluate_metrics, _Phase8Metric


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
        self, graph: object, phase7: _Phase7Execution, backend: object
    ) -> _Phase8LifecycleExecution:
        """Execute the admitted complete groups through the sole NDD boundary.

        Requests use fresh opaque wire tokens and are reconciled back to the
        compiler-issued datum keys before any candidate reaches Phase 4.
        """
        plan = _compile_phase8_plan(graph)
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
        operations = tuple(
            _backend_group_operation(
                backend,
                _Phase8GroupInput(
                    {member: original_by_datum[member] for member in members if member in original_by_datum},
                    {member: applied_by_datum[member] for member in members if member in applied_by_datum},
                ),
            )
            for members in groups
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


_GroupOperation = Callable[[tuple[object, ...], dict[object, str]], tuple[tuple[object, str], ...] | None]


class _Phase8InvocationInconsistent(RuntimeError):
    """Provider evidence cannot safely be assigned to one complete group."""


@dataclass(frozen=True, slots=True, repr=False)
class _Phase8WireGroup:
    """Invocation-private correlations for one complete group operation."""

    group_token: str
    member_tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True, repr=False)
class _Phase8GroupInput:
    """Phase-8-owned provenance needed to admit the conditional zero route."""

    originals: dict[object, str]
    phase7_applied: dict[object, bool]
    accepted_mentions: tuple[object, ...] = ()


def _opaque_token() -> str:
    """Allocate an unguessable correlation capability, never a positional label."""
    return secrets.token_urlsafe(24)


def _backend_group_operation(backend: object, group_input: _Phase8GroupInput | None = None) -> _GroupOperation:
    def operation(members: tuple[object, ...], baselines: dict[object, str]) -> tuple[tuple[object, str], ...] | None:
        wire = _new_wire(members)
        tokens = wire.member_tokens
        request = _operation_request(wire, members, baselines)
        analysis = _dispatch(backend, _Phase8Operation.ANALYZE, request)
        if analysis is None or not _reconcile_common(analysis, "analyzed_member_tokens", tokens, ()):
            return None
        obligations = _admit_obligations(analysis, tokens)
        if obligations is None:
            return None
        privacy, utility = obligations
        if not privacy:
            if utility or not _zero_route_admitted(members, baselines, group_input):
                return None
            return tuple((member, baselines[member]) for member in members)
        rewrite_wire = _new_wire(members)
        rewrite_tokens = rewrite_wire.member_tokens
        rewrite_map = dict(zip(rewrite_tokens, members, strict=True))
        active_request = {
            **_operation_request(rewrite_wire, members, baselines),
            "privacy_obligations": privacy,
            "utility_obligations": utility,
        }
        revision = _dispatch(backend, _Phase8Operation.REWRITE, active_request)
        if revision is None or not _reconcile_common(revision, None, rewrite_tokens, ()):
            return None
        current = _revisions(revision, rewrite_map)
        if current is None:
            return None
        limits = dict(getattr(_load_phase8_contract(), "limits", ()))
        for repair_round in range(limits.get("max_repair_iterations", 0) + 1):
            evaluation_wire = _new_wire(members)
            evaluation_tokens = evaluation_wire.member_tokens
            evaluation_map = dict(zip(evaluation_tokens, members, strict=True))
            evaluation_request = {
                **_operation_request(evaluation_wire, members, baselines),
                "privacy_obligations": privacy,
                "utility_obligations": utility,
                "revisions": _revision_request(current, evaluation_map),
            }
            evaluation = _dispatch(backend, _Phase8Operation.EVALUATE, evaluation_request)
            metric = _metric(evaluation, evaluation_tokens, _obligation_tokens(privacy), _obligation_tokens(utility))
            if metric is None:
                return None
            if not metric.needs_repair:
                return tuple((member, current[member]) for member in members)
            if repair_round == limits.get("max_repair_iterations", 0):
                return None
            repair_wire = _new_wire(members)
            repair_tokens = repair_wire.member_tokens
            repair_map = dict(zip(repair_tokens, members, strict=True))
            repair_request = {
                **_operation_request(repair_wire, members, baselines),
                "privacy_obligations": privacy,
                "utility_obligations": utility,
                "revisions": _revision_request(current, repair_map),
                "repair_round": repair_round + 1,
            }
            repaired = _dispatch(backend, _Phase8Operation.REPAIR, repair_request)
            if repaired is None or not _reconcile_common(repaired, None, repair_tokens, ()):
                return None
            current = _revisions(repaired, repair_map)
            if current is None:
                return None
        return None

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


def _new_wire(members: tuple[object, ...]) -> _Phase8WireGroup:
    """Allocate attempt-local capabilities; they must never span a stage."""
    return _Phase8WireGroup(_opaque_token(), tuple(_opaque_token() for _ in members))


def _operation_request(
    wire: _Phase8WireGroup, members: tuple[object, ...], baselines: dict[object, str]
) -> dict[str, object]:
    """Build the complete content-bearing workframe owned by this service."""
    return {
        "group_token": wire.group_token,
        "operation_token": _opaque_token(),
        "members": [
            {"member_token": token, "ordinal": ordinal, "phase7_baseline": baselines[member]}
            for ordinal, (member, token) in enumerate(zip(members, wire.member_tokens, strict=True))
        ],
        # Graph handoffs with no accepted context or mentions still carry the
        # fields explicitly; later graph profiles populate the same schemas.
        "context_bindings": [],
        "accepted_mentions": [],
        "privacy_obligations": [],
        "utility_obligations": [],
    }


def _revision_request(current: dict[object, str], token_to_member: dict[str, object]) -> list[dict[str, str]]:
    return [{"member_token": token, "text": current[member]} for token, member in token_to_member.items()]


def _reconcile_common(
    payload: object, member_field: str | None, members: tuple[str, ...], contexts: tuple[str, ...]
) -> bool:
    return (member_field is None or _exact_tokens(payload, member_field, members)) and _exact_tokens(
        payload, "consumed_context_binding_tokens", contexts
    )


def _admit_obligations(
    payload: object, members: tuple[str, ...]
) -> tuple[list[dict[str, object]], list[dict[str, object]]] | None:
    """Allocate obligation tokens locally after exact provenance validation."""
    admitted: list[list[dict[str, object]]] = []
    for field in ("privacy_obligations", "utility_obligations"):
        raw = _field(payload, field)
        limits = dict(getattr(_load_phase8_contract(), "limits", ()))
        limit_name = (
            "max_privacy_obligations_per_group"
            if field == "privacy_obligations"
            else "max_utility_obligations_per_group"
        )
        if not isinstance(raw, list) or len(raw) > limits.get(limit_name, 0):
            return None
        values: list[dict[str, object]] = []
        for value in raw:
            statement = _field(value, "statement")
            if (
                not isinstance(statement, str)
                or not statement
                or len(statement.encode()) > limits.get("max_obligation_statement_utf8_bytes", 0)
            ):
                return None
            if field == "privacy_obligations":
                owners = _field(value, "source_member_tokens", [])
                kind = _field(value, "kind")
                sensitivity = _field(value, "sensitivity")
                mention_tokens = _field(value, "source_mention_tokens", [])
                if (
                    not isinstance(owners, list)
                    or not owners
                    or len(owners) != len(set(owners))
                    or not set(owners) <= set(members)
                    or kind not in {"direct", "latent", "combination"}
                    or sensitivity not in {"high", "medium", "low"}
                    or not isinstance(mention_tokens, list)
                    or len(mention_tokens) != len(set(mention_tokens))
                ):
                    return None
                values.append(
                    {
                        "obligation_token": _opaque_token(),
                        "statement": statement,
                        "kind": kind,
                        "sensitivity": sensitivity,
                        "source_member_tokens": owners,
                        "source_mention_tokens": mention_tokens,
                    }
                )
            else:
                importance = _field(value, "importance")
                if importance not in {"critical", "important"}:
                    return None
                values.append({"obligation_token": _opaque_token(), "statement": statement, "importance": importance})
        admitted.append(values)
    return admitted[0], admitted[1]


def _obligation_tokens(obligations: list[dict[str, object]]) -> tuple[str, ...]:
    return tuple(cast(str, obligation["obligation_token"]) for obligation in obligations)


def _dispatch(backend: object, operation: _Phase8Operation, request: dict[str, object]) -> object | None:
    method = getattr(backend, "run_operation", None)
    if not callable(method):
        return None
    try:
        result = method(operation, request)
    except Exception:
        return None
    if getattr(result, "operation", None) is not operation:
        raise _Phase8InvocationInconsistent("operation correlation mismatch")
    if getattr(result, "failed", True):
        if getattr(result, "failure_kind", None) == "invocation_inconsistent":
            raise _Phase8InvocationInconsistent("unattributable provider failure")
        return None
    return getattr(result, "payload", None)


def _field(payload: object, name: str, default: object = None) -> object:
    if isinstance(payload, dict):
        return payload.get(name, default)
    return getattr(payload, name, default)


def _exact_tokens(payload: object, name: str, expected: tuple[str, ...]) -> bool:
    observed = _field(payload, name)
    return _exact_token_list(observed, expected)


def _exact_token_list(observed: object, expected: tuple[str, ...]) -> bool:
    return (
        isinstance(observed, list)
        and len(observed) == len(expected)
        and set(observed) == set(expected)
        and len(set(observed)) == len(observed)
    )


def _revisions(payload: object, token_to_member: dict[str, object]) -> dict[object, str] | None:
    revisions = _field(payload, "revisions")
    if not isinstance(revisions, list):
        return None
    result: dict[object, str] = {}
    for revision in revisions:
        token, text = _field(revision, "member_token"), _field(revision, "text")
        if not isinstance(token, str) or not isinstance(text, str) or token not in token_to_member:
            return None
        member = token_to_member[token]
        if member in result:
            return None
        result[member] = text
    return result if len(result) == len(token_to_member) else None


def _metric(
    payload: object, tokens: tuple[str, ...], privacy_tokens: tuple[str, ...], utility_tokens: tuple[str, ...]
) -> _Phase8Metric | None:
    if payload is None or not _reconcile_common(payload, "evaluated_member_tokens", tokens, ()):
        return None
    privacy = _field(payload, "privacy_answers", [])
    utility = _field(payload, "utility_answers", [])
    if (
        not isinstance(privacy, list)
        or not isinstance(utility, list)
        or not _exact_answer_tokens(privacy, privacy_tokens)
        or not _exact_answer_tokens(utility, utility_tokens)
    ):
        return None
    try:
        privacy_values = tuple(_privacy_answer(answer) for answer in privacy)
        utility_values = tuple(_utility_answer(answer) for answer in utility)
    except (TypeError, ValueError):
        return None
    return _evaluate_metrics(
        privacy_values, utility_values, repair_any_high=True, repair_threshold=0.0, utility_floor=0.5
    )


def _exact_answer_tokens(answers: list[object], expected: tuple[str, ...]) -> bool:
    return _exact_token_list([_field(answer, "obligation_token") for answer in answers], expected)


def _privacy_answer(answer: object) -> tuple[str, float, bool]:
    sensitivity, confidence, leaked = (
        _field(answer, "sensitivity", "high"),
        _field(answer, "confidence"),
        _field(answer, "deducible"),
    )
    if not isinstance(sensitivity, str) or not isinstance(confidence, (int, float)) or leaked not in {"yes", "no"}:
        raise ValueError
    return sensitivity, float(confidence), leaked == "yes"


def _utility_answer(answer: object) -> tuple[int, float]:
    importance, score = _field(answer, "importance", "important"), _field(answer, "preservation_score")
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
    invocation_inconsistent = False
    for members, operation in zip(groups, operations, strict=True):
        group_baselines = {member: baselines[member] for member in members if member in baselines}
        if len(group_baselines) != len(members):
            states.append("blocked")
            continue
        try:
            result = operation(members, group_baselines)
        except _Phase8InvocationInconsistent:
            states.append("inconsistent")
            invocation_inconsistent = True
            continue
        except Exception:
            states.append("failed")
            continue
        if not _complete_candidate(members, result):
            states.append("failed")
            continue
        candidates.update(cast(tuple[tuple[object, str], ...], result))
        states.append("succeeded")
    return candidates, states, invocation_inconsistent


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
