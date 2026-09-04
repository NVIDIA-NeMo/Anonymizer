# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure finite oracle for the private Phase 8 grouped-rewrite contract.

The model deliberately imports no production, dataframe, or provider code.
It reduces symbolic identities and terminal evidence only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TypedDict

REFERENCE_MODEL_VERSION = "phase8-grouped-rewrite-reference-model/v1"
GENERATOR_VERSION = "phase8-finite-envelope/v1"
MAX_DATUMS = 4
MAX_GROUPS = 2
MAX_MEMBERS = 4
MAX_REPAIRS = 3
MAX_WORKFRAME_BYTES = 65_536

_LOCAL_TERMINALS = {"failed", "blocked", "inconsistent"}
_GLOBAL_TERMINALS = {"cancelled", "lost", "global_inconsistent"}
_PRECEDENCE = {"blocked": 0, "failed": 1, "cancelled": 2, "lost": 3, "inconsistent": 4}


@dataclass(frozen=True, slots=True)
class ReferenceGroup:
    members: tuple[str, ...]
    terminal_events: tuple[str, ...] = ("succeeded",)
    result_keys: tuple[str, ...] | None = None
    evaluations: tuple[bool, ...] = (False,)
    repair_keys: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class ReferenceCase:
    name: str
    targets: tuple[str, ...]
    groups: tuple[ReferenceGroup, ...]
    atomic_groups: tuple[tuple[str, ...], ...]
    dependencies: tuple[tuple[str, str], ...] = ()
    max_repairs: int = 0
    strict: bool = True
    mention_evidence: str = "exact"
    context_evidence: str = "exact"
    consumed_binding_evidence: str = "exact"
    capability: str = "stable"
    retention: str = "disabled"
    prompt: str = "stable"
    model_route: str = "exact"
    failure_evidence: str = "none"
    pre_cleanup: str = "verified"
    post_cleanup: str = "verified"
    workframe_bytes: int = 1


@dataclass(frozen=True, slots=True)
class ReferenceResult:
    admission: str
    group_states: tuple[str, ...]
    aggregate: str
    invocation: str
    released: tuple[str, ...]
    reason: str | None = None


class _ReferenceCaseBase(TypedDict):
    targets: tuple[str, ...]
    groups: tuple[ReferenceGroup, ...]
    atomic_groups: tuple[tuple[str, ...], ...]


def reduce_reference(case: ReferenceCase) -> ReferenceResult:
    """Reduce one bounded symbolic trace into terminal and release evidence."""
    rejected = _admission_rejection(case)
    if rejected is not None:
        return ReferenceResult("rejected", (), "blocked", "failed", (), rejected)
    compiled = tuple(sorted(case.groups, key=lambda group: min(case.targets.index(key) for key in group.members)))
    corruption = _pre_dispatch_corruption(case)
    if corruption is not None:
        return ReferenceResult(
            "admitted",
            tuple("blocked" for _ in compiled),
            "inconsistent",
            "inconsistent",
            (),
            corruption,
        )
    states: list[str] = []
    eligible: set[str] = set()
    invocation = "completed"
    stopped = False
    for group_index, group in enumerate(compiled):
        if stopped:
            states.append("blocked")
            continue
        state = (
            "failed"
            if group_index == 0 and case.workframe_bytes > MAX_WORKFRAME_BYTES
            else _reduce_group(group, case.max_repairs)
        )
        states.append(state)
        if state == "succeeded":
            eligible.update(group.members)
        elif state in _GLOBAL_TERMINALS:
            invocation = "inconsistent" if state == "global_inconsistent" else state
            stopped = True
    projected = tuple("inconsistent" if state == "global_inconsistent" else state for state in states)
    non_success = tuple(state for state in projected if state != "succeeded")
    aggregate = max(non_success, key=_PRECEDENCE.__getitem__) if non_success else "succeeded"
    if invocation != "completed":
        return ReferenceResult("admitted", projected, aggregate, invocation, (), invocation)
    cleanup = _cleanup_terminal(case.pre_cleanup, case.post_cleanup)
    if cleanup is not None:
        cleanup_invocation, reason = cleanup
        return ReferenceResult("admitted", projected, aggregate, cleanup_invocation, (), reason)
    released = _phase4_release(case.targets, case.atomic_groups, case.dependencies, eligible)
    return ReferenceResult("admitted", projected, aggregate, "completed", released)


def _admission_rejection(case: ReferenceCase) -> str | None:
    if not case.strict:
        return "strict_false"
    if (
        not case.targets
        or len(case.targets) != len(set(case.targets))
        or len(case.targets) > MAX_DATUMS
        or not case.groups
        or len(case.groups) > MAX_GROUPS
        or type(case.max_repairs) is not int
        or not 0 <= case.max_repairs <= MAX_REPAIRS
    ):
        return "invocation_limit_or_shape"
    declared = tuple(key for group in case.groups for key in group.members)
    if any(not group.members or len(group.members) > MAX_MEMBERS for group in case.groups):
        return "group_limit_or_empty"
    if len(declared) != len(set(declared)) or set(declared) != set(case.targets):
        return "partition"
    if any(sum(set(group.members) <= set(atomic) for atomic in case.atomic_groups) != 1 for group in case.groups):
        return "atomic_refinement"
    if not case.atomic_groups:
        return "atomic_partition"
    atomic_members = tuple(key for group in case.atomic_groups for key in group)
    if len(atomic_members) != len(set(atomic_members)) or set(atomic_members) != set(case.targets):
        return "atomic_partition"
    if (
        len(case.dependencies) != len(set(case.dependencies))
        or any(
            left not in case.targets or right not in case.targets or left == right for left, right in case.dependencies
        )
        or _has_dependency_cycle(case.targets, case.dependencies)
    ):
        return "dependency_graph"
    if type(case.workframe_bytes) is not int or case.workframe_bytes < 0:
        return "workframe_shape"
    return None


def _pre_dispatch_corruption(case: ReferenceCase) -> str | None:
    if case.mention_evidence != "exact":
        return "mention_reconciliation"
    if case.context_evidence != "exact" or case.consumed_binding_evidence != "exact":
        return "context_reconciliation"
    if (
        case.capability != "stable"
        or case.retention != "disabled"
        or case.prompt != "stable"
        or case.model_route != "exact"
    ):
        return "capability"
    if case.failure_evidence not in {"none", "bound"}:
        return "failure_attribution"
    return None


def _reduce_group(group: ReferenceGroup, max_repairs: int) -> str:
    if not group.terminal_events:
        return "inconsistent"
    first = group.terminal_events[0]
    if first in _LOCAL_TERMINALS | _GLOBAL_TERMINALS:
        return first
    if first != "succeeded":
        return "inconsistent"
    result_keys = group.members if group.result_keys is None else group.result_keys
    if set(result_keys) != set(group.members) or len(result_keys) != len(group.members):
        return "inconsistent"
    if not group.evaluations:
        return "inconsistent"
    if False in group.evaluations[:-1]:
        return "inconsistent"
    needed = len(group.evaluations) - 1
    if len(group.repair_keys) != needed or needed > max_repairs:
        return "inconsistent"
    if any(set(keys) != set(group.members) or len(keys) != len(group.members) for keys in group.repair_keys):
        return "inconsistent"
    if group.evaluations[-1]:
        return "failed" if needed == max_repairs else "inconsistent"
    return "succeeded"


def _has_dependency_cycle(targets: tuple[str, ...], dependencies: tuple[tuple[str, str], ...]) -> bool:
    outgoing = {target: [] for target in targets}
    indegree = {target: 0 for target in targets}
    for prerequisite, dependent in dependencies:
        outgoing[prerequisite].append(dependent)
        indegree[dependent] += 1
    ready = [target for target in targets if indegree[target] == 0]
    visited = 0
    while ready:
        current = ready.pop()
        visited += 1
        for dependent in outgoing[current]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
    return visited != len(targets)


def _cleanup_terminal(pre: str, post: str) -> tuple[str, str] | None:
    evidence = (pre, post)
    if any(item not in {"verified", "failed"} for item in evidence):
        return "inconsistent", "cleanup_unconfirmed"
    if "failed" in evidence:
        return "failed", "cleanup_failed"
    return None


def _phase4_release(
    targets: tuple[str, ...],
    atomic_groups: tuple[tuple[str, ...], ...],
    dependencies: tuple[tuple[str, str], ...],
    eligible: set[str],
) -> tuple[str, ...]:
    withheld = {key for atomic in atomic_groups if not set(atomic) <= eligible for key in atomic}
    changed = True
    while changed:
        changed = False
        for prerequisite, dependent in dependencies:
            if prerequisite in withheld and dependent not in withheld:
                withheld.add(dependent)
                changed = True
        for atomic in atomic_groups:
            if withheld.intersection(atomic) and not set(atomic) <= withheld:
                withheld.update(atomic)
                changed = True
    return tuple(key for key in targets if key not in withheld and key in eligible)


def finite_reference_cases() -> tuple[ReferenceCase, ...]:
    """Return the governed finite corpus spanning every frozen decision class."""
    a, b, c, d = "a", "b", "c", "d"
    one = (ReferenceGroup((a,)),)
    pair = (ReferenceGroup((a, b)),)
    separate = (ReferenceGroup((a,)), ReferenceGroup((b,)))
    base_one: _ReferenceCaseBase = {"targets": (a,), "groups": one, "atomic_groups": ((a,),)}
    base_pair: _ReferenceCaseBase = {"targets": (a, b), "groups": pair, "atomic_groups": ((a, b),)}
    directed = (
        ReferenceCase("valid-single", **base_one),
        ReferenceCase("valid-group", **base_pair),
        ReferenceCase("declared-order-normalized", (a, b), tuple(reversed(separate)), ((a,), (b,))),
        ReferenceCase(
            "declared-order-terminal-normalized",
            (a, b),
            (ReferenceGroup((b,)), ReferenceGroup((a,), ("failed",))),
            ((a,), (b,)),
        ),
        ReferenceCase("partial-result", (a, b), (ReferenceGroup((a, b), result_keys=(a,)),), ((a, b),)),
        ReferenceCase("extra-result", (a,), (ReferenceGroup((a,), result_keys=(a, b)),), ((a,),)),
        ReferenceCase(
            "local-failed-disconnected", (a, b), (ReferenceGroup((a,), ("failed",)), *separate[1:]), ((a,), (b,))
        ),
        ReferenceCase(
            "local-inconsistent-disconnected",
            (a, b),
            (ReferenceGroup((a,), ("inconsistent",)), *separate[1:]),
            ((a,), (b,)),
        ),
        ReferenceCase(
            "precedence-failed-inconsistent",
            (a, b),
            (ReferenceGroup((a,), ("failed",)), ReferenceGroup((b,), ("inconsistent",))),
            ((a,), (b,)),
        ),
        ReferenceCase("atomic-sibling-withheld", (a, b), (ReferenceGroup((a,), ("failed",)), *separate[1:]), ((a, b),)),
        ReferenceCase(
            "dependency-closure",
            (a, b),
            (ReferenceGroup((a,), ("failed",)), *separate[1:]),
            ((a,), (b,)),
            ((a, b),),
        ),
        ReferenceCase("cancel-stops", (a, b), (ReferenceGroup((a,), ("cancelled",)), *separate[1:]), ((a,), (b,))),
        ReferenceCase("loss-stops", (a, b), (ReferenceGroup((a,), ("lost",)), *separate[1:]), ((a,), (b,))),
        ReferenceCase(
            "global-inconsistent-stops",
            (a, b),
            (ReferenceGroup((a,), ("global_inconsistent",)), *separate[1:]),
            ((a,), (b,)),
        ),
        ReferenceCase("late-success-absorbed", (a,), (ReferenceGroup((a,), ("failed", "succeeded")),), ((a,),)),
        ReferenceCase(
            "repair-pass",
            (a,),
            (ReferenceGroup((a,), evaluations=(True, False), repair_keys=((a,),)),),
            ((a,),),
            max_repairs=1,
        ),
        ReferenceCase("repair-exhausted", (a,), (ReferenceGroup((a,), evaluations=(True,), repair_keys=()),), ((a,),)),
        ReferenceCase(
            "subset-repair",
            (a, b),
            (ReferenceGroup((a, b), evaluations=(True, False), repair_keys=((a,),)),),
            ((a, b),),
            max_repairs=1,
        ),
        ReferenceCase(
            "directed-three-repairs-pass",
            (a,),
            (ReferenceGroup((a,), evaluations=(True, True, True, False), repair_keys=((a,), (a,), (a,))),),
            ((a,),),
            max_repairs=3,
        ),
        ReferenceCase("fourth-repair-rejected", max_repairs=4, **base_one),
        ReferenceCase("skipped-evaluation", (a,), (ReferenceGroup((a,), evaluations=()),), ((a,),)),
        ReferenceCase("strict-false", strict=False, **base_one),
        ReferenceCase("coverage-gap", (a, b), one, ((a,), (b,))),
        ReferenceCase("overlap", (a, b), (ReferenceGroup((a, b)), ReferenceGroup((b,))), ((a, b),)),
        ReferenceCase("cross-atomic", (a, b), pair, ((a,), (b,))),
        ReferenceCase("datum-limit-exact", (a, b, c, d), (ReferenceGroup((a, b, c, d)),), ((a, b, c, d),)),
        ReferenceCase(
            "datum-limit-over", (a, b, c, d, "e"), (ReferenceGroup((a, b, c, d, "e")),), ((a, b, c, d, "e"),)
        ),
        ReferenceCase(
            "group-limit-over",
            (a, b, c),
            (ReferenceGroup((a,)), ReferenceGroup((b,)), ReferenceGroup((c,))),
            ((a,), (b,), (c,)),
        ),
        *(
            ReferenceCase(f"mention-{kind}", mention_evidence=kind, **base_one)
            for kind in ("missing", "duplicate", "foreign", "wrong_owner")
        ),
        *(
            ReferenceCase(f"context-{kind}", context_evidence=kind, **base_one)
            for kind in ("missing", "foreign", "owner_swap", "ordinal_swap", "flattened")
        ),
        *(
            ReferenceCase(f"consumed-binding-{kind}", consumed_binding_evidence=kind, **base_one)
            for kind in ("missing", "duplicate", "foreign")
        ),
        ReferenceCase("capability-missing", capability="missing", **base_one),
        ReferenceCase("capability-drift", capability="drift", **base_one),
        ReferenceCase("retention-unknown", retention="unknown", **base_one),
        ReferenceCase("retention-enabled", retention="enabled", **base_one),
        ReferenceCase("prompt-drift", prompt="drift", **base_one),
        ReferenceCase("model-config-drift", model_route="config_drift", **base_one),
        ReferenceCase("wrong-model-role", model_route="wrong_role", **base_one),
        ReferenceCase("fallback-model-route", model_route="fallback", **base_one),
        ReferenceCase("failed-record-bound", failure_evidence="bound", **base_one),
        ReferenceCase("failed-record-record-id-only", failure_evidence="record_id_only", **base_one),
        ReferenceCase("failed-record-foreign", failure_evidence="foreign", **base_one),
        ReferenceCase("cleanup-pre-failed", pre_cleanup="failed", **base_one),
        ReferenceCase("cleanup-pre-missing", pre_cleanup="missing", **base_one),
        ReferenceCase("cleanup-pre-duplicate", pre_cleanup="duplicate", **base_one),
        ReferenceCase("cleanup-post-failed", post_cleanup="failed", **base_one),
        ReferenceCase("cleanup-post-contradictory", post_cleanup="contradictory", **base_one),
        ReferenceCase("workframe-limit-exact", workframe_bytes=MAX_WORKFRAME_BYTES, **base_one),
        ReferenceCase("workframe-limit-over", workframe_bytes=MAX_WORKFRAME_BYTES + 1, **base_one),
    )
    return directed + _finite_envelope_cases()


def _finite_envelope_cases() -> tuple[ReferenceCase, ...]:
    """Enumerate the bounded structural, dependency, and repair envelopes."""
    cases: list[ReferenceCase] = []
    for target_count in range(1, MAX_DATUMS + 1):
        targets = tuple(f"d{index}" for index in range(target_count))
        for partition_index, groups in enumerate(_partitions_up_to_two(targets)):
            atomic_options = (groups,) if len(groups) == 1 else (groups, (targets,))
            for atomic_index, atomic_groups in enumerate(atomic_options):
                cases.append(
                    ReferenceCase(
                        f"envelope-shape-{target_count}-{partition_index}-{atomic_index}",
                        targets,
                        tuple(ReferenceGroup(group) for group in groups),
                        atomic_groups,
                    )
                )
    for target_count in range(2, MAX_DATUMS + 1):
        targets = tuple(f"d{index}" for index in range(target_count))
        groups = ((targets[0],), targets[1:])
        possible_edges = tuple(
            (targets[left], targets[right]) for left in range(target_count) for right in range(left + 1, target_count)
        )
        for mask in range(1 << len(possible_edges)):
            dependencies = tuple(edge for index, edge in enumerate(possible_edges) if mask & (1 << index))
            cases.append(
                ReferenceCase(
                    f"envelope-dag-{target_count}-{mask}",
                    targets,
                    (ReferenceGroup(groups[0], ("failed",)), ReferenceGroup(groups[1])),
                    groups,
                    dependencies,
                )
            )
    for max_repairs in range(3):
        for pass_round in range(max_repairs + 1):
            evaluations = (True,) * pass_round + (False,)
            repairs = (("d0",),) * pass_round
            cases.append(
                ReferenceCase(
                    f"envelope-repair-{max_repairs}-pass-{pass_round}",
                    ("d0",),
                    (ReferenceGroup(("d0",), evaluations=evaluations, repair_keys=repairs),),
                    (("d0",),),
                    max_repairs=max_repairs,
                )
            )
        cases.append(
            ReferenceCase(
                f"envelope-repair-{max_repairs}-exhausted",
                ("d0",),
                (
                    ReferenceGroup(
                        ("d0",),
                        evaluations=(True,) * (max_repairs + 1),
                        repair_keys=(("d0",),) * max_repairs,
                    ),
                ),
                (("d0",),),
                max_repairs=max_repairs,
            )
        )
    return tuple(cases)


def _partitions_up_to_two(values: tuple[str, ...]) -> tuple[tuple[tuple[str, ...], ...], ...]:
    if len(values) == 1:
        return ((values,),)
    partitions: list[tuple[tuple[str, ...], ...]] = [(values,)]
    first = values[0]
    tail = values[1:]
    for mask in range(1 << len(tail)):
        left = (first, *(value for index, value in enumerate(tail) if mask & (1 << index)))
        right = tuple(value for index, value in enumerate(tail) if not mask & (1 << index))
        if right:
            partitions.append((left, right))
    return tuple(partitions)


def case_by_name(name: str) -> ReferenceCase:
    return next(case for case in finite_reference_cases() if case.name == name)


def canonical_corpus_bytes() -> bytes:
    records = [
        {"case": asdict(case), "result": asdict(reduce_reference(case))}
        for case in sorted(finite_reference_cases(), key=lambda item: item.name)
    ]
    return json.dumps(
        {
            "cases": records,
            "generator_version": GENERATOR_VERSION,
            "reference_model_version": REFERENCE_MODEL_VERSION,
            "schema_version": "phase8-reference-corpus/v2",
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def reference_manifest() -> dict[str, object]:
    cases = finite_reference_cases()
    return {
        "actual_event_count": sum(_event_count(case) for case in cases),
        "canonical_serialization": "UTF8_compact_sorted_key_JSON_complete_corpus_no_trailing_newline",
        "case_count": len(cases),
        "canonical_trace_count": len(cases),
        "corpus_sha256": hashlib.sha256(canonical_corpus_bytes()).hexdigest(),
        "dependency_envelope_case_count": sum(case.name.startswith("envelope-dag-") for case in cases),
        "directed_case_count": sum(not case.name.startswith("envelope-") for case in cases),
        "generator_version": GENERATOR_VERSION,
        "repair_envelope_case_count": sum(case.name.startswith("envelope-repair-") for case in cases),
        "reference_model_version": REFERENCE_MODEL_VERSION,
        "schema_version": "phase8-reference-manifest/v2",
        "structural_envelope_case_count": sum(case.name.startswith("envelope-shape-") for case in cases),
        "terminal_case_counts": {
            terminal: sum(reduce_reference(case).invocation == terminal for case in cases)
            for terminal in ("completed", "failed", "cancelled", "lost", "inconsistent")
        },
    }


def _event_count(case: ReferenceCase) -> int:
    return (
        sum(len(group.terminal_events) + len(group.evaluations) + len(group.repair_keys) for group in case.groups) + 2
    )


@dataclass(frozen=True, slots=True)
class Case:
    """Compatibility projection retained for the early Phase 8 release test."""

    groups: tuple[tuple[str, ...], ...]
    atomic_groups: tuple[tuple[str, ...], ...]
    failed_groups: tuple[int, ...] = ()
    embargo: bool = False


def reduce(case: Case) -> tuple[str, ...]:
    groups = tuple(
        ReferenceGroup(members, ("failed",) if index in case.failed_groups else ("succeeded",))
        for index, members in enumerate(case.groups)
    )
    result = reduce_reference(
        ReferenceCase(
            "compatibility",
            tuple(key for group in case.groups for key in group),
            groups,
            case.atomic_groups,
            pre_cleanup="missing" if case.embargo else "verified",
        )
    )
    return result.released
