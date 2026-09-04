# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest

from anonymizer.engine.execution.graph import (
    _AtomicGroup,
    _DatumDependency,
    _DatumId,
    _DatumPurpose,
    _ProtectionGraph,
    _RewriteGroup,
    _TextDatum,
)
from anonymizer.engine.execution.phase8_admission import (
    _compile_phase8_plan,
    _is_admitted_phase8_plan,
    _Phase8Plan,
)
from anonymizer.engine.execution.phase8_runtime import (
    _Phase8FaultKind,
    _Phase8GroupOutcome,
    _Phase8OperationFault,
    _Phase8Reason,
    _run_group_operation,
)
from anonymizer.engine.execution.phase8_service import _Phase8GroupedRewriteProtectionService
from anonymizer.engine.execution.phase8_validation import _Phase8Metric
from tests.engine.execution.phase8_reference_model import (
    MAX_WORKFRAME_BYTES,
    ReferenceCase,
    ReferenceGroup,
    finite_reference_cases,
    reduce_reference,
)


def _stable_runtime_case(case: ReferenceCase) -> bool:
    result = reduce_reference(case)
    return (
        result.admission == "admitted"
        and case.mention_evidence == "exact"
        and case.context_evidence == "exact"
        and case.consumed_binding_evidence == "exact"
        and case.capability == "stable"
        and case.retention == "disabled"
        and case.prompt == "stable"
        and case.model_route == "exact"
        and case.failure_evidence in {"none", "bound"}
        and case.pre_cleanup == "verified"
        and case.post_cleanup == "verified"
        and case.workframe_bytes <= MAX_WORKFRAME_BYTES
    )


_STABLE_CASES = tuple(case for case in finite_reference_cases() if _stable_runtime_case(case))
_STRUCTURAL_CASES = tuple(case for case in finite_reference_cases() if case.name.startswith("envelope-shape-"))
_REPAIR_CASES = tuple(case for case in finite_reference_cases() if case.name.startswith("envelope-repair-"))


@pytest.mark.parametrize("case", _STRUCTURAL_CASES, ids=lambda case: case.name)
def test_production_admission_matches_every_bounded_reference_shape(case: ReferenceCase) -> None:
    graph, _ids = _production_graph(case)
    production = _compile_phase8_plan(graph, max_repairs=case.max_repairs)

    assert isinstance(production, _Phase8Plan)
    assert _is_admitted_phase8_plan(production)
    assert tuple(tuple(member.value for member in group.members) for group in production.groups) == tuple(
        group.members
        for group in sorted(case.groups, key=lambda group: min(case.targets.index(key) for key in group.members))
    )


@pytest.mark.parametrize("case", _STABLE_CASES, ids=lambda case: case.name)
def test_production_runtime_and_phase4_match_every_stable_reference_trace(case: ReferenceCase) -> None:
    expected = reduce_reference(case)
    graph, ids = _production_graph(case)
    plan = _compile_phase8_plan(graph, max_repairs=case.max_repairs)
    assert isinstance(plan, _Phase8Plan)
    reference_by_members = {frozenset(group.members): group for group in case.groups}
    operations = tuple(
        _operation(
            reference_by_members[frozenset(member.value for member in manifest.members)],
            ids,
            case.max_repairs,
        )
        for manifest in plan.groups
    )
    execution = _Phase8GroupedRewriteProtectionService().run_lifecycle(
        groups=tuple(manifest.members for manifest in plan.groups),
        atomic_groups=tuple(tuple(ids[key] for key in group) for group in case.atomic_groups),
        dependencies=tuple((ids[left], ids[right]) for left, right in case.dependencies),
        phase7_released=tuple((ids[key], f"baseline-{key}") for key in case.targets),
        phase7_cleanup_verified=True,
        phase7_global_embargo=False,
        operations=operations,
    )

    assert execution.terminal_group_states == expected.group_states
    assert tuple(cast(_DatumId, member).value for member, _value in execution.released) == expected.released
    assert execution.global_embargo is (expected.invocation in {"cancelled", "lost", "inconsistent"})


@pytest.mark.parametrize("case", _REPAIR_CASES, ids=lambda case: case.name)
def test_production_repair_schedule_matches_every_bounded_reference_trace(case: ReferenceCase) -> None:
    expected = reduce_reference(case)
    member = _DatumId("d0")
    operation = _operation(case.groups[0], {"d0": member}, case.max_repairs)

    outcome = operation((member,), {member: "baseline-d0"})

    assert isinstance(outcome, _Phase8GroupOutcome)
    assert outcome.state == expected.group_states[0]
    assert outcome.repair_iterations == len(case.groups[0].repair_keys)


def test_every_reference_case_is_bound_to_a_production_conformance_class() -> None:
    classified: dict[str, str] = {}
    for case in finite_reference_cases():
        expected = reduce_reference(case)
        if _stable_runtime_case(case):
            classified[case.name] = "runtime-and-phase4"
        elif expected.admission == "rejected" or not case.strict:
            classified[case.name] = "admission"
        elif case.pre_cleanup != "verified" or case.post_cleanup != "verified":
            classified[case.name] = "cleanup"
        elif case.workframe_bytes > MAX_WORKFRAME_BYTES:
            classified[case.name] = "bounded-backend"
        else:
            classified[case.name] = "authority-and-reconciliation"

    assert tuple(classified) == tuple(case.name for case in finite_reference_cases())
    assert set(classified.values()) == {
        "admission",
        "authority-and-reconciliation",
        "bounded-backend",
        "cleanup",
        "runtime-and-phase4",
    }


def _production_graph(case: ReferenceCase) -> tuple[_ProtectionGraph, dict[str, _DatumId]]:
    referenced = set(case.targets)
    for group in case.groups:
        referenced.update(group.members)
        referenced.update(group.result_keys or ())
        for repair_keys in group.repair_keys:
            referenced.update(repair_keys)
    ids = {key: _DatumId(key) for key in referenced}
    graph = _ProtectionGraph(
        datums=tuple(_TextDatum(ids[key], f"text-{key}", _DatumPurpose.TARGET) for key in case.targets),
        links=(),
        context_scopes=(),
        coherence_scopes=(),
        atomic_groups=tuple(_AtomicGroup(tuple(ids[key] for key in group)) for group in case.atomic_groups),
        dependencies=tuple(_DatumDependency(ids[left], ids[right]) for left, right in case.dependencies),
        rewrite_groups=tuple(_RewriteGroup(tuple(ids[key] for key in group.members)) for group in case.groups),
    )
    return graph, ids


def _operation(
    group: ReferenceGroup, ids: dict[str, _DatumId], max_repairs: int
) -> Callable[[tuple[object, ...], dict[object, str]], _Phase8GroupOutcome]:
    def run(members: tuple[object, ...], baselines: dict[object, str]) -> _Phase8GroupOutcome:
        first = group.terminal_events[0] if group.terminal_events else "inconsistent"

        def analyze() -> tuple[bool, bool]:
            kind = {
                "failed": _Phase8FaultKind.FAILED,
                "cancelled": _Phase8FaultKind.CANCELLED,
                "lost": _Phase8FaultKind.LOST,
                "inconsistent": _Phase8FaultKind.INCONSISTENT,
                "global_inconsistent": _Phase8FaultKind.INCONSISTENT,
            }.get(first)
            if kind is not None:
                raise _Phase8OperationFault(
                    kind,
                    _Phase8Reason.INVOCATION_INCONSISTENT,
                    invocation_global=first == "global_inconsistent",
                    trusted_stop=first == "cancelled",
                )
            return False, False

        result_keys = group.members if group.result_keys is None else group.result_keys
        evaluations = iter(group.evaluations)

        return _run_group_operation(
            members,
            baselines,
            analyze=analyze,
            rewrite=lambda _current: {ids[key]: f"revision-{key}" for key in result_keys},
            evaluate=lambda _current: _metric(next(evaluations, None)),
            repair=lambda _current, round_number: {
                ids[key]: f"repair-{round_number}-{key}" for key in group.repair_keys[round_number - 1]
            },
            max_repairs=max_repairs,
        )

    return run


def _metric(needs_repair: bool | None) -> _Phase8Metric:
    if needs_repair is None:
        return cast(_Phase8Metric, None)
    return _Phase8Metric(1.0, float(needs_repair), float(needs_repair), needs_repair, needs_repair)
