# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from anonymizer.engine.execution.graph import _AtomicGroup, _DatumId, _DatumPurpose, _ProtectionGraph, _RewriteGroup, _TextDatum
from anonymizer.engine.execution.phase8_admission import _compile_phase8_plan, _Phase8AdmissionCode, _Phase8Plan, _Phase8Rejected
from anonymizer.engine.execution.phase8_contract import _is_admitted_phase8_contract, _load_phase8_contract, _Phase8GroupedRewriteContract
from anonymizer.engine.execution.phase8_runtime import _run_group_operation
from anonymizer.engine.execution.phase8_validation import _evaluate_metrics


def _graph(groups: tuple[tuple[int, ...], ...]) -> _ProtectionGraph:
    ids = tuple(_DatumId(str(index)) for index in range(2))
    return _ProtectionGraph(
        tuple(_TextDatum(identifier, f"text-{index}", _DatumPurpose.TARGET) for index, identifier in enumerate(ids)),
        (), (), (),
        (_AtomicGroup(ids),),
        rewrite_groups=tuple(_RewriteGroup(tuple(ids[index] for index in group)) for group in groups),
    )


def test_phase8_contract_loader_admits_exact_frozen_contract() -> None:
    contract = _load_phase8_contract()
    assert _is_admitted_phase8_contract(contract)
    assert isinstance(contract, _Phase8GroupedRewriteContract)
    assert contract.digest == "597a410aee8cb8ca428e82737f385213ce9ce47eae216caea68ebc2f9907d227"
    assert dict(contract.limits)["max_repair_iterations"] == 3


def test_phase8_admission_requires_one_flat_exact_target_partition() -> None:
    accepted = _compile_phase8_plan(_graph(((1, 0),)))
    assert isinstance(accepted, _Phase8Plan)
    assert tuple(tuple(member.value for member in group.members) for group in accepted.groups) == (("1", "0"),)
    rejected = _compile_phase8_plan(_graph(((0,),)))
    assert isinstance(rejected, _Phase8Rejected)
    assert rejected.code is _Phase8AdmissionCode.COVERAGE_GAP


def test_phase8_runtime_never_adopts_a_partial_group_repair() -> None:
    members = (object(), object())
    baselines = {members[0]: "one", members[1]: "two"}
    metric = _evaluate_metrics((), ((1, 0.0),), repair_any_high=False, repair_threshold=0.0, utility_floor=0.5)
    assert metric is not None
    outcome = _run_group_operation(
        members, baselines,
        analyze=lambda: (False, False),
        rewrite=lambda values: values,
        evaluate=lambda _values: metric,
        repair=lambda values, _round: {next(iter(values)): "only-one"},
        max_repairs=1,
    )
    assert outcome.state == "failed"
    assert outcome.revisions is None


def test_phase8_zero_obligation_route_requires_all_guards() -> None:
    member = object()
    outcome = _run_group_operation(
        (member,), {member: "baseline"},
        analyze=lambda: (True, False),
        rewrite=lambda values: values,
        evaluate=lambda _values: (_ for _ in ()).throw(AssertionError("no evaluation")),
        repair=lambda values, _round: values,
        max_repairs=0,
    )
    assert outcome.state == "failed"
