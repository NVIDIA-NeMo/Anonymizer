# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from typing import cast

from pytest import MonkeyPatch

import anonymizer.engine.execution.phase8_contract as phase8_contract
from anonymizer.engine.execution.accounting_outcomes import _AccountingResult
from anonymizer.engine.execution.graph import (
    _AtomicGroup,
    _DatumId,
    _DatumPurpose,
    _ProtectionGraph,
    _RewriteGroup,
    _TextDatum,
)
from anonymizer.engine.execution.phase7_application import _AppliedDatum
from anonymizer.engine.execution.phase7_runtime import (
    _Phase7CleanupAttestation,
    _Phase7Execution,
    _Phase7Phase4Evidence,
)
from anonymizer.engine.execution.phase8_admission import (
    _compile_phase8_plan,
    _Phase8AdmissionCode,
    _Phase8Plan,
    _Phase8Rejected,
)
from anonymizer.engine.execution.phase8_contract import (
    _canonical_digest,
    _compile_phase8_contract,
    _is_admitted_phase8_contract,
    _load_phase8_contract,
    _Phase8GroupedRewriteContract,
)
from anonymizer.engine.execution.phase8_runtime import _run_group_operation
from anonymizer.engine.execution.phase8_service import _Phase8GroupedRewriteProtectionService
from anonymizer.engine.execution.phase8_validation import _evaluate_metrics
from tests.engine.execution.phase8_reference_model import Case, reduce


def _graph(groups: tuple[tuple[int, ...], ...]) -> _ProtectionGraph:
    ids = tuple(_DatumId(str(index)) for index in range(2))
    return _ProtectionGraph(
        tuple(_TextDatum(identifier, f"text-{index}", _DatumPurpose.TARGET) for index, identifier in enumerate(ids)),
        (),
        (),
        (),
        (_AtomicGroup(ids),),
        rewrite_groups=tuple(_RewriteGroup(tuple(ids[index] for index in group)) for group in groups),
    )


def test_phase8_contract_loader_admits_exact_frozen_contract() -> None:
    contract = _load_phase8_contract()
    assert _is_admitted_phase8_contract(contract)
    assert isinstance(contract, _Phase8GroupedRewriteContract)
    assert contract.digest == "597a410aee8cb8ca428e82737f385213ce9ce47eae216caea68ebc2f9907d227"
    assert dict(contract.limits)["max_repair_iterations"] == 3


def _digest_valid_envelope() -> dict[str, object]:
    path = Path("src/anonymizer/engine/execution/phase8_grouped_rewrite_contract.json")
    envelope = copy.deepcopy(json.loads(path.read_text()))
    assert isinstance(envelope["contract"], dict)
    envelope["digest"] = _canonical_digest(envelope["contract"])
    return envelope


def test_phase8_contract_loader_rejects_digest_valid_nested_shape_mutations(monkeypatch: MonkeyPatch) -> None:
    envelope = _digest_valid_envelope()
    contract = envelope["contract"]
    assert isinstance(contract, dict)
    contract["scope"]["unexpected"] = True
    envelope["digest"] = _canonical_digest(contract)
    monkeypatch.setattr(phase8_contract, "_DIGEST", envelope["digest"])

    assert not _is_admitted_phase8_contract(_compile_phase8_contract(envelope))


def test_phase8_contract_loader_rejects_digest_valid_type_and_limit_mutations(monkeypatch: MonkeyPatch) -> None:
    type_mutation = _digest_valid_envelope()
    type_contract = type_mutation["contract"]
    assert isinstance(type_contract, dict)
    type_contract["scheduling_and_limits"]["max_members_per_rewrite_group"] = "4"
    type_mutation["digest"] = _canonical_digest(type_contract)

    limit_mutation = _digest_valid_envelope()
    limit_contract = limit_mutation["contract"]
    assert isinstance(limit_contract, dict)
    del limit_contract["phase8_backend_capability"]["required_artifacts"]
    limit_mutation["digest"] = _canonical_digest(limit_contract)

    monkeypatch.setattr(phase8_contract, "_DIGEST", type_mutation["digest"])
    assert not _is_admitted_phase8_contract(_compile_phase8_contract(type_mutation))
    monkeypatch.setattr(phase8_contract, "_DIGEST", limit_mutation["digest"])
    assert not _is_admitted_phase8_contract(_compile_phase8_contract(limit_mutation))


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
        members,
        baselines,
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
        (member,),
        {member: "baseline"},
        analyze=lambda: (True, False),
        rewrite=lambda values: values,
        evaluate=lambda _values: (_ for _ in ()).throw(AssertionError("no evaluation")),
        repair=lambda values, _round: values,
        max_repairs=0,
    )
    assert outcome.state == "failed"


def test_phase8_zero_utility_only_scores_one_for_an_exact_baseline() -> None:
    non_baseline = _evaluate_metrics(
        (), (), repair_any_high=False, repair_threshold=0.0, utility_floor=0.5, exact_baseline=False
    )
    baseline = _evaluate_metrics(
        (), (), repair_any_high=False, repair_threshold=0.0, utility_floor=0.5, exact_baseline=True
    )
    assert non_baseline is not None
    assert baseline is not None
    assert non_baseline.utility_score == 0.0
    assert non_baseline.needs_repair
    assert baseline.utility_score == 1.0
    assert not baseline.needs_repair


def test_private_phase8_service_only_returns_a_complete_group_candidate() -> None:
    members = (object(), object())
    metric = _evaluate_metrics((), ((1, 1.0),), repair_any_high=False, repair_threshold=0.0, utility_floor=0.5)
    assert metric is not None
    result = _Phase8GroupedRewriteProtectionService().run_group(
        members,
        {members[0]: "one", members[1]: "two"},
        analyze=lambda: (False, False),
        rewrite=lambda values: values,
        evaluate=lambda _values: metric,
        repair=lambda values, _round: values,
        max_repairs=0,
    )
    assert result == ((members[0], "one"), (members[1], "two"))


def test_phase8_lifecycle_requires_a_released_phase7_baseline_and_withholds_the_atomic_group() -> None:
    """A failed complete group may not release its otherwise-successful sibling."""
    service = _Phase8GroupedRewriteProtectionService()
    first, second = _DatumId("first"), _DatumId("second")
    execution = service.run_lifecycle(
        groups=((first,), (second,)),
        atomic_groups=((first, second),),
        dependencies=(),
        phase7_released=((first, "one"), (second, "two")),
        phase7_cleanup_verified=True,
        phase7_global_embargo=False,
        operations=(
            lambda members, baselines: service.run_group(
                members,
                baselines,
                analyze=lambda: (False, False),
                rewrite=lambda values: values,
                evaluate=lambda _values: _metric(),
                repair=lambda values, _round: values,
                max_repairs=0,
            ),
            lambda members, baselines: service.run_group(
                members,
                baselines,
                analyze=lambda: (False, False),
                rewrite=lambda _values: {},
                evaluate=lambda _values: _metric(),
                repair=lambda values, _round: values,
                max_repairs=0,
            ),
        ),
    )
    assert execution.released == ()
    assert execution.global_embargo is False
    assert execution.cleanup_verified
    assert execution.terminal_group_states == ("succeeded", "failed")


def test_phase8_lifecycle_embargoes_before_dispatch_when_phase7_handoff_is_not_clean() -> None:
    service = _Phase8GroupedRewriteProtectionService()
    datum = _DatumId("only")
    dispatched = False

    def operation(_members: tuple[object, ...], _baselines: dict[object, str]) -> tuple[tuple[object, str], ...] | None:
        nonlocal dispatched
        dispatched = True
        raise AssertionError("must not dispatch")

    execution = service.run_lifecycle(
        groups=((datum,),),
        atomic_groups=((datum,),),
        dependencies=(),
        phase7_released=((datum, "baseline"),),
        phase7_cleanup_verified=False,
        phase7_global_embargo=False,
        operations=(operation,),
    )
    assert not dispatched
    assert execution.global_embargo
    assert execution.released == ()


def test_phase8_lifecycle_consumes_only_the_phase7_released_baseline_handoff() -> None:
    service = _Phase8GroupedRewriteProtectionService()
    datum = _DatumId("only")
    cleanup = _Phase7CleanupAttestation("phase7-cleanup-attestation/v1", True, 0, 0, True, 0, False)
    phase7 = _Phase7Execution(
        (),
        cleanup,
        _Phase7Phase4Evidence((), cast(_AccountingResult[object], object()), cleanup, False),
        (_AppliedDatum(datum, "baseline", True),),
    )
    execution = service.run_from_phase7_execution(
        groups=((datum,),),
        atomic_groups=((datum,),),
        dependencies=(),
        phase7=phase7,
        operations=(
            lambda members, baselines: service.run_group(
                members,
                baselines,
                analyze=lambda: (True, True),
                rewrite=lambda values: values,
                evaluate=lambda _values: _metric(),
                repair=lambda values, _round: values,
                max_repairs=0,
            ),
        ),
    )
    assert execution.released == ((datum, "baseline"),)


def _metric():
    metric = _evaluate_metrics((), ((1, 1.0),), repair_any_high=False, repair_threshold=0.0, utility_floor=0.5)
    assert metric is not None
    return metric


def test_phase8_reference_model_is_pure_and_withholds_an_atomic_group() -> None:
    tree = ast.parse(Path("tests/engine/execution/phase8_reference_model.py").read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(alias.name for alias in node.names)
    assert not {"pandas", "data_designer"}.intersection(names)
    assert reduce(Case((("a", "b"), ("c",)), (("a", "b"), ("c",)), (0,))) == ("c",)
