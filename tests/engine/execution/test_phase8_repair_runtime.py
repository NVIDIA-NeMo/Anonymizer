# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from anonymizer.engine.execution.graph import _DatumId
from anonymizer.engine.execution.phase8_ndd_backend import _Phase8Operation
from anonymizer.engine.execution.phase8_runtime import (
    _GroupBlocked,
    _GroupFailed,
    _Phase8Reason,
    _run_group_operation,
    _StageBlocked,
)
from anonymizer.engine.execution.phase8_service import (
    _backend_group_operation,
    _Phase8GroupedRewriteProtectionService,
    _Phase8GroupInput,
)
from anonymizer.engine.execution.phase8_validation import _evaluate_metrics


def _metric(*, needs_repair: bool):
    return _evaluate_metrics(
        (("high", 1.0, needs_repair),), ((1, 1.0),), repair_any_high=True, repair_threshold=0.0, utility_floor=0.5
    )


def test_missing_phase7_baseline_blocks_before_analysis_call() -> None:
    members = (object(), object())
    calls = 0

    def analyze() -> tuple[bool, bool]:
        nonlocal calls
        calls += 1
        return False, False

    outcome = _run_group_operation(
        members,
        {members[0]: "baseline"},
        analyze=analyze,
        rewrite=lambda values: values,
        evaluate=lambda _values: _metric(needs_repair=False),
        repair=lambda values, _round: values,
        max_repairs=0,
    )

    assert isinstance(outcome.terminal, _GroupBlocked)
    assert calls == 0
    assert outcome.ledger.reason(outcome.ledger.plan.stages[1]) == "prerequisite"


def test_lifecycle_missing_baseline_closes_compiled_ledger_without_provider_call() -> None:
    first, second = _DatumId("first"), _DatumId("second")
    provider_calls = 0
    outcomes = []

    class ForbiddenBackend:
        def run_operation(self, _operation: _Phase8Operation, _request: dict[str, object]):
            nonlocal provider_calls
            provider_calls += 1
            raise AssertionError("missing baselines must block before provider dispatch")

    operation = _backend_group_operation(
        ForbiddenBackend(),
        _Phase8GroupInput(
            {first: "one", second: "two"},
            {first: False, second: False},
            max_repairs=0,
        ),
    )

    def capture(members: tuple[object, ...], baselines: dict[object, str]):
        outcome = operation(members, baselines)
        outcomes.append(outcome)
        return outcome

    execution = _Phase8GroupedRewriteProtectionService().run_lifecycle(
        groups=((first, second),),
        atomic_groups=((first, second),),
        dependencies=(),
        phase7_released=((first, "one"),),
        phase7_cleanup_verified=True,
        phase7_global_embargo=False,
        operations=(capture,),
    )

    assert execution.terminal_group_states == ("blocked",)
    assert execution.released == ()
    assert provider_calls == 0
    assert len(outcomes) == 1
    assert outcomes[0].ledger.is_closed
    assert outcomes[0].ledger.reason(outcomes[0].ledger.plan.stages[0]) == "missing_baseline"
    assert all(outcomes[0].ledger.attempt_count(stage) == 0 for stage in outcomes[0].ledger.plan.stages)


def test_final_evaluation_needing_repair_fails_without_candidate() -> None:
    member = object()
    outcome = _run_group_operation(
        (member,),
        {member: "baseline"},
        analyze=lambda: (False, False),
        rewrite=lambda values: values,
        evaluate=lambda _values: _metric(needs_repair=True),
        repair=lambda values, _round: values,
        max_repairs=0,
    )

    assert isinstance(outcome.terminal, _GroupFailed)
    assert outcome.terminal.code is _Phase8Reason.REPAIR_EXHAUSTED
    assert outcome.revisions is None
    assert not isinstance(outcome.ledger.terminal(outcome.ledger.plan.stages[-1]), _StageBlocked)


def test_every_repair_is_followed_by_complete_group_evaluation() -> None:
    member = object()
    evaluations = 0

    def evaluate(_values: dict[object, str]):
        nonlocal evaluations
        evaluations += 1
        return _metric(needs_repair=evaluations == 1)

    outcome = _run_group_operation(
        (member,),
        {member: "baseline"},
        analyze=lambda: (False, False),
        rewrite=lambda values: values,
        evaluate=evaluate,
        repair=lambda values, _round: values,
        max_repairs=1,
    )

    assert outcome.state == "succeeded"
    assert evaluations == 2
    assert all(outcome.ledger.attempt_count(stage) == 1 for stage in outcome.ledger.plan.stages)


@pytest.mark.parametrize("rounds", range(4))
def test_exact_call_and_attempt_count_through_each_compiled_repair_bound(rounds: int) -> None:
    member = object()
    calls = 0
    evaluations = 0

    def analyze() -> tuple[bool, bool]:
        nonlocal calls
        calls += 1
        return False, False

    def rewrite(values: dict[object, str]) -> dict[object, str]:
        nonlocal calls
        calls += 1
        return values

    def evaluate(_values: dict[object, str]):
        nonlocal calls, evaluations
        calls += 1
        evaluations += 1
        return _metric(needs_repair=evaluations <= rounds)

    def repair(values: dict[object, str], _round: int) -> dict[object, str]:
        nonlocal calls
        calls += 1
        return values

    outcome = _run_group_operation(
        (member,),
        {member: "baseline"},
        analyze=analyze,
        rewrite=rewrite,
        evaluate=evaluate,
        repair=repair,
        max_repairs=rounds,
    )

    assert outcome.state == "succeeded"
    assert calls == 3 + 2 * rounds
    assert len(outcome.ledger._attempts) == 4 + 2 * rounds
    assert outcome.ledger.is_closed


@pytest.mark.parametrize("rounds", range(4))
def test_repair_exhaustion_closes_the_exact_compiled_route(rounds: int) -> None:
    member = object()
    calls = 0

    def counted_analyze() -> tuple[bool, bool]:
        nonlocal calls
        calls += 1
        return False, False

    def counted_revision(values: dict[object, str], _round: int | None = None) -> dict[object, str]:
        nonlocal calls
        calls += 1
        return values

    def counted_evaluation(_values: dict[object, str]):
        nonlocal calls
        calls += 1
        return _metric(needs_repair=True)

    outcome = _run_group_operation(
        (member,),
        {member: "baseline"},
        analyze=counted_analyze,
        rewrite=counted_revision,
        evaluate=counted_evaluation,
        repair=counted_revision,
        max_repairs=rounds,
    )

    assert isinstance(outcome.terminal, _GroupFailed)
    assert outcome.terminal.code is _Phase8Reason.REPAIR_EXHAUSTED
    assert outcome.revisions is None
    assert calls == 3 + 2 * rounds
    assert outcome.ledger.is_closed


def test_local_backend_failure_continues_to_a_disconnected_group() -> None:
    first, second = _DatumId("first"), _DatumId("second")
    second_calls = 0

    class LocalFailureBackend:
        def run_operation(self, operation: _Phase8Operation, _request: dict[str, object]):
            return _Result(operation, failed=True, failure_kind="local_failure")

    class IdentityBackend:
        def run_operation(self, operation: _Phase8Operation, request: dict[str, object]):
            nonlocal second_calls
            second_calls += 1
            members = request["members"]
            assert isinstance(members, list)
            return _Result(
                operation,
                payload={
                    "analyzed_member_tokens": [member["member_token"] for member in members],
                    "consumed_context_binding_tokens": [],
                    "privacy_obligations": [],
                    "utility_obligations": [],
                },
            )

    execution = _Phase8GroupedRewriteProtectionService().run_lifecycle(
        groups=((first,), (second,)),
        atomic_groups=((first,), (second,)),
        dependencies=(),
        phase7_released=((first, "one"), (second, "two")),
        phase7_cleanup_verified=True,
        phase7_global_embargo=False,
        operations=(
            _backend_group_operation(LocalFailureBackend(), _Phase8GroupInput({first: "one"}, {first: False})),
            _backend_group_operation(IdentityBackend(), _Phase8GroupInput({second: "two"}, {second: False})),
        ),
    )

    assert execution.terminal_group_states == ("failed", "succeeded")
    assert execution.released == ((second, "two"),)
    assert not execution.global_embargo
    assert second_calls == 1


def test_transport_loss_stops_later_group_dispatch_and_embargoes_release() -> None:
    first, second = _DatumId("first"), _DatumId("second")
    second_calls = 0

    class LostBackend:
        def run_operation(self, _operation: _Phase8Operation, _request: dict[str, object]):
            raise RuntimeError

    class ForbiddenBackend:
        def run_operation(self, _operation: _Phase8Operation, _request: dict[str, object]):
            nonlocal second_calls
            second_calls += 1
            raise AssertionError

    execution = _Phase8GroupedRewriteProtectionService().run_lifecycle(
        groups=((first,), (second,)),
        atomic_groups=((first,), (second,)),
        dependencies=(),
        phase7_released=((first, "one"), (second, "two")),
        phase7_cleanup_verified=True,
        phase7_global_embargo=False,
        operations=(
            _backend_group_operation(LostBackend(), _Phase8GroupInput({first: "one"}, {first: False})),
            _backend_group_operation(ForbiddenBackend(), _Phase8GroupInput({second: "two"}, {second: False})),
        ),
    )

    assert execution.terminal_group_states == ("lost", "blocked")
    assert execution.released == ()
    assert execution.global_embargo
    assert second_calls == 0


class _Result:
    def __init__(
        self,
        operation: _Phase8Operation,
        payload: object = None,
        *,
        failed: bool = False,
        failure_kind: str | None = None,
    ) -> None:
        self.operation = operation
        self.payload = payload
        self.failed = failed
        self.failure_kind = failure_kind
