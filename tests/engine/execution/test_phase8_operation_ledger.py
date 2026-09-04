# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import anonymizer.engine.execution.phase8_runtime as phase8_runtime


def _plan(rounds: int) -> phase8_runtime._Phase8GroupOperationPlan:
    plan = phase8_runtime._compile_group_operation_plan(rounds, 3)
    assert plan is not None
    return plan


@pytest.mark.parametrize(
    ("rounds", "expected"),
    [
        (0, ("validate-baselines", "analyze", "rewrite", "evaluate-0")),
        (1, ("validate-baselines", "analyze", "rewrite", "evaluate-0", "repair-1", "evaluate-1")),
        (
            2,
            (
                "validate-baselines",
                "analyze",
                "rewrite",
                "evaluate-0",
                "repair-1",
                "evaluate-1",
                "repair-2",
                "evaluate-2",
            ),
        ),
        (
            3,
            (
                "validate-baselines",
                "analyze",
                "rewrite",
                "evaluate-0",
                "repair-1",
                "evaluate-1",
                "repair-2",
                "evaluate-2",
                "repair-3",
                "evaluate-3",
            ),
        ),
    ],
)
def test_phase8_operation_plan_freezes_exact_stage_vectors(rounds: int, expected: tuple[str, ...]) -> None:
    plan = _plan(rounds)

    assert tuple(stage.name for stage in plan.stages) == expected
    assert len(plan.stages) == 4 + 2 * rounds


def test_zero_obligation_closes_every_conditional_stage_without_attempts() -> None:
    ledger = phase8_runtime._Phase8OperationLedger(_plan(2))
    ledger.succeed(phase8_runtime._Phase8Stage.validate_baselines())
    ledger.succeed(phase8_runtime._Phase8Stage.analyze())
    ledger.close_zero_route()

    assert isinstance(ledger.terminal(phase8_runtime._Phase8Stage.rewrite()), phase8_runtime._StageBlocked)
    assert ledger.reason(phase8_runtime._Phase8Stage.rewrite()) == "route_not_selected"
    assert all(ledger.attempt_count(stage) == 0 for stage in ledger.plan.stages[2:])
    assert ledger.is_closed


def test_terminal_is_absorbing_and_failure_blocks_descendants() -> None:
    ledger = phase8_runtime._Phase8OperationLedger(_plan(1))
    ledger.succeed(phase8_runtime._Phase8Stage.validate_baselines())
    ledger.fail(phase8_runtime._Phase8Stage.analyze(), "backend_failure")
    ledger.succeed(phase8_runtime._Phase8Stage.analyze())

    assert isinstance(ledger.terminal(phase8_runtime._Phase8Stage.analyze()), phase8_runtime._StageFailed)
    assert isinstance(ledger.terminal(phase8_runtime._Phase8Stage.rewrite()), phase8_runtime._StageBlocked)
    assert ledger.reason(phase8_runtime._Phase8Stage.rewrite()) == "prerequisite"
    assert ledger.attempt_count(phase8_runtime._Phase8Stage.analyze()) == 1
    assert ledger.is_closed


def test_ledger_never_dispatches_out_of_compiled_order() -> None:
    ledger = phase8_runtime._Phase8OperationLedger(_plan(0))

    assert not ledger.dispatch(phase8_runtime._Phase8Stage.rewrite())
    assert not ledger.succeed(phase8_runtime._Phase8Stage.rewrite())
    assert ledger.terminal(phase8_runtime._Phase8Stage.rewrite()) is None
    ledger.succeed(phase8_runtime._Phase8Stage.validate_baselines())
    assert not ledger.dispatch(phase8_runtime._Phase8Stage.rewrite())
    ledger.succeed(phase8_runtime._Phase8Stage.analyze())
    assert ledger.dispatch(phase8_runtime._Phase8Stage.rewrite())


def test_evaluation_pass_closes_later_compiled_rounds() -> None:
    ledger = phase8_runtime._Phase8OperationLedger(_plan(2))
    for stage in (
        phase8_runtime._Phase8Stage.validate_baselines(),
        phase8_runtime._Phase8Stage.analyze(),
        phase8_runtime._Phase8Stage.rewrite(),
    ):
        ledger.succeed(stage)
    ledger.succeed(phase8_runtime._Phase8Stage.evaluate(0))
    ledger.close_pass(phase8_runtime._Phase8Stage.evaluate(0))

    assert isinstance(ledger.terminal(phase8_runtime._Phase8Stage.evaluate(0)), phase8_runtime._StageSucceeded)
    assert all(ledger.reason(stage) == "no_repair_needed" for stage in ledger.plan.stages[4:])
    assert ledger.is_closed


def test_pre_dispatch_cancellation_has_no_attempt_and_post_dispatch_without_stop_is_lost() -> None:
    plan = _plan(0)
    cancelled = phase8_runtime._Phase8OperationLedger(plan)
    cancelled.cancel(phase8_runtime._Phase8Stage.validate_baselines(), trusted_stop=True, dispatched=False)
    assert cancelled.attempt_count(phase8_runtime._Phase8Stage.validate_baselines()) == 0
    assert isinstance(
        cancelled.terminal(phase8_runtime._Phase8Stage.validate_baselines()), phase8_runtime._StageCancelled
    )
    assert cancelled.is_closed

    lost = phase8_runtime._Phase8OperationLedger(plan)
    lost.succeed(phase8_runtime._Phase8Stage.validate_baselines())
    lost.cancel(phase8_runtime._Phase8Stage.analyze(), trusted_stop=False, dispatched=True)
    assert lost.attempt_count(phase8_runtime._Phase8Stage.analyze()) == 1
    assert isinstance(lost.terminal(phase8_runtime._Phase8Stage.analyze()), phase8_runtime._StageLost)


def test_receipts_and_terminal_reprs_are_content_free() -> None:
    ledger = phase8_runtime._Phase8OperationLedger(_plan(0))
    ledger.succeed(phase8_runtime._Phase8Stage.validate_baselines())
    ledger.fail(phase8_runtime._Phase8Stage.analyze(), "backend_failure")

    assert "text" not in repr(ledger.terminal(phase8_runtime._Phase8Stage.analyze()))
    assert "text" not in repr(ledger._attempts[phase8_runtime._Phase8Stage.analyze()])
    assert ledger._attempts[phase8_runtime._Phase8Stage.analyze()].terminal == "failed"


def test_local_failure_continues_but_global_terminals_embargo_later_dispatch() -> None:
    invocation = phase8_runtime._Phase8InvocationLedger()
    assert invocation.admit(phase8_runtime._GroupFailed("backend_failure"))
    assert invocation.admit(phase8_runtime._GroupInconsistent("local_reconciliation"))
    assert not invocation.global_embargo

    assert not invocation.admit(phase8_runtime._GroupLost("transport_lost"))
    assert invocation.global_embargo
    assert isinstance(invocation.aggregate(), phase8_runtime._GroupInconsistent)


def test_group_aggregate_uses_frozen_failure_precedence() -> None:
    invocation = phase8_runtime._Phase8InvocationLedger(
        group_terminals=[
            phase8_runtime._GroupSucceeded(),
            phase8_runtime._GroupBlocked("prerequisite"),
            phase8_runtime._GroupFailed("backend_failure"),
            phase8_runtime._GroupCancelled("cancellation"),
            phase8_runtime._GroupLost("transport_lost"),
            phase8_runtime._GroupInconsistent("correlation"),
        ]
    )

    assert isinstance(invocation.aggregate(), phase8_runtime._GroupInconsistent)
    invocation.group_terminals.pop()
    assert isinstance(invocation.aggregate(), phase8_runtime._GroupLost)

    successful = phase8_runtime._Phase8InvocationLedger(
        group_terminals=[phase8_runtime._GroupSucceeded(), phase8_runtime._GroupSucceeded()]
    )
    assert isinstance(successful.aggregate(), phase8_runtime._GroupSucceeded)
