# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-15 stage sequences run through ``run_stages`` (E15.1-E15.3)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from anonymizer.engine.graph_core import stages
from tests.spec_programs._builder import Checks


def _stages(order: list[str], tids: list[int], behaviours: dict[str, Callable[[], None]] | None = None):
    def stage(name: str) -> Callable[[], None]:
        def run() -> None:
            order.append(name)
            tids.append(threading.get_ident())
            if behaviours and name in behaviours:
                behaviours[name]()

        return run

    return [(name, stage(name)) for name in ("first", "second", "third")]


def _raise(exc: BaseException) -> Callable[[], None]:
    def run() -> None:
        raise exc

    return run


def program() -> list[dict[str, str]]:
    c = Checks("SP-15")
    order: list[str] = []
    tids: list[int] = []
    results: list[Any] = []
    real = stages.protect

    def capture(prepared: Any, **kw: Any) -> Any:
        results.append(real(prepared, **kw))
        return results[-1]

    setattr(stages, "protect", capture)
    try:
        stages.run_stages("w", _stages(order, tids))
    finally:
        setattr(stages, "protect", real)
    attempts = results[0].attempts if results else ()
    c.check(
        "E15.1",
        order == ["first", "second", "third"]
        and set(tids) == {threading.get_ident()}
        and len(results) == 1
        and len(results[0].targets) == 1
        and len(attempts) == 3
        and all(x.status == "succeeded" for x in attempts),
        f"stage runs={len(order)}, invocations={len(results)}, targets={len(results[0].targets)}, "
        f"attempts={len(attempts)}",
    )

    def raised_exception() -> tuple[bool, str]:
        order.clear()
        failure = RuntimeError("x")
        try:
            stages.run_stages("w", _stages(order, [], {"second": _raise(failure)}))
        except RuntimeError as exc:
            return exc is failure and order == ["first", "second"], f"raised to caller, stage runs={len(order)}"
        return False, f"nothing raised, stage runs={len(order)}"

    c.guard("E15.2", raised_exception)

    order.clear()
    interrupt = KeyboardInterrupt()
    try:
        stages.run_stages("w", _stages(order, [], {"second": _raise(interrupt)}))
    except KeyboardInterrupt as exc:
        c.check(
            "E15.3",
            exc is interrupt and order == ["first", "second"],
            f"interrupt propagated unchanged, stage runs={len(order)}",
        )
    else:
        c.check("E15.3", False, f"nothing raised, stage runs={len(order)}")
    return c.lines
