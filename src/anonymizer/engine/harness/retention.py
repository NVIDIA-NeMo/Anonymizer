# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live executions kept beside the results they produced, never on them.

Result objects (the facade's results and the engine's per-stage results) keep the release's
public surface: fields, ``asdict``, ``vars``, pickle state, ``repr`` and ``==`` carry no
execution. The execution that produced a result, and its retained leases, live in this
table for as long as the result object itself does.

An execution must never hold a strong reference to its own result: the module-level table
would then pin both forever.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from anonymizer.engine.graph_sdk.executor import ExecutionResult

_EXECUTIONS: dict[int, object] = {}


def attach_execution(owner: object, execution: object | None) -> None:
    """Keep ``execution`` alive exactly as long as ``owner``; ``None`` detaches."""
    key = id(owner)
    if execution is None:
        _EXECUTIONS.pop(key, None)
        return
    if key not in _EXECUTIONS:
        weakref.finalize(owner, _EXECUTIONS.pop, key, None)
    _EXECUTIONS[key] = execution


def attached_execution(owner: object) -> object | None:
    return _EXECUTIONS.get(id(owner))


class RetainsExecution:
    """Read-only ``_execution`` for engine stage results (frozen dataclasses)."""

    __slots__ = ()

    @property
    def _execution(self) -> ExecutionResult | None:
        return cast("ExecutionResult | None", attached_execution(self))
