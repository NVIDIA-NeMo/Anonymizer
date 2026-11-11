# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a fixed sequence of stages as one kernel invocation."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from anonymizer.engine.graph_core.authoring import (
    DATUM,
    GraphScope,
    Limits,
    Operation,
    ProtectionGraph,
    Requests,
    Workflow,
)
from anonymizer.engine.graph_core.execution import ProtectionResult, prepare, protect
from anonymizer.engine.graph_core.kernel import register_kind
from anonymizer.interface.errors import AnonymizerWorkflowError

STAGE_DONE = register_kind("stage-done", lambda v: v is True)


def call_on_empty_input(graph: ProtectionGraph, *operations: Operation) -> None:
    """With no targets the operations never ran; call each on the empty input, so its callee still sees it."""
    if not graph.targets:
        for operation in operations:
            operation.fn(())


def raise_if_cancelled(result: ProtectionResult) -> None:
    """Raise ``AnonymizerWorkflowError`` when a cancellation stopped any target before its steps finished."""
    if any(result.outcome(target).terminal == "cancelled" for target in result.targets):
        raise AnonymizerWorkflowError("The workflow was cancelled before it finished.")


def run_stages(name: str, stages: Sequence[tuple[str, Callable[[], None]]]) -> None:
    """Run ``stages`` in order as one kernel invocation, each an opaque operation over a single frame target.

    The first error a stage raised is raised, as when the stages were called directly. If an enclosing frame is
    cancelled, the stages not yet started are skipped and ``AnonymizerWorkflowError`` is raised.
    """

    def operation(stage_name: str, stage: Callable[[], None]) -> Operation:
        def run(target: object) -> bool:
            stage()
            return True

        return Operation(stage_name, run, {"target": DATUM}, {"done": STAGE_DONE}, Requests.opaque())

    scope = GraphScope()
    graph = ProtectionGraph(scope, (scope.datum(""),))
    workflow = Workflow(name, tuple(operation(*stage) for stage in stages))
    limits = Limits(max_operations=len(stages), max_targets=1, max_mentions=0, max_context_items=0)
    result = protect(prepare(graph, workflow, limits=limits))
    for entry in result.state.entries:
        if entry.error is not None:
            raise entry.error
    raise_if_cancelled(result)
