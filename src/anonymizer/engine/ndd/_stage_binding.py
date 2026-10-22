# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Route-neutral base for a graph stage bound around DataDesigner work.

A binding enters its stage's ready slots, publishes one result per source association and completes them
against the admitted graph. ``NddAdapter.run_workflow`` sees only the column hooks, so the NDD boundary does
not import any route's binding.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, cast

from anonymizer.engine.graph_sdk._execution_values import LocalCompleted
from anonymizer.engine.graph_sdk.requests import AssociationResult
from anonymizer.interface.errors import AnonymizerWorkflowError

if TYPE_CHECKING:
    from data_designer.config.column_types import ColumnConfigT

    from anonymizer.engine.graph_sdk._execution_values import DelegatedRequests, LocalAssessmentResult
    from anonymizer.engine.graph_sdk.requests import AssociationInput, PortArtifact, SemanticAssociation


def completed(
    association: Any,
    outputs: Iterable[PortArtifact] = (),
    outcome: str = "ok",
    assessments: tuple[LocalAssessmentResult, ...] = (),
) -> LocalCompleted:
    """One association's local result."""
    return LocalCompleted(
        assessments=assessments,
        results=(
            AssociationResult(
                association=association, outcome=outcome, outputs=tuple(outputs), consumed_context_ports=frozenset()
            ),
        ),
    )


class StageBinding:
    """Ready, publish and complete for one stage's slots; the column hooks default to pass-through."""

    requests: DelegatedRequests
    slots: Mapping[str, int]
    rejected = "Semantic result was rejected"
    # Each stage that enters ready slots names its own failures: a ready slot that would not enter, and
    # (for whole-frame stages) a frame only partly ready.
    not_ready: Callable[..., None]
    partially_ready: Callable[[], None]

    def columns(self, columns: list[ColumnConfigT]) -> list[ColumnConfigT]:
        """Rewrite the stage's column configs before the workflow is built."""
        return columns

    async def before(self, column: str, association: SemanticAssociation, row: dict[str, Any]) -> dict[str, Any]:
        """Observe or adjust one row's request body before the column generates it."""
        del column, association
        return row

    async def after(
        self, column: str, association: SemanticAssociation, row: dict[str, Any], *, prefix: bool = False
    ) -> None:
        """Publish one generated column value for its source association."""
        del column, association, row, prefix

    async def enter_ready(
        self, associations: tuple[Any, ...], *, whole: bool = False
    ) -> tuple[str, tuple[AssociationInput, ...], tuple[Any, ...]] | None:
        """Enter the first slot with ready sources: its name, its entered inputs and the selected sources."""
        ready = (
            await self.requests.ready_operations_many(associations)
            if associations
            else (await self.requests.ready_operations(None),)
        )
        for name, slot in self.slots.items():
            selected = (
                tuple(a for a, slots in zip(associations, ready, strict=True) if slot in slots) if associations else ()
            )
            if selected or not associations and slot in ready[0]:
                if whole and len(selected) != len(associations):
                    self.partially_ready()
                entered = (
                    await self.requests.enter_operations(slot, selected)
                    if selected
                    else (await self.requests.enter_operation(slot, None),)
                )
                if any(item is None for item in entered):
                    self.not_ready(empty=not associations)
                return name, cast(tuple["AssociationInput", ...], tuple(entered)), selected
        return None

    def check_outcomes(self, outcomes: tuple[Any, ...]) -> None:
        """Inspect raw outcomes before the success check."""
        del outcomes

    async def complete_results(self, results: Iterable[tuple[SemanticAssociation, Any]]) -> None:
        """Complete each association's result; every completed one must be accepted."""
        results = tuple(results)
        outcomes = await self.requests.complete_operations(results)
        self.check_outcomes(outcomes)
        if len(outcomes) != len(results) or any(
            outcome.condition != "result" or outcome.category != "success"
            for (_, result), outcome in zip(results, outcomes)
            if isinstance(result, LocalCompleted)
        ):
            raise AnonymizerWorkflowError(self.rejected)
