# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DD lowering of admitted Rewrite evaluation and repair operations.

This private consumer is not selected by the public route while bounded repair
and final candidate qualification are incomplete.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict, TypeVar, cast

import pandas as pd
from data_designer.config.column_types import ColumnConfigT

from anonymizer.engine.constants import (
    COL_ENTITIES_BY_VALUE,
    COL_LEAKED_PRIVACY_ITEMS,
    COL_PRIVACY_QA,
    COL_QUALITY_QA,
    COL_REWRITE_BASELINE_TEXT,
    COL_REWRITTEN_TEXT,
    COL_REWRITTEN_TEXT_INITIAL,
    COL_REWRITTEN_TEXT_NEXT,
    COL_TEXT,
)
from anonymizer.engine.graph_sdk._effect_values import EffectRejected
from anonymizer.engine.graph_sdk._execution_values import DelegatedRequests
from anonymizer.engine.graph_sdk.executor import LocalCompleted
from anonymizer.engine.graph_sdk.requests import (
    ArtifactValue,
    AssociationInput,
    AssociationResult,
    PortArtifact,
    SemanticAssociation,
    TextArtifactValue,
)
from anonymizer.engine.ndd._source_frame import WorkflowSourceFrame
from anonymizer.engine.rewrite.artifacts import (
    REWRITE_TYPES,
    EvaluationMetricsValue,
    PrivacyAnswer,
    PrivacyAnswersValue,
    PrivacyQuestion,
    QualityAnswer,
    QualityAnswersValue,
    QualityComparison,
    QualityComparisonValue,
    QualityQuestion,
    RewriteQAValue,
)
from anonymizer.engine.rewrite.parsers import (
    normalize_payload,
    parse_privacy_answers,
    parse_privacy_qa,
    parse_quality_answers,
    parse_quality_qa,
)
from anonymizer.engine.schemas.rewrite import QACompareResultsSchema
from anonymizer.graph.workflow import ArtifactType
from anonymizer.interface.errors import AnonymizerWorkflowError

if TYPE_CHECKING:
    from anonymizer.engine.rewrite.combined_rewrite_workflow import EvaluationState


_T = TypeVar("_T")


@dataclass
class RewriteEvaluationBinding:
    requests: DelegatedRequests
    source: WorkflowSourceFrame
    slots: dict[str, int]
    loop: asyncio.AbstractEventLoop
    state: EvaluationState
    iteration_slots: dict[int, dict[str, int]] = field(default_factory=dict)
    combined: bool = False
    iteration: int = 0
    publication_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def for_iteration(self, iteration: int) -> RewriteEvaluationBinding:
        from anonymizer.engine.rewrite.combined_rewrite_workflow import _evaluation_state, _repair_state

        return replace(
            self,
            slots=self.iteration_slots[iteration - 1],
            iteration=iteration,
            state=_evaluation_state(
                iteration if self.combined else None,
                _repair_state(iteration - 1).rewritten_text if self.combined else COL_REWRITTEN_TEXT,
            ),
        )

    def selected(self, column: str) -> RewriteEvaluationBinding:
        if not self.combined or self.iteration:
            return self
        from anonymizer.engine.rewrite.combined_rewrite_workflow import _repair_state

        for index in self.iteration_slots:
            selected = self.for_iteration(index + 1)
            if column in {*selected.state.__dict__.values(), *_repair_state(index).__dict__.values()}:
                return selected
        return self

    @property
    def repair_columns(self) -> tuple[str, str]:
        from anonymizer.engine.rewrite.combined_rewrite_workflow import _repair_state

        state = _repair_state(self.iteration - 1) if self.combined else None
        return (
            (state.leaked_items, state.rewritten_text) if state else (COL_LEAKED_PRIVACY_ITEMS, COL_REWRITTEN_TEXT_NEXT)
        )

    async def passthrough(self, frame: pd.DataFrame) -> None:
        from anonymizer.engine.rewrite.rewrite_workflow import _has_entities

        for position, (_, row) in enumerate(frame.iterrows()):
            if _has_entities(row[COL_ENTITIES_BY_VALUE]):
                continue
            association = self.source.associations[position]
            stages: tuple[tuple[str, dict[str, ArtifactValue]], ...] = (
                (
                    "prefix",
                    {
                        "candidate": TextArtifactValue(text=row[COL_TEXT]),
                        "baseline": TextArtifactValue(text=row[COL_TEXT]),
                        "qa": RewriteQAValue(quality=(), privacy=()),
                    },
                ),
                ("quality", {"answers": QualityAnswersValue(answers=())}),
                ("privacy", {"privacy": PrivacyAnswersValue(answers=())}),
                ("comparison", {"comparison": QualityComparisonValue(comparisons=())}),
                (
                    "metrics",
                    {
                        "metrics": EvaluationMetricsValue(
                            utility=1.0, leakage=0.0, weighted_leakage=0.0, high_leak=False, needs_repair=False
                        )
                    },
                ),
            )
            for name, values in stages:
                bound = await self.requests.enter_operation(self.slots[name], association)
                assert bound is not None
                if name == "metrics":
                    values.update(
                        {
                            port.port.removeprefix("in_"): port.value
                            for port in bound.inputs
                            if port.port.removeprefix("in_") in {"source", "baseline", "qa", "candidate", "privacy"}
                        }
                    )
                await self.publish(bound, values, "stop" if name == "metrics" else "skipped")

    async def finish(self) -> dict[SemanticAssociation, SemanticAssociation]:
        selected = {}
        for association in self.source.associations:
            for name in ("join", "initial", "final"):
                try:
                    bound = await self.requests.enter_operation(self.slots[name], association)
                except EffectRejected:
                    break
                if bound is None:
                    continue
                values = {} if name == "join" else {"candidate": bound.inputs[0].value}
                control = (
                    "again"
                    if name == "join"
                    and cast(EvaluationMetricsValue, bound.inputs[0].value).needs_repair
                    and self.iteration_slots
                    else "stop"
                    if name == "join"
                    else "ok"
                )
                await self.publish(bound, values, control)
                if name != "join":
                    assert isinstance(bound.association, SemanticAssociation)
                    selected[association] = bound.association
        return selected

    def columns(self, columns: list[ColumnConfigT]) -> list[ColumnConfigT]:
        from anonymizer.engine.rewrite.combined_rewrite_workflow import _remap_custom_column

        return [
            _remap_custom_column(
                column,
                {},
                additional_inputs=(
                    COL_QUALITY_QA,
                    COL_PRIVACY_QA,
                    COL_REWRITE_BASELINE_TEXT,
                    self.selected(column.name).state.rewritten_text,
                ),
            )
            if column.name
            in {self.selected(column.name).state.quality_reanswer, self.selected(column.name).state.privacy_reanswer}
            else column
            for column in columns
        ]

    async def _call(self, operation: Coroutine[Any, Any, _T]) -> _T:
        return await asyncio.shield(asyncio.wrap_future(asyncio.run_coroutine_threadsafe(operation, self.loop)))

    def stage(self, frame: pd.DataFrame, positions: tuple[int, ...], *, iteration: int = 0) -> _StageArguments:
        return {
            "source_frame": WorkflowSourceFrame.capture(
                frame, associations=tuple(self.source.associations[index] for index in positions), invocation_bound=True
            ),
            "rewrite_binding": self.for_iteration(iteration) if iteration else self,
        }

    async def before(self, column: str, association: SemanticAssociation, row: dict[str, Any]) -> dict[str, Any]:
        selected = self.selected(column)
        if selected is not self:
            return await selected.before(column, association, row)
        names = {
            self.state.quality_reanswer: "quality",
            self.state.privacy_reanswer: "privacy",
            self.state.quality_compare: "comparison",
            self.state.utility_score: "metrics",
        }
        if self.iteration:
            names[self.repair_columns[0]] = "repair"
        name = names.get(column)
        if name is None:
            return row
        if not self.iteration and name in {"quality", "privacy"}:
            async with self.publication_lock:
                prefix = await self._call(self.requests.enter_operation(self.slots["prefix"], association))
                if prefix is not None:
                    await self.after(self.state.rewritten_text, association, row, prefix=True)
        bound = await self._call(self.requests.enter_operation(self.slots[name], association))
        if bound is None:
            return row
        result = row.copy()
        input_state = self.state
        if name == "repair" and self.combined:
            from anonymizer.engine.rewrite.combined_rewrite_workflow import _evaluation_state, _repair_state

            input_state = _evaluation_state(
                self.iteration - 1,
                COL_REWRITTEN_TEXT_INITIAL if self.iteration == 1 else _repair_state(self.iteration - 2).rewritten_text,
            )
        for port in bound.inputs:
            value = port.value
            port_name = port.port.removeprefix("in_")
            if type(value) is RewriteQAValue:
                result[COL_QUALITY_QA] = {"items": [item._asdict() for item in value.quality]}
                result[COL_PRIVACY_QA] = {"items": [item._asdict() for item in value.privacy]}
            elif type(value) is TextArtifactValue:
                if port_name == "source":
                    result[COL_TEXT] = value.text
                elif port_name == "baseline":
                    result[COL_REWRITE_BASELINE_TEXT] = value.text
                elif port_name == "candidate":
                    result[input_state.rewritten_text] = value.text
            elif type(value) is QualityAnswersValue:
                result[self.state.quality_reanswer] = {"answers": [item._asdict() for item in value.answers]}
            elif type(value) is PrivacyAnswersValue:
                result[input_state.privacy_reanswer] = {"answers": [item._asdict() for item in value.answers]}
            elif type(value) is EvaluationMetricsValue:
                result.update(
                    {
                        input_state.utility_score: value.utility,
                        input_state.leakage_mass: value.leakage,
                        input_state.weighted_leakage_rate: value.weighted_leakage,
                        input_state.any_high_leaked: value.high_leak,
                        input_state.needs_repair: value.needs_repair,
                    }
                )
            elif type(value) is QualityComparisonValue:
                result[self.state.quality_compare] = {"per_item": [item._asdict() for item in value.comparisons]}
        return result

    async def after(
        self, column: str, association: SemanticAssociation, row: dict[str, Any], *, prefix: bool = False
    ) -> None:
        values: dict[str, ArtifactValue]
        selected = self.selected(column)
        if selected is not self:
            return await selected.after(column, association, row, prefix=prefix)
        if self.iteration and column == self.repair_columns[1]:
            values = {"candidate": TextArtifactValue(text=row[column])}
            name = "repair"
        elif column == self.state.rewritten_text:
            if not prefix:
                return
            quality, privacy = parse_quality_qa(row[COL_QUALITY_QA]), parse_privacy_qa(row[COL_PRIVACY_QA])
            qa = RewriteQAValue(
                quality=tuple(
                    QualityQuestion(q.id, q.aspect, q.importance.value, q.question, q.reference_answer)
                    for q in quality.items
                ),
                privacy=tuple(
                    PrivacyQuestion(
                        q.id, q.question, q.sensitivity.value, q.entity_label, q.entity_value, q.category.value
                    )
                    for q in privacy.items
                ),
            )
            values = {
                "candidate": TextArtifactValue(text=row[column]),
                "baseline": TextArtifactValue(text=row[COL_REWRITE_BASELINE_TEXT]),
                "qa": qa,
            }
            name = "prefix"
        elif column == self.state.quality_reanswer:
            values = {
                "answers": QualityAnswersValue(
                    answers=tuple(QualityAnswer(q.id, q.answer) for q in parse_quality_answers(row[column]))
                )
            }
            name = "quality"
        elif column == self.state.privacy_reanswer:
            values = {
                "privacy": PrivacyAnswersValue(
                    answers=tuple(
                        PrivacyAnswer(q.id, q.answer.value, q.confidence, q.reason, tuple(q.evidence))
                        for q in parse_privacy_answers(row[column])
                    )
                )
            }
            name = "privacy"
        elif column == self.state.quality_compare:
            compared = QACompareResultsSchema.model_validate(normalize_payload(row[column]))
            values = {
                "comparison": QualityComparisonValue(
                    comparisons=tuple(QualityComparison(q.id, q.score, q.reason) for q in compared.per_item)
                )
            }
            name = "comparison"
        elif column == self.state.needs_repair:
            values = {
                "metrics": EvaluationMetricsValue(
                    utility=float(row[self.state.utility_score]),
                    leakage=float(row[self.state.leakage_mass]),
                    weighted_leakage=float(row[self.state.weighted_leakage_rate]),
                    high_leak=bool(row[self.state.any_high_leaked]),
                    needs_repair=bool(row[column]),
                )
            }
            name = "metrics"
        else:
            return
        bound = await self._call(self.requests.enter_operation(self.slots[name], association))
        if bound is None:
            return
        outcome = "ok"
        if name == "metrics":
            values.update(
                {
                    port.port.removeprefix("in_"): port.value
                    for port in bound.inputs
                    if port.port.removeprefix("in_") in {"source", "baseline", "qa", "candidate", "privacy"}
                }
            )
            needs_repair = cast(EvaluationMetricsValue, values["metrics"]).needs_repair
            outcome = (
                "stop" if not needs_repair else "again" if self.iteration < len(self.iteration_slots) else "exhausted"
            )
        await self.publish(bound, values, outcome)

    async def publish(self, bound: AssociationInput, values: dict[str, ArtifactValue], outcome: str = "ok") -> None:
        assert isinstance(bound.association, SemanticAssociation)
        completed = LocalCompleted(
            results=(
                AssociationResult(
                    association=bound.association,
                    outcome=outcome,
                    consumed_context_ports=frozenset(),
                    outputs=tuple(
                        PortArtifact(
                            port=port,
                            artifact_type=REWRITE_TYPES.get(
                                type(value), ArtifactType(name="anonymizer.text", revision=1)
                            ),
                            artifact=None,
                            value=value,
                        )
                        for port, value in values.items()
                    ),
                ),
            )
        )
        result = await self._call(self.requests.complete_operation(bound.association, completed))
        if result.condition != "result" or result.category != "success":
            raise AnonymizerWorkflowError("Rewrite semantic output was rejected")


class _StageArguments(TypedDict):
    source_frame: WorkflowSourceFrame
    rewrite_binding: RewriteEvaluationBinding
    request_bridge: NotRequired[None]
