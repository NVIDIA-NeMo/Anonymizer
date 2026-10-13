# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DD lowering of admitted initial Rewrite evaluation operations.

This private consumer is not selected by the public route while bounded repair
and final candidate qualification are incomplete.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict, TypeVar

import pandas as pd
from data_designer.config.column_types import ColumnConfigT

from anonymizer.engine.constants import (
    COL_PRIVACY_QA,
    COL_QUALITY_QA,
)
from anonymizer.engine.graph_sdk._execution_values import DelegatedRequests
from anonymizer.engine.graph_sdk.executor import LocalCompleted
from anonymizer.engine.graph_sdk.requests import (
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
    publication_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def columns(self, columns: list[ColumnConfigT]) -> list[ColumnConfigT]:
        from anonymizer.engine.rewrite.combined_rewrite_workflow import _remap_custom_column

        return [
            _remap_custom_column(
                column, {}, additional_inputs=(COL_QUALITY_QA, COL_PRIVACY_QA, self.state.rewritten_text)
            )
            if column.name in {self.state.quality_reanswer, self.state.privacy_reanswer}
            else column
            for column in columns
        ]

    async def _call(self, operation: Coroutine[Any, Any, _T]) -> _T:
        return await asyncio.shield(asyncio.wrap_future(asyncio.run_coroutine_threadsafe(operation, self.loop)))

    def stage(self, frame: pd.DataFrame, positions: tuple[int, ...]) -> _StageArguments:
        return {
            "source_frame": WorkflowSourceFrame.capture(
                frame, associations=tuple(self.source.associations[index] for index in positions), invocation_bound=True
            ),
            "rewrite_binding": self,
        }

    async def before(self, column: str, association: SemanticAssociation, row: dict[str, Any]) -> dict[str, Any]:
        names = {
            self.state.quality_reanswer: "quality",
            self.state.privacy_reanswer: "privacy",
            self.state.quality_compare: "comparison",
            self.state.utility_score: "metrics",
        }
        name = names.get(column)
        if name is None:
            return row
        if name in {"quality", "privacy"}:
            async with self.publication_lock:
                prefix = await self._call(self.requests.enter_operation(self.slots["prefix"], association))
                if prefix is not None:
                    await self.after(self.state.rewritten_text, association, row, prefix=True)
        bound = await self._call(self.requests.enter_operation(self.slots[name], association))
        assert bound is not None
        result = row.copy()
        for port in bound.inputs:
            value = port.value
            if type(value) is RewriteQAValue:
                result[COL_QUALITY_QA] = {"items": [item._asdict() for item in value.quality]}
                result[COL_PRIVACY_QA] = {"items": [item._asdict() for item in value.privacy]}
            elif type(value) is TextArtifactValue and port.port == "candidate":
                result[self.state.rewritten_text] = value.text
            elif type(value) is QualityAnswersValue:
                result[self.state.quality_reanswer] = {"answers": [item._asdict() for item in value.answers]}
            elif type(value) is PrivacyAnswersValue:
                result[self.state.privacy_reanswer] = {"answers": [item._asdict() for item in value.answers]}
            elif type(value) is QualityComparisonValue:
                result[self.state.quality_compare] = {"per_item": [item._asdict() for item in value.comparisons]}
        return result

    async def after(
        self, column: str, association: SemanticAssociation, row: dict[str, Any], *, prefix: bool = False
    ) -> None:
        if column == self.state.rewritten_text:
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
            values = {"candidate": TextArtifactValue(text=row[column]), "qa": qa}
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
                "answers": PrivacyAnswersValue(
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
        assert bound is not None and isinstance(bound.association, SemanticAssociation)
        completed = LocalCompleted(
            results=(
                AssociationResult(
                    association=bound.association,
                    outcome="ok",
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
        outcome = await self._call(self.requests.complete_operation(bound.association, completed))
        if outcome.condition != "result" or outcome.category != "success":
            raise AnonymizerWorkflowError("Rewrite semantic output was rejected")


class _StageArguments(TypedDict):
    source_frame: WorkflowSourceFrame
    rewrite_binding: RewriteEvaluationBinding
    request_bridge: NotRequired[None]
