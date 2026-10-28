# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-bound stage publication and lowerings of admitted Rewrite control."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, TypedDict, TypeVar, cast

import pandas as pd
import pyarrow as pa
from data_designer.config import ModelConfig
from data_designer.config.column_types import ColumnConfigT

from anonymizer.config.models import RewriteModelSelection
from anonymizer.config.rewrite import EvaluationCriteria, PrivacyGoal
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_ENTITIES_BY_VALUE,
    COL_LEAKAGE_MASS,
    COL_LEAKED_PRIVACY_ITEMS,
    COL_NEEDS_HUMAN_REVIEW,
    COL_NEEDS_REPAIR,
    COL_PRIVACY_QA,
    COL_PRIVACY_QA_REANSWER,
    COL_QUALITY_QA,
    COL_QUALITY_QA_COMPARE,
    COL_QUALITY_QA_REANSWER,
    COL_REPAIR_ITERATIONS,
    COL_REWRITE_BASELINE_TEXT,
    COL_REWRITE_REPLACEMENT_READY,
    COL_REWRITTEN_TEXT,
    COL_REWRITTEN_TEXT_INITIAL,
    COL_REWRITTEN_TEXT_NEXT,
    COL_TEXT,
    COL_UTILITY_SCORE,
    COL_WEIGHTED_LEAKAGE_RATE,
)
from anonymizer.engine.graph_sdk._effect_values import EffectCode, reject
from anonymizer.engine.graph_sdk._execution_delegated import drive
from anonymizer.engine.graph_sdk._execution_values import DelegatedRequests
from anonymizer.engine.graph_sdk._workflow_values import ArtifactType
from anonymizer.engine.graph_sdk.executor import AssessmentFinding, LocalAssessmentResult
from anonymizer.engine.graph_sdk.requests import (
    ArtifactValue,
    AssociationInput,
    AssociationResult,
    PortArtifact,
    SemanticAssociation,
    TextArtifactValue,
)
from anonymizer.engine.ndd._source_frame import WorkflowSourceFrame
from anonymizer.engine.ndd._stage_binding import StageBinding, completed
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
    RewriteCandidateValue,
    RewriteControlValue,
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
from anonymizer.interface.errors import AnonymizerWorkflowError

if TYPE_CHECKING:
    from anonymizer.engine.ndd.adapter import FailedRecord
    from anonymizer.engine.rewrite.combined_rewrite_workflow import EvaluationState
    from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow


_T = TypeVar("_T")
EVALUATION_PROMISE = "rewrite-evaluation"
EVALUATION_MEANING = "Current candidate meets captured repair criteria for its bound QA responses"
EVALUATION_PASS = AssessmentFinding(status="satisfied", code="repair_criteria_met")
EVALUATION_FAIL = AssessmentFinding(status="unsatisfied", code="repair_required")


@dataclass
class RewriteEvaluationBinding(StageBinding):
    requests: DelegatedRequests
    source: WorkflowSourceFrame
    slots: dict[str, int]
    state: EvaluationState
    criteria: EvaluationCriteria
    iteration_slots: dict[int, dict[str, int]] = field(default_factory=dict)
    combined: bool = False
    opaque: bool = False
    prefix_is_child: bool = False
    iteration: int = 0
    publication_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    prefix_observed: set[SemanticAssociation] = field(default_factory=set, repr=False)
    # No-entity sources: the legacy passthrough never flagged them for review, even when null.
    passthrough_sources: set[SemanticAssociation] = field(default_factory=set, repr=False)
    _entry_owner: RewriteEvaluationBinding | None = field(default=None, repr=False, compare=False)

    _map_prepare: Any | None = field(default=None, repr=False)
    # Object columns of the lowered frame that held missing cells when the evaluate-repair loop began.
    held_missing: frozenset[str] = field(default=frozenset(), repr=False)

    def prepare_replacement_maps(self, entity_rows: pd.DataFrame, passthrough_rows: pd.DataFrame):
        self.authorize_entry()
        return self._map_prepare(entity_rows, passthrough_rows) if self._map_prepare is not None else None

    def __post_init__(self) -> None:
        if self._entry_owner is None:
            self._entry_owner = self

    def authorize_entry(self) -> None:
        from anonymizer.engine.graph_sdk._execution_delegated import source_worker_borrow

        borrow = source_worker_borrow()
        if (
            self is not self._entry_owner
            or self.requests is not borrow
            or borrow.owner.closed
            or type(self.source) is not WorkflowSourceFrame
            or not self.source.invocation_bound
        ):
            reject(EffectCode.FOREIGN_OWNER)
        for association in self.source.associations:
            borrow.authorize_source_sync(association)

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
            self.passthrough_sources.add(association)
            stages: tuple[tuple[str, dict[str, ArtifactValue]], ...] = (
                (
                    "prefix",
                    {
                        "candidate": RewriteCandidateValue.capture(row[COL_TEXT]),
                        "baseline": RewriteCandidateValue.capture(row[COL_TEXT]),
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
                if self.opaque and name not in {"prefix", "metrics"}:
                    continue
                if self.opaque and name == "prefix":
                    values.pop("qa")
                if self.opaque and name == "metrics":
                    values["metrics"] = RewriteControlValue(needs_repair=False)
                bound = await self.requests.enter_operation(self.slots[name], association)
                assert bound is not None
                if name == "metrics":
                    values.update(
                        {
                            port.port.removeprefix("in_"): port.value
                            for port in bound.inputs
                            if port.port.removeprefix("in_") != "metrics"
                        }
                    )
                await self.publish(bound, values, "skipped")
                if name == "prefix":
                    self.prefix_observed.add(association)

    async def final_result(self, association: SemanticAssociation) -> AssociationResult:
        for name in ("join", "initial", "final"):
            completed = await self.requests.operation_result(self.slots[name], association)
            if completed is not None:
                if name != "join":
                    return completed
                continue
            bound = await self.requests.enter_operation(self.slots[name], association)
            if bound is None:
                continue
            values = {port.port.removeprefix("in_"): port.value for port in bound.inputs} if name != "join" else {}
            control = (
                "again"
                if name == "join"
                and cast(EvaluationMetricsValue, bound.inputs[0].value).needs_repair
                and self.iteration_slots
                else "stop"
                if name == "join"
                else "ok"
            )
            assessments: tuple[LocalAssessmentResult, ...] = ()
            if name != "join":
                initial = await self.requests.operation_result(self.slots["metrics"], association)
                assert initial is not None and isinstance(bound.association, SemanticAssociation)
                if initial.outcome == "skipped":
                    control = "skipped"
                elif not self.opaque:
                    actual = cast(EvaluationMetricsValue, values["metrics"])
                    expected = self._metric(bound, actual.iteration)
                    assessments = (
                        LocalAssessmentResult(
                            association=bound.association,
                            promise=EVALUATION_PROMISE,
                            evidence_port="metrics",
                            finding=EVALUATION_PASS
                            if expected == actual
                            and not expected.needs_repair
                            and isinstance(cast(RewriteCandidateValue, values["candidate"]).project(), str)
                            else EVALUATION_FAIL,
                        ),
                    )
            await self.publish(bound, values, control, assessments=assessments)
            if name != "join":
                completed = await self.requests.operation_result(self.slots[name], association)
                assert completed is not None
                return completed
        raise AnonymizerWorkflowError("Rewrite has no selected final candidate")

    def final_row(
        self, row: dict[str, Any], result: AssociationResult, association: SemanticAssociation
    ) -> dict[str, Any]:
        values = {port.port: port.value for port in result.outputs}
        passthrough = association in self.passthrough_sources
        metrics = cast(EvaluationMetricsValue, values["metrics"])
        row = row.copy()
        if self.opaque:
            row[COL_REWRITTEN_TEXT] = cast(RewriteCandidateValue, values["candidate"]).project()
            if not self.combined:
                row[COL_REPAIR_ITERATIONS] = metrics.iteration
            return row
        row.update(
            {
                COL_REWRITTEN_TEXT: cast(RewriteCandidateValue, values["candidate"]).project(),
                COL_UTILITY_SCORE: metrics.utility,
                COL_LEAKAGE_MASS: metrics.leakage,
                COL_WEIGHTED_LEAKAGE_RATE: metrics.weighted_leakage,
                COL_ANY_HIGH_LEAKED: metrics.high_leak,
                COL_NEEDS_REPAIR: metrics.needs_repair,
                COL_REPAIR_ITERATIONS: metrics.iteration,
                COL_QUALITY_QA_REANSWER: {
                    "answers": [v._asdict() for v in cast(QualityAnswersValue, values["answers"]).answers]
                },
                COL_PRIVACY_QA_REANSWER: {
                    "answers": [v._asdict() for v in cast(PrivacyAnswersValue, values["privacy"]).answers]
                },
                COL_QUALITY_QA_COMPARE: {
                    "per_item": [v._asdict() for v in cast(QualityComparisonValue, values["comparison"]).comparisons]
                },
                COL_NEEDS_HUMAN_REVIEW: not passthrough
                and (
                    cast(RewriteCandidateValue, values["candidate"]).project() is None
                    or (not self.combined and bool(pd.isna(cast(RewriteCandidateValue, values["candidate"]).project())))
                    or metrics.high_leak
                    or (
                        self.criteria.flag_utility_below is not None
                        and metrics.utility < self.criteria.flag_utility_below
                    )
                    or (
                        self.criteria.flag_leakage_above is not None
                        and metrics.leakage > self.criteria.flag_leakage_above
                    )
                ),
            }
        )
        return row

    async def finish(
        self, frame: pd.DataFrame, associations: tuple[SemanticAssociation, ...]
    ) -> tuple[dict[SemanticAssociation, SemanticAssociation], pd.DataFrame]:
        selected, rows = {}, []
        if self.opaque and self.combined:
            for association, row in zip(associations, frame.to_dict("records"), strict=True):
                if association not in self.prefix_observed:
                    observed = {**row, self.state.rewritten_text: row[COL_REWRITTEN_TEXT]}
                    await self.after(self.state.rewritten_text, association, observed, prefix=True)
                    observed[self.state.needs_repair] = bool(row.get(COL_NEEDS_REPAIR, False))
                    await self.after(self.state.needs_repair, association, observed)
        for association, row in zip(associations, frame.to_dict("records"), strict=True):
            result = await self.final_result(association)
            assert isinstance(result.association, SemanticAssociation)
            selected[association] = result.association
            rows.append(self.final_row(row, result, association))
        projected = pd.DataFrame(rows, index=frame.index)
        if self.opaque and COL_REWRITTEN_TEXT in frame:
            projected[COL_REWRITTEN_TEXT] = pd.Series(
                [
                    _released_cell(row[COL_REWRITTEN_TEXT], cell)
                    for row, cell in zip(rows, frame[COL_REWRITTEN_TEXT].tolist(), strict=True)
                ],
                index=frame.index,
                dtype=frame[COL_REWRITTEN_TEXT].dtype,
            )
        _restore_arrow_dtypes(projected, frame)
        _restore_object_dtypes(projected, frame, rows, self.held_missing)
        return selected, projected

    def lower_ready(
        self,
        runner: RewriteWorkflow,
        frame: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: RewriteModelSelection,
        privacy_goal: PrivacyGoal,
        evaluation: EvaluationCriteria,
        preview_num_records: int | None,
    ) -> tuple[pd.DataFrame, list[FailedRecord]]:
        """Batch only the phase occurrences made ready by the admitted Graph."""
        from anonymizer.engine.rewrite.rewrite_workflow import _join_new_columns, _normalize_evaluation_payloads
        from anonymizer.engine.rewrite.workflow_utils import derive_seed_columns, select_seed_cols
        from anonymizer.engine.row_partitioning import ROW_ORDER_COL

        failed = []
        if COL_REPAIR_ITERATIONS not in frame:
            frame[COL_REPAIR_ITERATIONS] = 0
        self.held_missing = frozenset(
            str(column) for column in frame.columns if frame[column].dtype == object and frame[column].isna().any()
        )
        evaluate = self.columns(runner._evaluate_wf.columns(selected_models=selected_models, evaluation=evaluation))
        repair = runner._repair_wf.columns(
            selected_models=selected_models, privacy_goal=privacy_goal, effective_threshold=evaluation.repair_threshold
        )
        stages = {self.slots["metrics" if self.opaque else "quality"]: ("evaluate", 0)}
        stages.update({slots["repair"]: ("repair", iteration + 1) for iteration, slots in self.iteration_slots.items()})
        stages.update(
            {
                slots["metrics" if self.opaque else "quality"]: ("evaluate", iteration + 1)
                for iteration, slots in self.iteration_slots.items()
            }
        )

        async def observe(rows: pd.DataFrame, name: str, iteration: int) -> None:
            selected = self.for_iteration(iteration) if iteration else self
            for ordinal, (_, row) in enumerate(rows.iterrows()):
                association = self.source.associations[int(row[ROW_ORDER_COL])]
                slot = selected.slots["prefix" if name == "prefix" else "repair" if name == "repair" else "metrics"]
                if name != "prefix" and await self.requests.operation_result(slot, association) is not None:
                    continue
                await selected.after(
                    selected.state.rewritten_text
                    if name == "prefix"
                    else selected.repair_columns[1]
                    if name == "repair"
                    else selected.state.needs_repair,
                    association,
                    {**row.to_dict(), selected.state.rewritten_text: rows[selected.state.rewritten_text].iloc[ordinal]}
                    if name == "prefix"
                    else row.to_dict(),
                    prefix=name == "prefix",
                )

        drive(observe(frame, "prefix", 0))

        async def ready() -> tuple[int, list[int]] | None:
            selected = {}
            for index, position in enumerate(frame[ROW_ORDER_COL]):
                for slot in await self.requests.ready_operations(self.source.associations[position]):
                    if slot in stages:
                        selected.setdefault(slot, []).append(index)
            return min(selected.items()) if selected else None

        while (batch := drive(ready())) is not None:
            slot, indices = batch
            name, iteration = stages[slot]
            rows = frame.iloc[indices].copy()
            columns = repair if name == "repair" else evaluate
            seed_cols = derive_seed_columns(columns, rows)
            seed = select_seed_cols(rows, seed_cols)
            result = runner._adapter.run_workflow(
                seed,
                model_configs=model_configs,
                columns=columns,
                workflow_name=f"rewrite-{name}-{iteration - 1 if name == 'repair' else iteration}",
                preview_num_records=preview_num_records,
                **self.stage(seed, tuple(rows[ROW_ORDER_COL]), iteration=iteration),
            )
            failed.extend(result.failed_records)
            rows = _join_new_columns(
                rows, result.dataframe, source_positions=result.source_positions, overwrite=True, seed_cols=seed_cols
            )
            if self.opaque:
                drive(observe(rows, name, iteration))
            if name == "repair":
                if COL_REWRITTEN_TEXT_NEXT in rows:
                    rows[COL_REWRITTEN_TEXT] = rows[COL_REWRITTEN_TEXT_NEXT]
                rows[COL_REPAIR_ITERATIONS] = rows[COL_REPAIR_ITERATIONS].apply(lambda value: int(value) + 1)
            else:
                _normalize_evaluation_payloads(rows)
            frame = pd.concat(
                [frame.iloc[[index for index in range(len(frame)) if index not in indices]], rows], ignore_index=True
            )
        unavailable = (
            ~frame[COL_REWRITE_REPLACEMENT_READY].apply(bool)
            if COL_REWRITE_REPLACEMENT_READY in frame
            else pd.Series(False, index=frame.index)
        )
        _withhold_rewritten_text(frame, unavailable)
        review = frame[COL_REWRITTEN_TEXT].isna() | unavailable | frame[COL_ANY_HIGH_LEAKED].apply(bool)
        if evaluation.flag_utility_below is not None:
            review |= frame[COL_UTILITY_SCORE].apply(float) < evaluation.flag_utility_below
        if evaluation.flag_leakage_above is not None:
            review |= frame[COL_LEAKAGE_MASS].apply(float) > evaluation.flag_leakage_above
        frame[COL_NEEDS_HUMAN_REVIEW] = review
        return frame, failed

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
                    COL_REWRITE_REPLACEMENT_READY,
                    self.selected(column.name).state.rewritten_text,
                ),
            )
            if column.name
            in {self.selected(column.name).state.quality_reanswer, self.selected(column.name).state.privacy_reanswer}
            else column
            for column in columns
        ]

    async def _call(self, operation: Coroutine[Any, Any, _T]) -> _T:
        return drive(operation)

    def stage(self, frame: pd.DataFrame, positions: tuple[int, ...], *, iteration: int = 0) -> _StageArguments:
        return {
            "source_frame": WorkflowSourceFrame.capture(
                frame, associations=tuple(self.source.associations[index] for index in positions), invocation_bound=True
            ),
            "rewrite_binding": self.for_iteration(iteration) if iteration else self,
        }

    async def before(self, column: str, association: SemanticAssociation, row: dict[str, Any]) -> dict[str, Any]:
        if self.combined and not self.opaque and column == COL_REWRITTEN_TEXT:
            return self.final_row(row, await self._call(self.final_result(association)), association)
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
        if name is None or name not in self.slots:
            return row
        if not self.iteration and name in ({"metrics"} if self.opaque else {"quality", "privacy"}):
            async with self.publication_lock:
                completed_prefix = (
                    await self._call(self.requests.operation_result(self.slots["prefix"], association))
                    if self.prefix_is_child
                    else None
                )
                prefix = (
                    None
                    if completed_prefix is not None
                    else await self._call(self.requests.enter_operation(self.slots["prefix"], association))
                )
                if prefix is not None:
                    await self.after(self.state.rewritten_text, association, row, prefix=True)
        bound = await self._call(self.requests.enter_operation(self.slots[name], association))
        if bound is None:
            if name == "repair" and self.combined:
                from anonymizer.engine.rewrite.combined_rewrite_workflow import _evaluation_state, _repair_state

                previous = _evaluation_state(
                    self.iteration - 1,
                    COL_REWRITTEN_TEXT_INITIAL
                    if self.iteration == 1
                    else _repair_state(self.iteration - 2).rewritten_text,
                )
                return {**row, previous.needs_repair: False}
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
            elif type(value) is RewriteCandidateValue:
                result[COL_REWRITE_BASELINE_TEXT if port_name == "baseline" else input_state.rewritten_text] = (
                    value.project()
                )
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
            values = {"candidate": RewriteCandidateValue.capture(row[column])}
            name = "repair"
        elif column == self.state.rewritten_text:
            if not prefix:
                return
            if self.opaque:
                bound = await self._call(self.requests.enter_operation(self.slots["prefix"], association))
                if bound is not None:
                    await self.publish(
                        bound,
                        {
                            "candidate": RewriteCandidateValue.capture(row[column]),
                            "baseline": RewriteCandidateValue.capture(
                                row.get(COL_REWRITE_BASELINE_TEXT, row[COL_TEXT])
                            ),
                        },
                    )
                self.prefix_observed.add(association)
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
                "candidate": RewriteCandidateValue.capture(row[column]),
                "baseline": RewriteCandidateValue.capture(row[COL_REWRITE_BASELINE_TEXT]),
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
            values = {}
            name = "metrics"
        else:
            return
        if name not in self.slots:
            return
        if (
            name == "prefix"
            and self.prefix_is_child
            and await self._call(self.requests.operation_result(self.slots[name], association)) is not None
        ):
            return
        bound = await self._call(self.requests.enter_operation(self.slots[name], association))
        if bound is None:
            return
        outcome = "ok"
        if name == "metrics":
            metric = (
                RewriteControlValue(needs_repair=bool(row[self.state.needs_repair]), iteration=self.iteration)
                if self.opaque
                else self._metric(bound, self.iteration)
            )
            values["metrics"] = metric
            if not self.opaque:
                if not isinstance(metric, EvaluationMetricsValue):
                    reject(EffectCode.INVALID_TYPE)
                row.update(
                    {
                        self.state.utility_score: metric.utility,
                        self.state.leakage_mass: metric.leakage,
                        self.state.weighted_leakage_rate: metric.weighted_leakage,
                        self.state.any_high_leaked: metric.high_leak,
                        self.state.needs_repair: metric.needs_repair,
                    }
                )
            values.update(
                {
                    port.port.removeprefix("in_"): port.value
                    for port in bound.inputs
                    if port.port.removeprefix("in_") != "metrics"
                }
            )
            needs_repair = metric.needs_repair
            outcome = (
                "stop" if not needs_repair else "again" if self.iteration < len(self.iteration_slots) else "exhausted"
            )
        await self.publish(bound, values, outcome)
        if name == "prefix":
            self.prefix_observed.add(association)

    def _metric(self, bound: AssociationInput, iteration: int) -> EvaluationMetricsValue:
        from anonymizer.engine.rewrite.evaluate import (
            compute_any_high_leaked,
            compute_leakage_mass,
            compute_utility_score,
            compute_weighted_leakage_rate,
            determine_repair_needs,
        )

        inputs = {port.port.removeprefix("in_"): port.value for port in bound.inputs}
        qa = cast(RewriteQAValue, inputs["qa"])
        privacy_qa = parse_privacy_qa({"items": [item._asdict() for item in qa.privacy]})
        privacy_answers = parse_privacy_answers(
            {"answers": [item._asdict() for item in cast(PrivacyAnswersValue, inputs["privacy"]).answers]}
        )
        comparisons = cast(QualityComparisonValue, inputs["comparison"]).comparisons
        importance = {item.id: item.importance for item in qa.quality}
        leakage = compute_leakage_mass(privacy_answers, privacy_qa, self.criteria.sensitivity_weights)
        high = compute_any_high_leaked(privacy_answers, privacy_qa)
        repairable = _repairable(
            combined=self.combined, ready=_replacement_ready(cast(RewriteCandidateValue, inputs["baseline"]))
        )
        metric = EvaluationMetricsValue(
            utility=compute_utility_score(
                [item.score for item in comparisons], [importance.get(item.id, "important") for item in comparisons]
            ),
            leakage=leakage,
            weighted_leakage=compute_weighted_leakage_rate(leakage, privacy_qa, self.criteria.sensitivity_weights),
            high_leak=high,
            needs_repair=repairable
            and determine_repair_needs(
                any_high_leaked=high,
                leakage_mass=leakage,
                repair_any_high_leak=self.criteria.repair_any_high_leak,
                effective_threshold=self.criteria.repair_threshold,
            ),
            iteration=iteration,
        )
        return metric

    async def publish(
        self,
        bound: AssociationInput,
        values: dict[str, ArtifactValue],
        outcome: str = "ok",
        *,
        assessments: tuple[LocalAssessmentResult, ...] = (),
    ) -> None:
        assert isinstance(bound.association, SemanticAssociation)
        result = await self._call(
            self.requests.complete_operation(
                bound.association,
                completed(
                    bound.association,
                    (
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
                    outcome,
                    assessments,
                ),
            )
        )
        if result.condition != "result" or result.category != "success":
            raise AnonymizerWorkflowError("Rewrite semantic output was rejected")


class _StageArguments(TypedDict):
    source_frame: WorkflowSourceFrame
    rewrite_binding: RewriteEvaluationBinding


def _replacement_ready(baseline: RewriteCandidateValue) -> bool:
    """Whether the rewrite baseline exists, i.e. its required replacements were applied.

    `_prepare_rewrite_tagged_text` leaves the baseline unset exactly when the
    replacement map could not be applied. Such a rewrite is withheld: the legacy
    loop never repairs it (a repair would start from text that still carries the
    original entities), and its rewritten text stays unavailable.
    """
    return isinstance(baseline.project(), str)


def _repairable(*, combined: bool, ready: bool) -> bool:
    """Whether a failing evaluation may send the rewrite to repair, as released.

    Repair follows the replacement baseline, never the candidate text: a blank or missing candidate
    whose replacements were applied is repaired like any other failing rewrite, on both schedules
    (how the missing cell is encoded, `None` or NaN, depends on the schedule and the pandas version).
    A withheld rewrite (no baseline) is never repaired by the staged loop and keeps its text null,
    while the combined graph repairs and publishes it like any other failing rewrite.
    """
    return combined or ready


def _withhold_rewritten_text(frame: pd.DataFrame, unavailable: pd.Series) -> None:
    """Null the rewritten text of withheld rows without assigning into an Arrow `null` column.

    When every rewrite in a DataDesigner batch is withheld, the rewritten column
    arrives as `null[pyarrow]`; a masked assignment into it aborts the process
    inside pyarrow. That column already holds only nulls, so give it the
    Arrow string dtype a partially withheld batch has: that of the batch's
    source text (`string[pyarrow]` under pandas 2, `large_string[pyarrow]`
    under pandas 3).
    """
    column = frame[COL_REWRITTEN_TEXT]
    if isinstance(column.dtype, pd.ArrowDtype) and pa.types.is_null(column.dtype.pyarrow_dtype):
        text = frame[COL_TEXT].dtype if COL_TEXT in frame.columns else None
        dtype = (
            text
            if isinstance(text, pd.ArrowDtype)
            and (pa.types.is_string(text.pyarrow_dtype) or pa.types.is_large_string(text.pyarrow_dtype))
            else pd.ArrowDtype(pa.string())
        )
        frame[COL_REWRITTEN_TEXT] = pd.Series(pd.array([None] * len(frame), dtype=dtype), index=frame.index)
        return


def _restore_arrow_dtypes(projected: pd.DataFrame, frame: pd.DataFrame) -> None:
    """Keep DataDesigner's Arrow-backed column dtypes across row-wise reassembly.

    `finish` rebuilds the published frame from per-row dicts, which decays
    Arrow-backed columns to NumPy/object dtypes. The legacy Rewrite path
    returned DataDesigner's Arrow-backed frames as-is, so restore each column
    whose source dtype was Arrow-backed. Values are unchanged; missing values
    become Arrow nulls (`pd.NA`), as in the legacy output. The rewritten text
    keeps its observed candidate values here; frame publication presents it
    with `rewritten_text_presentation_dtype`.
    """
    for column in projected.columns:
        dtype = frame[column].dtype if column in frame.columns else None
        if column != COL_REWRITTEN_TEXT and isinstance(dtype, pd.ArrowDtype) and projected[column].dtype != dtype:
            values = [None if _is_missing_scalar(value) else value for value in projected[column].tolist()]
            projected[column] = pd.Series(pd.array(values, dtype=dtype), index=projected.index, name=column)


def _restore_object_dtypes(
    projected: pd.DataFrame, frame: pd.DataFrame, rows: list[dict[str, Any]], held_missing: frozenset[str] = frozenset()
) -> None:
    """Keep object columns holding `None` cells object across row-wise reassembly.

    The legacy Rewrite path concatenated its partition frames, so an object
    column with `None` cells (e.g. withheld baselines beside passthrough NaN)
    stayed object with its cells as-is. Rebuilding from per-row dicts re-infers
    it: `None` and NaN become float64 NaN, and pandas 3 infers `str` for text.
    A column that held missing cells when the evaluate-repair loop began
    (`held_missing`) stays object after repair fills them: the legacy loop
    assigned the repaired text into that object column.
    """
    for column in projected.columns:
        if (
            column in frame.columns
            and frame[column].dtype == object
            and projected[column].dtype != object
            and (column in held_missing or any(row[column] is None for row in rows))
        ):
            projected[column] = pd.Series([row[column] for row in rows], index=projected.index, dtype=object)


def _is_missing_scalar(value: object) -> bool:
    """Whether a cell is a scalar missing marker (`None`, NaN, `pd.NA`, NaT).

    Row-wise reassembly decays a string column to pandas 3's `str` dtype, whose
    missing marker is NaN; Arrow rejects NaN in a string array. pandas 2
    converted every scalar missing marker to an Arrow null, so pass `None`.
    """
    return pd.api.types.is_scalar(value) and bool(pd.isna(value))


def _released_cell(candidate: object, cell: object) -> object:
    """Present an observed rewrite candidate as the released frame cell holding it.

    The legacy loop returned its frame cell, after its own dtype changes (an
    integer candidate becomes a float once a withheld row is nulled). Keep that
    cell whenever it holds the admitted candidate's value; otherwise publish the
    candidate.
    """
    if not (pd.api.types.is_scalar(candidate) and pd.api.types.is_scalar(cell)):
        return candidate
    if _is_missing_scalar(candidate) or _is_missing_scalar(cell):
        return cell if _is_missing_scalar(candidate) and _is_missing_scalar(cell) else candidate
    if isinstance(candidate, str) != isinstance(cell, str):
        return candidate
    try:
        return cell if bool(candidate == cell) else candidate
    except (TypeError, ValueError):
        return candidate


def rewritten_text_presentation_dtype(frame: pd.DataFrame) -> dict[str, pd.ArrowDtype]:
    """Return the legacy Arrow-backed presentation dtype of the rewritten text, if any."""
    dtype = frame[COL_REWRITTEN_TEXT].dtype if COL_REWRITTEN_TEXT in frame.columns else None
    return {COL_REWRITTEN_TEXT: dtype} if isinstance(dtype, pd.ArrowDtype) else {}
