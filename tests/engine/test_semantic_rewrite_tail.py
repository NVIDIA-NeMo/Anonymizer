# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Literal ownership controls for the incomplete private Rewrite tail consumer."""

from __future__ import annotations

import asyncio
from dataclasses import fields, replace
from threading import Event
from typing import cast

import pandas as pd
import pytest
from data_designer.engine.dataset_builders.utils.async_concurrency import ensure_async_engine_loop

from anonymizer import Rewrite
from anonymizer.engine.constants import COL_ENTITIES_BY_VALUE, COL_REWRITE_BASELINE_TEXT, COL_TEXT
from anonymizer.engine.graph_sdk._artifact_shape import ArtifactShape
from anonymizer.engine.graph_sdk._execution_values import OperationOutputKey
from anonymizer.engine.graph_sdk._ingress import inspect_artifact
from anonymizer.engine.graph_sdk.blocking import BlockingPool
from anonymizer.engine.graph_sdk.capabilities import FrozenConfig
from anonymizer.engine.graph_sdk.evidence import (
    AssessmentSubmission,
    QualificationLimits,
    admit_qualification,
    evidence_revision_view,
    evidence_validity,
    verify_evidence,
)
from anonymizer.engine.graph_sdk.executor import ExecutionLimits, LocalCompleted
from anonymizer.engine.graph_sdk.requests import AssociationResult
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.rewrite._semantic_profile import execute_initial_evaluation
from anonymizer.engine.rewrite.artifacts import (
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
from anonymizer.engine.rewrite.rewrite_workflow import RewriteResult, RewriteWorkflow
from anonymizer.graph import EffectRejected
from tests.retained import retained

# Hang-guard bound for waits on normally-fast in-memory thread signals in this
# module. Not a measurement of expected runtime -- only a generous ceiling so a
# hung test fails instead of blocking CI, chosen to stay robust under a loaded
# CI host.
_HANG_GUARD_TIMEOUT = 30.0


@pytest.mark.parametrize("importance", ["critical", "important"])
def test_closed_qa_preserves_existing_id_and_enum_semantics(importance):
    value = RewriteQAValue(quality=(QualityQuestion(-1, "fact", importance, "Q?", "A"),), privacy=())
    assert inspect_artifact(value)
    with pytest.raises(EffectRejected):
        replace(value, quality=(QualityQuestion(-1, "fact", "NOT_DECLARED", "Q?", "A"),))
    object.__setattr__(value, "shape", ArtifactShape(content_bytes=0, items=1))
    assert not inspect_artifact(value)


def test_closed_comparison_preserves_null_and_recounts_raw_membership():
    value = QualityComparisonValue(comparisons=(QualityComparison(1, 1.0, None),))
    assert value.comparisons[0].reason is None and inspect_artifact(value)
    object.__setattr__(value, "comparisons", value.comparisons * 1000)
    assert not inspect_artifact(value)


@pytest.mark.parametrize("extra", [False, True])
@pytest.mark.parametrize(
    "value, field",
    [
        (RewriteQAValue(quality=(QualityQuestion(1, "fact", "critical", "Q", "A"),), privacy=()), "quality"),
        (
            RewriteQAValue(quality=(), privacy=(PrivacyQuestion(1, "Q", "high", "person", "A", "direct_identifier"),)),
            "privacy",
        ),
        (QualityAnswersValue(answers=(QualityAnswer(1, "A"),)), "answers"),
        (PrivacyAnswersValue(answers=(PrivacyAnswer(1, "no", 1.0, "none", ()),)), "answers"),
        (QualityComparisonValue(comparisons=(QualityComparison(1, 1.0, None),)), "comparisons"),
    ],
)
def test_retained_closed_row_arity_rejects_before_access(value, field, extra):
    assert inspect_artifact(value)
    value = replace(value)
    row = getattr(value, field)[0]
    malformed = tuple.__new__(type(row), tuple(row) + (None,) if extra else ())
    object.__setattr__(value, field, (malformed,))
    assert not inspect_artifact(value)


@pytest.mark.parametrize("finish", ["normal", "outer_failure", "after_metrics", "partial", "rollback", "cancel"])
@pytest.mark.compat
def test_prefix_publication_keeps_whole_submission_alive(monkeypatch, finish):
    asyncio.run(_lifecycle(monkeypatch, finish))


async def _lifecycle(monkeypatch, finish):
    entered, release = Event(), Event()
    captured = {}

    def runner(self, source, *, binding, **kwargs):
        captured["calls"] = captured.get("calls", 0) + 1
        captured["rows"] = len(source)

        async def publish():
            for association in binding.source.associations:
                row = {
                    "_quality_qa": {"items": []},
                    "_privacy_qa": {"items": []},
                    binding.state.rewritten_text: "safe",
                    COL_REWRITE_BASELINE_TEXT: "same",
                }
                await binding.after(binding.state.rewritten_text, association, row, prefix=True)
            runtime = binding.requests.owner.runtime
            captured["prefix_refs"] = tuple(
                fact.artifact for key, fact in runtime.facts.provenance.items() if isinstance(key, OperationOutputKey)
            )
            for association in binding.source.associations:
                with pytest.raises(EffectRejected):
                    await binding._call(
                        binding.requests.complete_operation(
                            association,
                            LocalCompleted(
                                results=(
                                    AssociationResult(
                                        association=association,
                                        outcome="ok",
                                        outputs=(),
                                        consumed_context_ports=frozenset(),
                                    ),
                                )
                            ),
                        )
                    )

        asyncio.run_coroutine_threadsafe(publish(), ensure_async_engine_loop()).result()
        entered.set()
        assert release.wait(_HANG_GUARD_TIMEOUT)
        if finish == "outer_failure":
            raise RuntimeError("literal late callback failure")
        if finish in {"normal", "after_metrics", "partial"}:

            async def complete():
                for association in (
                    binding.source.associations[1:] if finish == "partial" else binding.source.associations
                ):
                    row = {
                        "_quality_qa": {"items": []},
                        "_privacy_qa": {"items": []},
                        binding.state.rewritten_text: "safe",
                        COL_REWRITE_BASELINE_TEXT: "same",
                    }
                    for column, output in (
                        (binding.state.quality_reanswer, {"answers": []}),
                        (binding.state.privacy_reanswer, {"answers": []}),
                        (binding.state.quality_compare, {"per_item": []}),
                    ):
                        row = await binding.before(column, association, row)
                        row[column] = output
                        await binding.after(column, association, row)
                    row = await binding.before(binding.state.utility_score, association, row)
                    row.update(
                        {
                            binding.state.utility_score: 1.0,
                            binding.state.leakage_mass: 0.0,
                            binding.state.weighted_leakage_rate: 0.0,
                            binding.state.any_high_leaked: False,
                            binding.state.needs_repair: False,
                        }
                    )
                    await binding.after(binding.state.needs_repair, association, row)

            asyncio.run_coroutine_threadsafe(complete(), ensure_async_engine_loop()).result()
            if finish == "after_metrics":
                raise RuntimeError("literal late callback failure")
        positions = (1,) if finish == "partial" else (0, 1)
        returned = source.iloc[list(positions)].copy()
        returned[binding.state.rewritten_text] = "safe"
        return RewriteResult(dataframe=returned, failed_records=[], source_positions=positions)

    monkeypatch.setattr(RewriteWorkflow, "_run_lowering", runner)
    monkeypatch.setattr("anonymizer.engine.rewrite.rewrite_workflow._has_entities", lambda value: True)
    pool = BlockingPool(max_workers=1)
    task = asyncio.create_task(
        execute_initial_evaluation(
            pd.DataFrame({COL_TEXT: ["same", "same"], COL_ENTITIES_BY_VALUE: [{}, {}]}),
            adapter=cast(NddAdapter, object()),
            combined=False,
            arguments={"evaluation": Rewrite(max_repair_iterations=0).evaluation},
            pool=pool,
            limits=ExecutionLimits(
                max_local_in_flight=1,
                max_remote_outstanding=None,
                max_runtime_artifacts=2 if finish == "rollback" else 20,
                max_runtime_artifact_bytes=None,
                max_collection_items=10,
            ),
        )
    )
    try:
        if finish == "rollback":
            result = await task
            assert not result.final_outputs and not result.frames
            assert len(result.artifacts) == 2 and result.publication_failures
            assert len(result.record.terminals) == 12 and all(state.complete for state in result.states)
            return
        assert await asyncio.to_thread(entered.wait, 5)
        assert captured["calls"] == 1 and captured["rows"] == 2
        assert len(set(captured["prefix_refs"])) == 6
        assert not task.done()
        if finish == "cancel":
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            assert not task.done()
        release.set()
        if finish == "cancel":
            with pytest.raises(asyncio.CancelledError) as error:
                await task
            result = retained(error.value)
        else:
            result = await task
        assert len(result.final_outputs) == (2 if finish == "normal" else 1 if finish == "partial" else 0)
        assert len(result.record.terminals) == (14 if finish == "normal" else 13 if finish == "partial" else 12)
        assert all(state.complete for state in result.states)
        assert not result.requests.dispatches
        assert len(result.frames) == (1 if finish in {"normal", "partial"} else 0)
        metrics = [value for _, value in result.artifacts if type(value) is EvaluationMetricsValue]
        assert len(metrics) == (4 if finish == "normal" else 2 if finish in {"after_metrics", "partial"} else 0)
        assert len(result.assessments) == (2 if finish == "normal" else 1 if finish == "partial" else 0)
        if result.assessments:
            _assert_final_evidence(result)
        if finish in {"outer_failure", "after_metrics"}:
            assert any(str(failure.cause) == "literal late callback failure" for failure in result.submission_failures)
    finally:
        release.set()
        await pool.close()


def _assert_final_evidence(result):
    limits = QualificationLimits(**{field.name: 1000 for field in fields(QualificationLimits)})
    execution = result._execution
    admitted = admit_qualification(execution=execution, productions=execution.assessment_productions, limits=limits)
    submissions = tuple(AssessmentSubmission(fact=fact) for fact in result.assessments)
    verified = verify_evidence(admitted=admitted, result=result, submissions=submissions)
    assert {item.subject for item in verified} == {item.candidate for item in result.final_outputs}
    for item, fact in zip(verified, result.assessments, strict=True):
        assert len(item.consumed_by_port) == 8

        def view(configuration: FrozenConfig):
            return evidence_revision_view(
                admitted=admitted,
                result=result,
                artifacts=tuple(reference for reference, _ in result.artifacts),
                configurations=((fact.node, configuration),),
            )

        assert evidence_validity(evidence=item, current=view(fact.environment.configuration)) == "current"
        assert evidence_validity(evidence=item, current=view(FrozenConfig(fields=()))) == "stale"
    inspection = admit_qualification(execution=execution, productions=(), limits=limits)
    assert inspection.productions == ()
    extra = replace(execution.assessment_productions[0], promise="undeclared")
    nonlocal_production = replace(
        execution.assessment_productions[0],
        node=next(policy.node for policy in execution.policies if policy.kind == "delegated"),
    )
    for productions in (
        execution.assessment_productions[:1],
        execution.assessment_productions * 2,
        (*execution.assessment_productions, extra),
        (nonlocal_production,),
    ):
        with pytest.raises(EffectRejected):
            admit_qualification(execution=execution, productions=productions, limits=limits)


@pytest.mark.parametrize("opaque", [False, True])
@pytest.mark.parametrize("ceiling", ["max_runtime_artifact_bytes", "max_remote_outstanding"])
def test_unknown_dd_bound_rejects_before_callback(monkeypatch, opaque, ceiling):
    from unittest.mock import Mock

    runner = Mock(side_effect=AssertionError("callback must not enter"))
    monkeypatch.setattr(RewriteWorkflow, "_run_lowering", runner)
    criteria = Rewrite(max_repair_iterations=0).evaluation
    limits = ExecutionLimits(
        max_local_in_flight=1,
        max_remote_outstanding=None,
        max_runtime_artifacts=20,
        max_runtime_artifact_bytes=None,
        max_collection_items=10,
    )

    async def run():
        pool = BlockingPool(max_workers=1)
        try:
            await execute_initial_evaluation(
                pd.DataFrame({COL_TEXT: ["same"], COL_ENTITIES_BY_VALUE: [{}]}),
                adapter=cast(NddAdapter, object()),
                combined=False,
                arguments={"evaluation": criteria},
                pool=pool,
                limits=replace(limits, **{ceiling: 100}),
                opaque=opaque,
            )
        finally:
            await pool.close()

    with pytest.raises(EffectRejected, match="unsupported"):
        asyncio.run(run())
    runner.assert_not_called()
