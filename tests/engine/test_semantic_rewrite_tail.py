# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Literal ownership controls for the incomplete private Rewrite tail consumer."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from threading import Event
from typing import cast

import pandas as pd
import pytest
from data_designer.engine.dataset_builders.utils.async_concurrency import ensure_async_engine_loop

from anonymizer import Rewrite
from anonymizer.engine.constants import COL_TEXT
from anonymizer.engine.graph_sdk._artifact_shape import ArtifactShape
from anonymizer.engine.graph_sdk._effect_values import EffectRejected
from anonymizer.engine.graph_sdk._ingress import inspect_artifact
from anonymizer.engine.graph_sdk.blocking import BlockingPool
from anonymizer.engine.graph_sdk.capabilities import FrozenConfig
from anonymizer.engine.graph_sdk.executor import ExecutionLimits, LocalCompleted
from anonymizer.engine.graph_sdk.requests import AssociationResult
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.rewrite._semantic_profile import execute_initial_evaluation
from anonymizer.engine.rewrite.artifacts import (
    EvaluationMetricsValue,
    QualityComparison,
    QualityComparisonValue,
    QualityQuestion,
    RewriteQAValue,
)
from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow


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


@pytest.mark.parametrize("finish", ["normal", "outer_failure", "after_metrics", "partial", "rollback", "cancel"])
def test_prefix_publication_keeps_whole_submission_alive(monkeypatch, finish):
    asyncio.run(_lifecycle(monkeypatch, finish))


async def _lifecycle(monkeypatch, finish):
    entered, release = Event(), Event()
    captured = {}

    def runner(self, source, **kwargs):
        binding = self._semantic_binding
        captured["calls"] = captured.get("calls", 0) + 1
        captured["rows"] = len(source)

        async def publish():
            for association in binding.source.associations:
                row = {"_quality_qa": {"items": []}, "_privacy_qa": {"items": []}, binding.state.rewritten_text: "safe"}
                await binding.after(binding.state.rewritten_text, association, row, prefix=True)
            runtime = binding.requests.owner.runtime
            captured["prefix_refs"] = tuple(runtime.facts.produced.values())
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
        assert release.wait(5)
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

    monkeypatch.setattr(RewriteWorkflow, "run", runner)
    pool = BlockingPool(max_workers=1)
    task = asyncio.create_task(
        execute_initial_evaluation(
            pd.DataFrame({COL_TEXT: ["same", "same"]}),
            adapter=cast(NddAdapter, object()),
            combined=False,
            arguments={"evaluation": Rewrite(max_repair_iterations=0).evaluation},
            configuration=FrozenConfig(fields=()),
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
            assert len(result.record.terminals) == 10 and all(state.complete for state in result.states)
            return
        assert await asyncio.to_thread(entered.wait, 5)
        assert captured["calls"] == 1 and captured["rows"] == 2
        assert len(set(captured["prefix_refs"])) == 4
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
            result = getattr(error.value, "_anonymizer_execution")
        else:
            result = await task
        assert len(result.final_outputs) == (2 if finish == "normal" else 1 if finish == "partial" else 0)
        assert len(result.record.terminals) == 10
        assert all(state.complete for state in result.states)
        assert not result.requests.dispatches and not result.frames
        metrics = [value for _, value in result.artifacts if type(value) is EvaluationMetricsValue]
        assert len(metrics) == (2 if finish in {"normal", "after_metrics"} else 1 if finish == "partial" else 0)
        if finish in {"outer_failure", "after_metrics"}:
            assert any(str(failure.cause) == "literal late callback failure" for failure in result.submission_failures)
    finally:
        release.set()
        await pool.close()
