# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mixed detector/native Rewrite entry and retained observation controls."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import get_ident
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Rewrite
from anonymizer.config.models import ReplaceModelSelection, RewriteModelSelection
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_ENTITIES_BY_VALUE,
    COL_LEAKAGE_MASS,
    COL_NEEDS_HUMAN_REVIEW,
    COL_NEEDS_REPAIR,
    COL_REPAIR_ITERATIONS,
    COL_REWRITTEN_TEXT,
    COL_UTILITY_SCORE,
)
from anonymizer.engine.detection._frame_publication import PublishedFrame
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult, EntityDetectionWorkflow
from anonymizer.engine.graph_sdk.executor import ExecutionResult
from anonymizer.engine.ndd.adapter import NddAdapter, WorkflowRunResult
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow
from anonymizer.engine.rewrite._semantic_evaluation import RewriteEvaluationBinding
from anonymizer.engine.rewrite.combined_rewrite_workflow import CombinedRewriteWorkflow
from anonymizer.engine.rewrite.rewrite_workflow import RewriteResult, RewriteWorkflow
from anonymizer.graph._values import InvocationId

pytestmark = pytest.mark.compat


@pytest.fixture
def mixed_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., Any]:
    path = tmp_path / "source.csv"
    pd.DataFrame({"text": ["Alice works.", "Alice reads."]}).to_csv(path, index=False)
    monkeypatch.setattr(
        LlmReplaceWorkflow,
        "generate_map_only",
        lambda self, frame, **kw: WorkflowRunResult(frame.copy(), [], tuple(range(len(frame)))),
    )

    def run(
        case: str,
        combined: bool,
        candidate: object = "safe",
        selected: tuple[int, ...] = (0, 1),
        actual_combined: bool | None = None,
    ) -> tuple[Any, list[InvocationId], list[int], int, Mock]:
        strategy = Rewrite(use_combined_graph=combined, max_repair_iterations=0)
        caller = get_ident()
        threads = []
        adapter = Mock(spec=NddAdapter)
        kind = (
            CombinedRewriteWorkflow if (combined if actual_combined is None else actual_combined) else RewriteWorkflow
        )
        runner = None
        if case == "inherited_subclass":

            class InheritedRewrite(RewriteWorkflow):
                pass

            class InheritedCombined(CombinedRewriteWorkflow):
                pass

            runner = (InheritedCombined if kind is CombinedRewriteWorkflow else InheritedRewrite)(adapter)
            facade = (
                Anonymizer(combined_rewrite_runner=cast(CombinedRewriteWorkflow, runner))
                if combined
                else Anonymizer(rewrite_runner=runner)
            )
            detector = facade._detection_workflow
        else:
            detector = (
                Mock(spec=EntityDetectionWorkflow) if case == "mock_detection" else EntityDetectionWorkflow(adapter)
            )
            facade = Anonymizer(detection_workflow=detector)
            facade._adapter.run_workflow = adapter.run_workflow

        # Caller callbacks need not accept the internal source_frame keyword.
        def detect(
            frame: pd.DataFrame,
            *,
            model_configs: Any,
            selected_models: Any,
            gliner_detection_threshold: Any,
            validation_max_entities_per_call: Any,
            validation_excerpt_window_chars: Any,
            entity_labels: Any,
            excluded_entity_labels: Any,
            entity_label_examples: Any,
            privacy_goal: Any,
            data_summary: Any,
            tag_latent_entities: Any,
            compute_grouped_entities: Any,
            preview_num_records: Any,
        ) -> EntityDetectionResult:
            assert privacy_goal is strategy.privacy_goal
            assert model_configs is facade._model_configs
            assert selected_models is facade._selected_models.detection
            threads.append(get_ident())
            out = frame.iloc[list(selected)].copy()
            out[COL_ENTITIES_BY_VALUE] = [{"entities_by_value": [{"value": "Alice", "labels": ["person"]}]}] * len(out)
            return EntityDetectionResult(out, [], selected)

        monkeypatch.setattr(detector, "run", detect)

        def response(frame: pd.DataFrame, **kwargs: Any) -> WorkflowRunResult:
            threads.append(get_ident())
            out = frame.copy()
            out[COL_REWRITTEN_TEXT] = pd.Series([candidate] * len(out), index=out.index, dtype=object)
            for column, value in (
                (COL_UTILITY_SCORE, 0.9),
                (COL_LEAKAGE_MASS, 0.0),
                (COL_ANY_HIGH_LEAKED, False),
                (COL_NEEDS_REPAIR, False),
                (COL_REPAIR_ITERATIONS, 0),
                (COL_NEEDS_HUMAN_REVIEW, False),
            ):
                out[column] = value
            return WorkflowRunResult(out, [], tuple(range(len(out))))

        adapter.run_workflow.side_effect = response
        alias = facade._model_configs[0].alias
        facade._selected_models.replace = ReplaceModelSelection(replacement_generator=alias)
        facade._selected_models.rewrite = RewriteModelSelection(
            **{key: alias for key in RewriteModelSelection.model_fields}
        )
        allocated = []
        original = InvocationId.new

        def allocate(*args: Any, **kwargs: Any) -> InvocationId:
            value = original(*args, **kwargs)
            allocated.append(value)
            return value

        with monkeypatch.context() as patch:
            patch.setattr(InvocationId, "new", allocate)
            result = facade.run(
                config=AnonymizerConfig(rewrite=strategy, emit_telemetry=False),
                data=AnonymizerInput(source=str(path)),
            )
        return result, allocated, threads, caller, adapter

    return run


@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize("case", ["mock_detection", "exact_detection", "inherited_subclass"])
def test_mixed_native_routes_keep_one_source_bound_owner(
    mixed_run: Callable[..., Any], case: str, combined: bool
) -> None:
    result, allocated, threads, caller, adapter = mixed_run(case, combined)
    assert result.dataframe["text_rewritten"].tolist() == ["safe", "safe"]
    assert len(allocated) == 1
    execution = result._execution
    assert len(execution.frames) == 1
    assert isinstance(execution.frames[0], PublishedFrame)
    assert execution.frames[0].project(execution)[COL_REWRITTEN_TEXT].tolist() == ["safe", "safe"]
    assert threads and set(threads) == {caller}
    assert not execution.requests.dispatches
    assert adapter.run_workflow.call_count == (1 if combined else 2)


@pytest.mark.parametrize("combined", [False, True])
def test_mixed_detection_reorder_drop_keeps_source_positions(mixed_run: Callable[..., Any], combined: bool) -> None:
    result, allocated, _, _, _ = mixed_run("mock_detection", combined, selected=(1,))
    assert len(allocated) == 1
    assert result.dataframe["text"].tolist() == ["Alice reads."]
    execution = result._execution
    published = execution.frames[0]
    assert isinstance(published, PublishedFrame)
    assert len(published._sources) == 1
    assert published.project(execution)[COL_REWRITTEN_TEXT].tolist() == ["safe"]


@pytest.mark.parametrize("combined", [False, True])
def test_inherited_receiver_uses_its_actual_schedule(mixed_run: Callable[..., Any], combined: bool) -> None:
    result, allocated, _, _, adapter = mixed_run("inherited_subclass", combined, actual_combined=not combined)
    assert result.dataframe["text_rewritten"].tolist() == ["safe", "safe"]
    assert len(allocated) == 1
    assert adapter.run_workflow.call_count == (2 if combined else 1)


@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize("mutation", ["candidate", "dtype", "dtype_numeric", "shape"])
def test_retained_observation_mutation_rejects(mixed_run: Callable[..., Any], combined: bool, mutation: str) -> None:
    result, _, _, _, _ = mixed_run("mock_detection", combined, candidate=42 if mutation == "dtype_numeric" else "123")
    execution = result._execution
    published = execution.frames[0]
    if mutation in {"candidate", "shape"}:
        ref = dict(published._text)[COL_REWRITTEN_TEXT][0]
        value = dict(execution.artifacts)[ref]
        if mutation == "candidate":
            object.__setattr__(value, "value", "456")  # Same type and byte length.
        else:
            from anonymizer.engine.graph_sdk._artifact_shape import ArtifactShape

            object.__setattr__(value, "shape", ArtifactShape(content_bytes=0, items=1))
    else:
        object.__setattr__(
            published,
            "_observation_dtypes",
            ((COL_REWRITTEN_TEXT, np.dtype("float64" if mutation == "dtype_numeric" else "int64")),),
        )
    with pytest.raises((ValueError, RuntimeError)):
        published.project(execution)


@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize("representation", ["dict", "model", "numpy", "parquet"])
@pytest.mark.parametrize("mutated", [False, True])
def test_structured_source_admission_and_mutation(
    monkeypatch: pytest.MonkeyPatch, combined: bool, representation: str, mutated: bool
) -> None:
    from anonymizer.config.rewrite import EvaluationCriteria, PrivacyGoal
    from anonymizer.engine.constants import COL_TEXT
    from anonymizer.engine.graph_sdk._effect_values import EffectRejected
    from anonymizer.engine.ndd.model_loader import load_default_model_selection

    frame = pd.DataFrame(
        {
            COL_TEXT: ["Alice"],
            COL_ENTITIES_BY_VALUE: [{"entities_by_value": [{"value": "Alice", "labels": ["person"]}]}],
        }
    )
    from anonymizer.engine.schemas.detection import EntitiesByValueSchema

    if representation == "model":
        frame.iat[0, 1] = EntitiesByValueSchema.model_validate(frame.iat[0, 1])
    elif representation == "numpy":
        frame.iat[0, 1] = np.array(frame.iat[0, 1], dtype=object)
    elif representation == "parquet":
        from anonymizer.engine.constants import COL_LATENT_ENTITIES

        frame.iat[0, 1] = {
            "entities_by_value": np.array(
                [{"value": "Alice", "labels": np.array(["person"], dtype=object)}], dtype=object
            )
        }
        frame[COL_LATENT_ENTITIES] = [{"latent_entities": np.array([], dtype=object)}]
    opaque = object()
    frame["caller_object"] = [opaque]
    metadata = {"caller": [opaque]}
    frame["caller_metadata"] = [metadata]
    adapter = Mock(spec=NddAdapter)
    runner = (CombinedRewriteWorkflow if combined else RewriteWorkflow)(adapter)
    lower = runner._run_lowering

    def changed(frame: pd.DataFrame, *, binding: RewriteEvaluationBinding, **kwargs: Any) -> RewriteResult:
        assert binding.source.dataframe["caller_object"].iloc[0] is opaque
        assert binding.source.dataframe["caller_metadata"].iloc[0] is metadata
        assert binding.source.seed()["caller_metadata"].iloc[0] is metadata
        payload = frame[COL_ENTITIES_BY_VALUE].iloc[0]
        if mutated:
            if representation == "model":
                payload.entities_by_value[0].value = "Mallory"
            else:
                (payload.item() if representation == "numpy" else payload)["entities_by_value"][0]["value"] = "Mallory"
        from anonymizer.engine.rewrite.parsers import normalize_payload

        assert (
            normalize_payload(binding.source.dataframe[COL_ENTITIES_BY_VALUE].iloc[0])["entities_by_value"][0]["value"]
            == "Alice"
        )
        return lower(frame, binding=binding, **kwargs)

    monkeypatch.setattr(runner, "_run_lowering", changed)
    replace = Mock(side_effect=lambda rows, **kw: WorkflowRunResult(rows.copy(), [], tuple(range(len(rows)))))
    monkeypatch.setattr(LlmReplaceWorkflow, "generate_map_only", replace)

    def response(rows: pd.DataFrame, **kwargs: Any) -> WorkflowRunResult:
        out = rows.copy()
        for column, value in (
            (COL_REWRITTEN_TEXT, "safe"),
            (COL_UTILITY_SCORE, 0.9),
            (COL_LEAKAGE_MASS, 0.0),
            (COL_ANY_HIGH_LEAKED, False),
            (COL_NEEDS_REPAIR, False),
            (COL_NEEDS_HUMAN_REVIEW, False),
        ):
            out[column] = value
        return WorkflowRunResult(out, [], tuple(range(len(out))))

    adapter.run_workflow.side_effect = response
    models = load_default_model_selection()
    arguments: dict[str, Any] = dict(
        model_configs=[],
        selected_models=models.rewrite,
        replace_model_selection=models.replace,
        privacy_goal=PrivacyGoal(protect="Protect all names", preserve="Preserve original meaning"),
        evaluation=EvaluationCriteria(max_repair_iterations=0),
    )
    if mutated:
        with pytest.raises(EffectRejected):
            runner.run(frame, **arguments)
        assert not replace.called and not adapter.run_workflow.called
    else:
        result = runner.run(frame, **arguments)
        assert result.dataframe[COL_REWRITTEN_TEXT].tolist() == ["safe"]
        assert adapter.run_workflow.called
        assert result.dataframe["caller_object"].iloc[0] is opaque
        assert result.dataframe["caller_metadata"].iloc[0] is metadata


@pytest.mark.parametrize("combined", [False, True])
def test_cancel_during_opaque_detection_drains_one_owner_before_rewrite(
    monkeypatch: pytest.MonkeyPatch, combined: bool
) -> None:
    import asyncio
    from threading import Event

    from anonymizer.engine.constants import COL_TEXT
    from anonymizer.engine.graph_sdk.blocking import BlockingPool
    from anonymizer.engine.graph_sdk.executor import ExecutionLimits
    from anonymizer.engine.rewrite._semantic_profile import _configuration, execute_initial_evaluation

    entered, release, exited = Event(), Event(), Event()
    source = pd.DataFrame({COL_TEXT: ["Alice"]})
    detection = Mock(spec=EntityDetectionWorkflow)

    def detect(frame: pd.DataFrame) -> EntityDetectionResult:
        entered.set()
        try:
            assert release.wait(5)
            result = frame.copy()
            result[COL_ENTITIES_BY_VALUE] = [{"entities_by_value": [{"value": "Alice", "labels": ["person"]}]}]
            return EntityDetectionResult(result, [], (0,))
        finally:
            exited.set()

    detection.run = detect
    adapter = Mock(spec=NddAdapter)
    allocated = []
    original = InvocationId.new

    def allocate(*args: Any, **kwargs: Any) -> InvocationId:
        value = original(*args, **kwargs)
        allocated.append(value)
        return value

    monkeypatch.setattr(InvocationId, "new", allocate)

    async def operation() -> None:
        pool = BlockingPool(max_workers=1)
        arguments = {"evaluation": Rewrite(max_repair_iterations=0).evaluation}
        task = asyncio.create_task(
            execute_initial_evaluation(
                source,
                adapter=adapter,
                combined=combined,
                arguments=arguments,
                configuration=_configuration(arguments, {}),
                pool=pool,
                limits=ExecutionLimits(
                    max_local_in_flight=1,
                    max_remote_outstanding=None,
                    max_runtime_artifacts=20,
                    max_runtime_artifact_bytes=None,
                    max_collection_items=10,
                ),
                detection=detection,
                detection_arguments={},
                opaque=True,
            )
        )
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0.01)
                assert not task.done() and not exited.is_set()
            release.set()
            with pytest.raises(asyncio.CancelledError) as error:
                await task
            execution = getattr(error.value, "_anonymizer_execution")
            assert exited.is_set() and len(allocated) == 1
            assert not adapter.run_workflow.called
            assert not execution.frames and not execution.final_outputs
            assert not execution.requests.dispatches
            assert all(state.complete for state in execution.states)
        finally:
            release.set()
            await pool.close()

    asyncio.run(operation())


@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize("mutated", [False, True])
def test_opaque_detector_with_native_adapter_preserves_caller_and_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, combined: bool, mutated: bool
) -> None:
    from data_designer.interface.data_designer import DataDesigner

    from anonymizer.engine.graph_sdk._effect_values import EffectRejected

    path = tmp_path / "source.csv"
    pd.DataFrame({"text": ["No sensitive entities."]}).to_csv(path, index=False)
    strategy = Rewrite(use_combined_graph=combined, max_repair_iterations=0)
    detector = Mock(spec=EntityDetectionWorkflow)
    facade = Anonymizer(detection_workflow=detector)
    caller = get_ident()

    def detect(frame: pd.DataFrame, **kwargs: Any) -> EntityDetectionResult:
        assert get_ident() == caller
        assert kwargs["privacy_goal"] is strategy.privacy_goal
        assert "source_frame" not in kwargs
        if mutated:
            kwargs["privacy_goal"].preserve = "Changed after admitted detection"
        result = frame.copy()
        result[COL_ENTITIES_BY_VALUE] = [{"entities_by_value": []}]
        return EntityDetectionResult(result, [], (0,))

    detector.run = detect

    def unexpected(*args: Any, **kwargs: Any) -> None:
        pytest.fail("no-entity or rejected source reached DataDesigner")

    monkeypatch.setattr(DataDesigner, "create", unexpected)
    monkeypatch.setattr(DataDesigner, "preview", unexpected)
    arguments: dict[str, Any] = dict(
        config=AnonymizerConfig(rewrite=strategy, emit_telemetry=False), data=AnonymizerInput(source=str(path))
    )
    if mutated:
        with pytest.raises(EffectRejected):
            facade.run(**arguments)
    else:
        result = facade.run(**arguments)
        assert result.dataframe["text_rewritten"].tolist() == ["No sensitive entities."]
        execution = cast(ExecutionResult, result._execution)
        assert len(execution.frames) == 1 and isinstance(execution.frames[0], PublishedFrame)
        assert not execution.requests.dispatches
