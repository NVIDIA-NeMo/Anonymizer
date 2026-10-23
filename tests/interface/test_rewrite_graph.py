# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Both existing Rewrite schedules share one nonqualifying invocation owner."""

from __future__ import annotations

import pandas as pd
import pytest

from anonymizer import AnonymizerConfig, AnonymizerInput, Rewrite
from anonymizer.engine.constants import COL_REWRITTEN_TEXT
from tests.engine import test_detection_source_frame as detection_fixtures
from tests.engine.test_detection_source_frame import public_anonymizer
from tests.retained import retained

scripted_detection = detection_fixtures.scripted_detection


@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize(
    "texts,preview,error",
    [
        ([], False, "Failed to load generated dataset"),
        (["Alice travels."], True, "Dataset is empty"),
        (["Alice fails."], False, "scripted row failure"),
    ],
)
def test_builtin_rewrite_zero_source_and_all_failed_execution(
    scripted_detection, tmp_path, combined, texts, preview, error
):
    from anonymizer.config.models import ReplaceModelSelection, RewriteModelSelection
    from anonymizer.interface.errors import AnonymizerWorkflowError

    facade = public_anonymizer(scripted_detection)
    facade._selected_models.replace = ReplaceModelSelection(replacement_generator="singleton")
    facade._selected_models.rewrite = RewriteModelSelection(
        **{key: "singleton" for key in RewriteModelSelection.model_fields}
    )
    path = tmp_path / "rewrite.csv"
    pd.DataFrame({"text": pd.Series(texts, dtype=str)}).to_csv(path, index=False)
    config = AnonymizerConfig(rewrite=Rewrite(use_combined_graph=combined), emit_telemetry=False)
    with pytest.raises(AnonymizerWorkflowError, match=error) as caught:
        options = dict(config=config, data=AnonymizerInput(source=str(path)))
        facade.preview(**options, num_records=0) if preview else facade.run(**options)
    execution = retained(caught.value)
    assert not execution.frames and not execution.final_outputs
    assert not execution.requests.dispatches
    assert len(execution.record.targets) == (0 if preview or not texts else 1)


@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize("order", ["before_delegate", "after_delegate"])
def test_caller_detector_mutating_privacy_goal_runs_on_the_admitted_goal(
    tmp_path, monkeypatch, scripted_detection, combined, order
):
    """C1: a caller detector that mutates rewrite.privacy_goal and delegates to builtin detection succeeds;
    Rewrite, result.rewrite_config and evaluate() all use the goal admitted with the run."""
    import json
    from unittest.mock import Mock

    import httpx

    from anonymizer.config.models import EvaluateModelSelection, ReplaceModelSelection, RewriteModelSelection
    from anonymizer.engine.constants import (
        COL_ANY_HIGH_LEAKED,
        COL_LEAKAGE_MASS,
        COL_NEEDS_HUMAN_REVIEW,
        COL_NEEDS_REPAIR,
        COL_REPAIR_ITERATIONS,
        COL_UTILITY_SCORE,
    )
    from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
    from anonymizer.engine.ndd.adapter import WorkflowRunResult
    from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow
    from anonymizer.engine.rewrite.combined_rewrite_workflow import CombinedRewriteWorkflow
    from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow

    marker = "MUTATED-BY-CALLER-DETECTOR"
    prompts = []
    adapter = Mock()

    def response(frame, **kwargs):
        prompts.append(json.dumps([column.model_dump(mode="json") for column in kwargs["columns"]]))
        result = frame.copy()
        result[COL_REWRITTEN_TEXT] = "observed safe"
        result[COL_UTILITY_SCORE] = 0.9
        result[COL_LEAKAGE_MASS] = 0.0
        result[COL_ANY_HIGH_LEAKED] = False
        result[COL_NEEDS_REPAIR] = False
        result[COL_REPAIR_ITERATIONS] = 0
        result[COL_NEEDS_HUMAN_REVIEW] = False
        return WorkflowRunResult(result, [], tuple(range(len(result))))

    adapter.run_workflow.side_effect = response
    runner = (CombinedRewriteWorkflow if combined else RewriteWorkflow)(adapter)

    class MutatingDetector(EntityDetectionWorkflow):
        def run(self, dataframe, **kwargs):  # type: ignore[override]
            if order == "before_delegate":
                kwargs["privacy_goal"].preserve = marker
            detected = super().run(dataframe, **kwargs)
            if order == "after_delegate":
                kwargs["privacy_goal"].preserve = marker
            return detected

    base = public_anonymizer(scripted_detection)
    facade = public_anonymizer(
        scripted_detection,
        detection_workflow=MutatingDetector(base._adapter),
        **{"combined_rewrite_runner" if combined else "rewrite_runner": runner},
    )
    facade._selected_models.replace = ReplaceModelSelection(replacement_generator="singleton")
    facade._selected_models.rewrite = RewriteModelSelection(
        **{key: "singleton" for key in RewriteModelSelection.model_fields}
    )
    monkeypatch.setattr(
        LlmReplaceWorkflow,
        "generate_map_only",
        lambda self, frame, **kwargs: WorkflowRunResult(frame.copy(), [], tuple(range(len(frame)))),
    )
    from tests.engine.test_detection_source_frame import _modify_dd_responses

    def latent(request, answer):
        if "LatentEntitiesSchema" not in request.content.decode():
            return answer
        payload = answer.json()
        payload["choices"][0]["message"]["content"] = '```json\n{"latent_entities": []}\n```'
        return httpx.Response(200, json=payload)

    _modify_dd_responses(scripted_detection, monkeypatch, latent)
    strategy = Rewrite(use_combined_graph=combined, max_repair_iterations=0)
    assert strategy.privacy_goal is not None
    admitted = strategy.privacy_goal.model_copy(deep=True)
    path = tmp_path / "goal.csv"
    pd.DataFrame({"text": ["Alice travels."]}).to_csv(path, index=False)
    result = facade.run(
        config=AnonymizerConfig(rewrite=strategy, emit_telemetry=False), data=AnonymizerInput(source=str(path))
    )
    assert strategy.privacy_goal.preserve == marker  # the caller's own object changed
    assert result.rewrite_config == admitted and result.rewrite_config.preserve != marker
    assert all(marker not in prompt for prompt in prompts)

    judged = []

    class Stop(Exception):
        pass

    def evaluate(*args, **kwargs):
        judged.append(kwargs["privacy_goal"])
        raise Stop

    monkeypatch.setattr(facade._rewrite_runner, "evaluate", evaluate)
    facade._selected_models.evaluate = EvaluateModelSelection(
        **{key: "singleton" for key in EvaluateModelSelection.model_fields}
    )
    with pytest.raises(Stop):
        facade.evaluate(result)
    assert judged == [admitted]
