# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, EvaluateConfig, Rewrite
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import (
    COL_ATTRIBUTE_FIDELITY_VALID,
    COL_DETECTED_ENTITIES,
    COL_DETECTION_JUDGE,
    COL_DETECTION_VALID,
    COL_ENTITIES_BY_VALUE,
    COL_ENTITY_COVERAGE,
    COL_FINAL_ENTITIES,
    COL_JUDGE_EVALUATION,
    COL_MISSED_ENTITIES,
    COL_RELATIONAL_CONSISTENCY_VALID,
    COL_REWRITTEN_TEXT,
    COL_TYPE_FIDELITY_VALID,
    RECORD_ID_COLUMN,
)
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult, EntityDetectionWorkflow
from anonymizer.engine.evaluation.entity_coverage_judge import EntityCoverageWorkflow
from anonymizer.engine.ndd.adapter import WorkflowRunResult
from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow
from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.interface.errors import InvalidInputError
from tests.detector_results import as_detector_result, input_record_ids, returns_detector_result
from tests.engine.test_rewrite_workflow import _PRIVACY_GOAL, _REPLACE_PATCH, _ScriptedAdapter


@pytest.mark.parametrize("method", ["run", "preview"])
def test_input_column_named_like_the_rewritten_text_is_rejected(method: str, tmp_path: Path) -> None:
    """An input column named ``_rewritten_text`` used to be released as the rewritten text, reported ``met``."""
    entities = {"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]}
    detector = returns_detector_result(Mock(spec=EntityDetectionWorkflow))
    detector.run.side_effect = lambda frame, **_: as_detector_result(
        frame,
        EntityDetectionResult(
            dataframe=frame.assign(**{RECORD_ID_COLUMN: ["rec"], COL_ENTITIES_BY_VALUE: [entities]}),
            failed_records=[],
            source_positions=(0,),
        ),
    )
    csv_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Alice works at Acme"], COL_REWRITTEN_TEXT: ["note"]}).to_csv(csv_path, index=False)
    adapter = _ScriptedAdapter({"rec": 0})
    anonymizer = Anonymizer(detection_workflow=detector, rewrite_runner=RewriteWorkflow(adapter=Mock(wraps=adapter)))

    with patch(_REPLACE_PATCH) as mock_replace_cls, pytest.raises(InvalidInputError, match=COL_REWRITTEN_TEXT):
        mock_replace_cls.return_value.generate_map_only.side_effect = lambda frame, **_: Mock(
            dataframe=frame, failed_records=[], source_positions=tuple(range(len(frame)))
        )
        getattr(anonymizer, method)(
            config=AnonymizerConfig(
                rewrite=Rewrite(privacy_goal=_PRIVACY_GOAL, risk_tolerance="minimal"), emit_telemetry=False
            ),
            data=AnonymizerInput(source=str(csv_path), text_column="text"),
        )

    detector.run.assert_not_called()
    assert adapter.workflow_names == []


_ENTITIES = {"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]}
_FINAL = {"entities": [{"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5}]}


def _detector() -> Mock:
    detector = returns_detector_result(Mock(spec=EntityDetectionWorkflow))
    detector.run.side_effect = lambda frame, **_: as_detector_result(
        frame,
        EntityDetectionResult(
            dataframe=frame.assign(
                **{
                    RECORD_ID_COLUMN: ["rec"],
                    COL_ENTITIES_BY_VALUE: [_ENTITIES],
                    COL_FINAL_ENTITIES: [_FINAL],
                    COL_DETECTED_ENTITIES: [_FINAL],
                }
            ),
            failed_records=[],
            source_positions=(0,),
        ),
    )
    return detector


def test_run_does_not_show_input_values_as_evaluation_columns(tmp_path: Path) -> None:
    """Input columns named like evaluation results were passed into ``run()`` results as if ``evaluate()`` had run."""
    evaluation_names = [
        COL_JUDGE_EVALUATION,
        COL_ENTITY_COVERAGE,
        COL_MISSED_ENTITIES,
        COL_TYPE_FIDELITY_VALID,
        COL_RELATIONAL_CONSISTENCY_VALID,
        COL_ATTRIBUTE_FIDELITY_VALID,
    ]
    csv_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Alice works at Acme"], **{name: ["input"] for name in evaluation_names}}).to_csv(
        csv_path, index=False
    )
    anonymizer = Anonymizer(detection_workflow=_detector())

    result = anonymizer.run(
        config=AnonymizerConfig(replace=Redact(), emit_telemetry=False),
        data=AnonymizerInput(source=str(csv_path), text_column="text"),
    )

    assert not set(evaluation_names) & set(result.dataframe.columns)
    assert {f"{name}__input" for name in evaluation_names} <= set(result.trace_dataframe.columns)


def test_input_detection_valid_does_not_replace_the_detection_judge_verdict(tmp_path: Path) -> None:
    """An input ``detection_valid`` of 1.0 kept a failed detection-validity advisory reading ``passed``."""
    csv_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Alice works at Acme"], COL_DETECTION_VALID: [1.0], COL_JUDGE_EVALUATION: ["input"]}).to_csv(
        csv_path, index=False
    )
    adapter = _ScriptedAdapter({input_record_ids(AnonymizerInput(source=str(csv_path), text_column="text"))[0]: 0})
    anonymizer = Anonymizer(detection_workflow=_detector(), rewrite_runner=RewriteWorkflow(adapter=Mock(wraps=adapter)))
    with patch(_REPLACE_PATCH) as mock_replace_cls:
        mock_replace_cls.return_value.generate_map_only.side_effect = lambda frame, **_: Mock(
            dataframe=frame, failed_records=[], source_positions=tuple(range(len(frame)))
        )
        result = anonymizer.run(
            config=AnonymizerConfig(
                rewrite=Rewrite(privacy_goal=_PRIVACY_GOAL, risk_tolerance="minimal"), emit_telemetry=False
            ),
            data=AnonymizerInput(source=str(csv_path), text_column="text"),
        )

    invalid = {"all_valid": False, "invalid_entities": [{"value": "Alice", "label": "first_name", "reasoning": "no"}]}
    judged = {"privacy": {"score": "low"}}

    class JudgeAdapter:
        def run_workflow(self, seed: pd.DataFrame, *, workflow_name: str, **_: object) -> WorkflowRunResult:
            out = seed.copy()
            if workflow_name == "rewrite-final-judge":
                out[COL_JUDGE_EVALUATION] = [judged for _ in range(len(out))]
            if workflow_name == "replace-detection-judge":
                out[COL_DETECTION_JUDGE] = [invalid for _ in range(len(out))]
            return WorkflowRunResult(dataframe=out, failed_records=[], source_positions=tuple(range(len(out))))

    def coverage(self: object, frame: pd.DataFrame, **_: object) -> tuple[pd.DataFrame, list]:
        out = frame.copy()
        out[COL_ENTITY_COVERAGE] = 1.0
        out[COL_MISSED_ENTITIES] = [[] for _ in range(len(out))]
        return out, []

    judge_adapter = Mock(wraps=JudgeAdapter())
    evaluator = Anonymizer(detection_workflow=Mock(), rewrite_runner=RewriteWorkflow(adapter=judge_adapter))
    evaluator._adapter = judge_adapter
    with patch.object(EntityCoverageWorkflow, "run_non_critical", coverage):
        evaluated = evaluator.evaluate(result, config=EvaluateConfig(compute_detection_validity=True))

    assert evaluated.dataframe[COL_DETECTION_VALID].tolist() == [0.0]
    assert evaluated.dataframe[COL_JUDGE_EVALUATION].tolist() == [judged]
