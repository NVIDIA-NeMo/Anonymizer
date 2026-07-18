# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import re
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, EvaluateConfig, Rewrite
from anonymizer.config.replace_strategies import Redact, Substitute
from anonymizer.engine.constants import (
    COL_ATTRIBUTE_FIDELITY_VALID,
    COL_DETECTED_ENTITIES,
    COL_DETECTION_VALID,
    COL_ENTITIES_BY_VALUE,
    COL_ENTITY_COVERAGE,
    COL_FINAL_ENTITIES,
    COL_JUDGE_EVALUATION,
    COL_RELATIONAL_CONSISTENCY_VALID,
    COL_REPLACED_TEXT,
    COL_REWRITTEN_TEXT,
    COL_TEXT,
    COL_TYPE_FIDELITY_VALID,
)
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult, EntityDetectionWorkflow
from anonymizer.engine.evaluation.entity_coverage_judge import EntityCoverageWorkflow
from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN, FailedRecord, NddAdapter
from anonymizer.engine.replace.replace_runner import ReplacementResult, ReplacementWorkflow
from anonymizer.engine.rewrite.rewrite_workflow import RewriteResult, RewriteWorkflow
from anonymizer.interface.anonymizer import Anonymizer


@pytest.fixture
def stub_input(tmp_path: Path) -> AnonymizerInput:
    csv_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Alice works at Acme", "Bob likes cats"]}).to_csv(csv_path, index=False)
    return AnonymizerInput(source=str(csv_path))


def _make_logging_anonymizer(
    *,
    detection_entities: list[list[dict]] | None = None,
    detection_failures: list[FailedRecord] | None = None,
    replace_failures: list[FailedRecord] | None = None,
) -> Anonymizer:
    entities = detection_entities or [
        [{"value": "Alice", "label": "first_name"}, {"value": "Acme", "label": "organization"}],
        [{"value": "Bob", "label": "first_name"}],
    ]
    det_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme", "Bob likes cats"],
            COL_DETECTED_ENTITIES: entities,
            COL_FINAL_ENTITIES: entities,
        }
    )
    detection_workflow = Mock(spec=EntityDetectionWorkflow)
    detection_workflow.run.return_value = EntityDetectionResult(
        dataframe=det_df,
        failed_records=detection_failures or [],
    )
    _replace_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme", "Bob likes cats"],
            COL_REPLACED_TEXT: ["[REDACTED] works at [REDACTED]", "[REDACTED] likes cats"],
        }
    )
    replace_runner = Mock(spec=ReplacementWorkflow)
    replace_runner.run.return_value = ReplacementResult(dataframe=_replace_df, failed_records=replace_failures or [])
    _replace_eval_df = _replace_df.copy()
    _replace_eval_df[COL_ENTITY_COVERAGE] = [1.0, 1.0]
    replace_runner.evaluate.return_value = ReplacementResult(dataframe=_replace_eval_df, failed_records=[])
    _rewrite_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme", "Bob likes cats"],
            COL_REWRITTEN_TEXT: ["Beth works at Globex", "Rob likes cats"],
            "utility_score": [0.85, 0.90],
            "leakage_mass": [0.3, 0.1],
            "any_high_leaked": [False, False],
            "needs_human_review": [False, False],
        }
    )
    rewrite_runner = Mock(spec=RewriteWorkflow)
    rewrite_runner.run.return_value = RewriteResult(dataframe=_rewrite_df, failed_records=[])
    _rewrite_eval_df = _rewrite_df.copy()
    _rewrite_eval_df[COL_ENTITIES_BY_VALUE] = [
        {"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]},
        {"entities_by_value": [{"value": "Bob", "labels": ["first_name"]}]},
    ]
    _rewrite_eval_df[COL_JUDGE_EVALUATION] = [{"privacy": {"score": "high"}}] * 2
    _rewrite_eval_df[COL_ENTITY_COVERAGE] = [1.0, 1.0]
    rewrite_runner.evaluate.return_value = RewriteResult(dataframe=_rewrite_eval_df, failed_records=[])
    return Anonymizer(
        detection_workflow=detection_workflow,
        replace_runner=replace_runner,
        rewrite_runner=rewrite_runner,
    )


def test_run_logs_pipeline_stages(stub_input: AnonymizerInput, caplog: pytest.LogCaptureFixture) -> None:
    config = AnonymizerConfig(replace=Redact())

    with caplog.at_level(logging.INFO, logger="anonymizer"):
        anonymizer = _make_logging_anonymizer()
        anonymizer.run(config=config, data=stub_input)

    messages = caplog.text
    assert "📂 Loaded 2 records from" in messages
    assert "🔎 detector:" in messages
    assert "✅ validator:" in messages
    assert "🧩 augmenter:" in messages
    assert "🔍 Running entity detection on 2 records" in messages
    assert "📋 Detection complete" in messages
    assert "3 entities" in messages
    assert re.search(r"\[\d+\.\ds\]", messages), "Detection timing not found"
    assert "labels:" in messages
    assert "first_name=2" in messages
    assert "organization=1" in messages
    assert "🔄 Running Redact replacement" in messages
    assert "📋 Replacement complete" in messages
    assert re.search(r"Replacement complete.*\[\d+\.\ds\]", messages), "Replacement timing not found"
    assert "🎉 Pipeline complete" in messages
    assert "2 records processed" in messages


def test_run_logs_failure_counts(stub_input: AnonymizerInput, caplog: pytest.LogCaptureFixture) -> None:
    anonymizer = _make_logging_anonymizer(
        detection_failures=[FailedRecord(record_id="r1", step="detection", reason="timeout")],
        replace_failures=[FailedRecord(record_id="r2", step="replace", reason="parse error")],
    )
    config = AnonymizerConfig(replace=Redact())

    with caplog.at_level(logging.INFO, logger="anonymizer"):
        anonymizer.run(config=config, data=stub_input)

    messages = caplog.text
    assert "📋 Detection complete" in messages
    assert "1 failed" in messages
    assert "🎉 Pipeline complete" in messages
    assert "2 total failures" in messages


def test_run_without_replacement_skips_replace_logs(
    stub_input: AnonymizerInput, caplog: pytest.LogCaptureFixture
) -> None:
    anonymizer = _make_logging_anonymizer()
    config = AnonymizerConfig(rewrite=Rewrite())

    with caplog.at_level(logging.INFO, logger="anonymizer"):
        anonymizer.run(config=config, data=stub_input)

    messages = caplog.text
    assert "🔍 Running entity detection" in messages
    assert "✏️ Running rewrite pipeline" in messages
    assert "🔄 Running Redact replacement" not in messages
    assert "📋 Replacement complete" not in messages
    assert "🎉 Pipeline complete" in messages


def test_preview_logs_preview_mode(stub_input: AnonymizerInput, caplog: pytest.LogCaptureFixture) -> None:
    anonymizer = _make_logging_anonymizer()
    config = AnonymizerConfig(replace=Redact())

    with caplog.at_level(logging.INFO, logger="anonymizer"):
        anonymizer.preview(config=config, data=stub_input, num_records=5)

    messages = caplog.text
    assert "👀 Preview mode: 📂 Loaded 2 records" in messages


def test_evaluate_replace_logs_stages(stub_input: AnonymizerInput, caplog: pytest.LogCaptureFixture) -> None:
    anonymizer = _make_logging_anonymizer()
    run_result = anonymizer.run(config=AnonymizerConfig(replace=Redact()), data=stub_input)
    caplog.clear()

    with caplog.at_level(logging.INFO, logger="anonymizer"):
        anonymizer.evaluate(run_result)

    messages = caplog.text
    assert "🧪 Running Redact evaluation on 2 records" in messages
    assert "Running replace judges" in messages
    assert "📋 Replace judges complete" in messages
    assert re.search(r"Replace judges complete.*\[\d+\.\ds\]", messages), "Replace evaluation timing not found"
    assert "🎉 Evaluation complete — 2 records processed" in messages
    assert re.search(r"Evaluation complete.*\[\d+\.\ds\]", messages), "Total evaluation timing not found"


def test_evaluate_rewrite_logs_stages(stub_input: AnonymizerInput, caplog: pytest.LogCaptureFixture) -> None:
    anonymizer = _make_logging_anonymizer()
    run_result = anonymizer.run(config=AnonymizerConfig(rewrite=Rewrite()), data=stub_input)
    caplog.clear()

    with (
        patch("anonymizer.interface.anonymizer.EntityCoverageWorkflow") as coverage_workflow,
        caplog.at_level(logging.INFO, logger="anonymizer"),
    ):
        coverage_workflow.return_value.run_non_critical.return_value = (
            anonymizer._rewrite_runner.evaluate.return_value.dataframe,
            [],
        )
        anonymizer.evaluate(run_result)

    messages = caplog.text
    assert "🧪 Running rewrite evaluation on 2 records" in messages
    assert "Running rewrite judges" in messages
    assert "📋 Rewrite judges complete" in messages
    assert re.search(r"Rewrite judges complete.*\[\d+\.\ds\]", messages), "Rewrite evaluation timing not found"
    assert "Running entity coverage" in messages
    assert "📋 Entity coverage complete" in messages
    assert re.search(r"Entity coverage complete.*\[\d+\.\ds\]", messages), "Coverage timing not found"
    assert "🎉 Evaluation complete — 2 records processed" in messages


def test_evaluate_rewrite_logs_unavailable_entity_coverage(
    stub_input: AnonymizerInput, caplog: pytest.LogCaptureFixture
) -> None:
    anonymizer = _make_logging_anonymizer()
    run_result = anonymizer.run(config=AnonymizerConfig(rewrite=Rewrite()), data=stub_input)
    coverage_df = anonymizer._rewrite_runner.evaluate.return_value.dataframe.copy()
    coverage_df[COL_ENTITY_COVERAGE] = None
    caplog.clear()

    with (
        patch("anonymizer.interface.anonymizer.EntityCoverageWorkflow") as coverage_workflow,
        caplog.at_level(logging.INFO, logger="anonymizer"),
    ):
        coverage_workflow.return_value.run_non_critical.return_value = (coverage_df, [])
        anonymizer.evaluate(run_result)

    messages = caplog.text
    assert "Entity coverage score unavailable for 2/2 records" in messages
    assert "📋 Entity coverage complete" in messages
    assert "🎉 Evaluation complete — 2 records processed" in messages


def test_evaluate_debug_logs_config_and_failures_without_sensitive_context(
    stub_input: AnonymizerInput, caplog: pytest.LogCaptureFixture
) -> None:
    anonymizer = _make_logging_anonymizer()
    run_result = anonymizer.run(config=AnonymizerConfig(replace=Redact()), data=stub_input)
    run_result.data_summary = "confidential dataset description"
    anonymizer._replace_runner.evaluate.return_value = ReplacementResult(
        dataframe=anonymizer._replace_runner.evaluate.return_value.dataframe,
        failed_records=[FailedRecord(record_id="r1", step="entity-coverage-judge", reason="timeout")],
    )
    caplog.clear()

    with caplog.at_level(logging.DEBUG, logger="anonymizer"):
        anonymizer.evaluate(run_result)

    messages = caplog.text
    assert "evaluation config: mode=Redact" in messages
    assert "data_summary_provided=True" in messages
    assert "active evaluation judges: entity_coverage_judge" in messages
    assert "evaluation models:" in messages
    assert "1 evaluation failed record(s)" in messages
    assert "r1 (entity-coverage-judge: timeout)" in messages
    assert "confidential dataset description" not in messages


def test_evaluate_logs_unavailable_scores_for_all_active_replace_judges(
    stub_input: AnonymizerInput, caplog: pytest.LogCaptureFixture
) -> None:
    anonymizer = _make_logging_anonymizer()
    run_result = anonymizer.run(config=AnonymizerConfig(replace=Substitute()), data=stub_input)
    unavailable_df = anonymizer._replace_runner.evaluate.return_value.dataframe.copy()
    for column in (
        COL_ENTITY_COVERAGE,
        COL_DETECTION_VALID,
        COL_TYPE_FIDELITY_VALID,
        COL_RELATIONAL_CONSISTENCY_VALID,
        COL_ATTRIBUTE_FIDELITY_VALID,
    ):
        unavailable_df[column] = None
    anonymizer._replace_runner.evaluate.return_value = ReplacementResult(
        dataframe=unavailable_df,
        failed_records=[],
    )
    caplog.clear()

    with caplog.at_level(logging.INFO, logger="anonymizer"):
        anonymizer.evaluate(run_result, config=EvaluateConfig(compute_detection_validity=True))

    messages = caplog.text
    assert (
        "Replace evaluation scores unavailable — entity coverage: 2/2; detection validity: 2/2; "
        "type fidelity: 2/2; relational consistency: 2/2; attribute fidelity: 2/2."
    ) in messages
    assert "📋 Replace judges complete" in messages
    assert "🎉 Evaluation complete — 2 records processed" in messages


def test_evaluate_rewrite_warns_when_rewrite_judge_is_unavailable_but_coverage_succeeds(
    stub_input: AnonymizerInput, caplog: pytest.LogCaptureFixture
) -> None:
    anonymizer = _make_logging_anonymizer()
    run_result = anonymizer.run(config=AnonymizerConfig(rewrite=Rewrite()), data=stub_input)
    evaluation_df = anonymizer._rewrite_runner.evaluate.return_value.dataframe.copy()
    evaluation_df[COL_JUDGE_EVALUATION] = [{"privacy": {"score": "high"}}, None]
    anonymizer._rewrite_runner.evaluate.return_value = RewriteResult(
        dataframe=evaluation_df,
        failed_records=[FailedRecord(record_id="r2", step="rewrite-final-judge", reason="timeout")],
    )
    caplog.clear()

    with (
        patch("anonymizer.interface.anonymizer.EntityCoverageWorkflow") as coverage_workflow,
        caplog.at_level(logging.INFO, logger="anonymizer"),
    ):
        coverage_workflow.return_value.run_non_critical.return_value = (evaluation_df, [])
        anonymizer.evaluate(run_result)

    messages = caplog.text
    assert "Rewrite judge score unavailable for 1/2 records" in messages
    assert "Entity coverage score unavailable" not in messages


def test_evaluate_rewrite_does_not_warn_for_expected_no_entity_passthrough(
    stub_input: AnonymizerInput, caplog: pytest.LogCaptureFixture
) -> None:
    anonymizer = _make_logging_anonymizer()
    run_result = anonymizer.run(config=AnonymizerConfig(rewrite=Rewrite()), data=stub_input)
    evaluation_df = anonymizer._rewrite_runner.evaluate.return_value.dataframe.copy()
    evaluation_df[COL_ENTITIES_BY_VALUE] = [
        {"entities_by_value": []},
        {"entities_by_value": [{"value": "Bob", "labels": ["first_name"]}]},
    ]
    evaluation_df[COL_JUDGE_EVALUATION] = [None, {"privacy": {"score": "high"}}]
    anonymizer._rewrite_runner.evaluate.return_value = RewriteResult(dataframe=evaluation_df, failed_records=[])
    caplog.clear()

    with (
        patch("anonymizer.interface.anonymizer.EntityCoverageWorkflow") as coverage_workflow,
        caplog.at_level(logging.INFO, logger="anonymizer"),
    ):
        coverage_workflow.return_value.run_non_critical.return_value = (evaluation_df, [])
        anonymizer.evaluate(run_result)

    assert "Rewrite judge score unavailable" not in caplog.text


def test_evaluate_substitute_warns_only_for_the_unavailable_judge(
    stub_input: AnonymizerInput, caplog: pytest.LogCaptureFixture
) -> None:
    anonymizer = _make_logging_anonymizer()
    run_result = anonymizer.run(config=AnonymizerConfig(replace=Substitute()), data=stub_input)
    evaluation_df = anonymizer._replace_runner.evaluate.return_value.dataframe.copy()
    evaluation_df[COL_TYPE_FIDELITY_VALID] = [True, None]
    evaluation_df[COL_RELATIONAL_CONSISTENCY_VALID] = [True, True]
    evaluation_df[COL_ATTRIBUTE_FIDELITY_VALID] = [True, True]
    anonymizer._replace_runner.evaluate.return_value = ReplacementResult(dataframe=evaluation_df, failed_records=[])
    caplog.clear()

    with caplog.at_level(logging.INFO, logger="anonymizer"):
        anonymizer.evaluate(run_result)

    messages = caplog.text
    assert "Type fidelity score unavailable for 1/2 records" in messages
    assert "Replace evaluation scores unavailable" not in messages
    assert "Entity coverage score unavailable" not in messages


def test_evaluate_caught_replace_workflow_exception_logs_debug_and_public_warning(
    stub_input: AnonymizerInput, caplog: pytest.LogCaptureFixture
) -> None:
    setup_anonymizer = _make_logging_anonymizer()
    run_result = setup_anonymizer.run(config=AnonymizerConfig(replace=Redact()), data=stub_input)
    run_result.trace_dataframe[COL_ENTITIES_BY_VALUE] = [
        {"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]},
        {"entities_by_value": [{"value": "Bob", "labels": ["first_name"]}]},
    ]

    adapter = Mock(spec=NddAdapter)
    adapter._attach_record_ids.side_effect = lambda dataframe: dataframe.assign(
        **{RECORD_ID_COLUMN: [f"r{i}" for i in range(len(dataframe))]}
    )
    adapter.run_workflow.side_effect = RuntimeError("simulated workflow outage")
    anonymizer = Anonymizer(
        data_designer=Mock(),
        replace_runner=ReplacementWorkflow(adapter=adapter),
    )
    caplog.clear()

    with (
        patch.object(EntityCoverageWorkflow, "column_config", return_value=Mock()),
        caplog.at_level(logging.DEBUG, logger="anonymizer"),
    ):
        evaluated = anonymizer.evaluate(run_result)

    messages = caplog.text
    assert "Replace judges workflow failed; evaluation scores may be unavailable." in messages
    assert "Entity coverage score unavailable for 2/2 records" in messages
    assert evaluated.trace_dataframe[COL_ENTITY_COVERAGE].isna().all()


def test_preview_set_preview_num_records_capped(stub_input: AnonymizerInput) -> None:
    """preview() must propagate preview_num_records so downstream workflows use DataDesigner.preview()."""
    anonymizer = _make_logging_anonymizer()
    config = AnonymizerConfig(replace=Redact())

    # stub_input has 2 rows; num_records=5 is clamped to min(5, 2) = 2
    anonymizer.preview(config=config, data=stub_input, num_records=5)

    det_call_kwargs = anonymizer._detection_workflow.run.call_args.kwargs
    assert det_call_kwargs["preview_num_records"] == 2

    rep_call_kwargs = anonymizer._replace_runner.run.call_args.kwargs
    assert rep_call_kwargs["preview_num_records"] == 2


def test_preview_set_preview_num_records_not_capped(stub_input: AnonymizerInput) -> None:
    """When num_records < available rows, preview_num_records is forwarded as-is."""
    anonymizer = _make_logging_anonymizer()
    config = AnonymizerConfig(replace=Redact())

    # stub_input has 2 rows; num_records=1 fits, so no clamping
    anonymizer.preview(config=config, data=stub_input, num_records=1)

    det_call_kwargs = anonymizer._detection_workflow.run.call_args.kwargs
    assert det_call_kwargs["preview_num_records"] == 1

    rep_call_kwargs = anonymizer._replace_runner.run.call_args.kwargs
    assert rep_call_kwargs["preview_num_records"] == 1


def test_run_set_preview_num_records(stub_input: AnonymizerInput) -> None:
    """run() must pass preview_num_records=None so workflows use full execution."""
    anonymizer = _make_logging_anonymizer()
    config = AnonymizerConfig(replace=Redact())

    anonymizer.run(config=config, data=stub_input)

    det_call_kwargs = anonymizer._detection_workflow.run.call_args.kwargs
    assert det_call_kwargs["preview_num_records"] is None

    rep_call_kwargs = anonymizer._replace_runner.run.call_args.kwargs
    assert rep_call_kwargs["preview_num_records"] is None


def test_configure_logging_with_config_object() -> None:
    from anonymizer.logging import LoggingConfig, configure_logging

    configure_logging(LoggingConfig.verbose())
    assert logging.getLogger("data_designer").level == logging.INFO


def _create_csv(num_records: int, tmp_path: Path) -> str:
    path = tmp_path / "input.csv"
    pd.DataFrame({"text": [f"Name{i} works here" for i in range(num_records)]}).to_csv(path, index=False)
    return str(path)


def test_preview_with_large_input_only_loads_preview_rows(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """preview() should truncate during input loading, not after."""
    num_total = 100
    num_preview = 5
    entities = [
        [{"value": f"Name{i}", "label": "first_name", "start_position": 0, "end_position": 5}]
        for i in range(num_preview)
    ]
    det_df = pd.DataFrame(
        {
            COL_TEXT: [f"Name{i} works here" for i in range(num_preview)],
            COL_DETECTED_ENTITIES: entities,
            COL_FINAL_ENTITIES: entities,
        }
    )
    detection_workflow = Mock(spec=EntityDetectionWorkflow)
    detection_workflow.run.return_value = EntityDetectionResult(dataframe=det_df, failed_records=[])
    _replace_df = pd.DataFrame(
        {
            COL_TEXT: [f"Name{i} works here" for i in range(num_preview)],
            COL_REPLACED_TEXT: ["[REDACTED] works here" for _ in range(num_preview)],
        }
    )
    replace_runner = Mock(spec=ReplacementWorkflow)
    replace_runner.run.return_value = ReplacementResult(dataframe=_replace_df, failed_records=[])
    anonymizer = Anonymizer(detection_workflow=detection_workflow, replace_runner=replace_runner)
    config = AnonymizerConfig(replace=Redact())

    csv_path = _create_csv(num_total, tmp_path)
    input_data = AnonymizerInput(source=csv_path)

    with caplog.at_level(logging.INFO, logger="anonymizer"):
        anonymizer.preview(config=config, data=input_data, num_records=num_preview)

    messages = caplog.text
    assert "Preview mode:" in messages
    assert f"Loaded {num_preview} records" in messages
    assert f"Loaded {num_total} records" not in messages
    assert f"Running entity detection on {num_preview} records" in messages


def test_local_replace_logs_progress_for_large_datasets(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """When record count >= 50, ProgressTracker emits interval logs."""
    num_records = 100
    entities = [
        [{"value": f"Name{i}", "label": "first_name", "start_position": 0, "end_position": 5}]
        for i in range(num_records)
    ]
    det_df = pd.DataFrame(
        {
            COL_TEXT: [f"Name{i} works here" for i in range(num_records)],
            COL_DETECTED_ENTITIES: entities,
            COL_FINAL_ENTITIES: entities,
        }
    )
    detection_workflow = Mock(spec=EntityDetectionWorkflow)
    detection_workflow.run.return_value = EntityDetectionResult(dataframe=det_df, failed_records=[])
    replace_runner = ReplacementWorkflow()
    anonymizer = Anonymizer(detection_workflow=detection_workflow, replace_runner=replace_runner)
    config = AnonymizerConfig(replace=Redact())

    csv_path = _create_csv(num_records, tmp_path)
    input_data = AnonymizerInput(source=csv_path)

    with caplog.at_level(logging.INFO, logger="anonymizer"):
        anonymizer.run(config=config, data=input_data)

    assert "Replacement progress:" in caplog.text
    assert "rec/s" in caplog.text
