# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import io
import logging
import warnings
from collections.abc import Coroutine, Iterator
from datetime import datetime, timezone
from types import CoroutineType
from typing import Any
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest
from data_designer.config.column_configs import LLMTextColumnConfig
from data_designer.config.models import ModelConfig
from data_designer.interface.data_designer import DataDesigner
from pydantic import ValidationError

import anonymizer.logging as anonymizer_logging
from anonymizer.config.anonymizer_config import Detect
from anonymizer.config.models import EvaluateModelSelection
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import COL_ENTITIES_BY_VALUE, COL_TEXT
from anonymizer.engine.detection.chunked_validation import _dispatch_chunk, _dispatch_chunk_async
from anonymizer.engine.evaluation.detection_judge import DetectionJudgeWorkflow
from anonymizer.engine.evaluation.entity_coverage_judge import EntityCoverageWorkflow
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.replace.llm_replace_workflow import _filter_replacement_map_to_input_entities
from anonymizer.engine.rewrite.evaluate import _normalize_answer_items
from anonymizer.engine.schemas import EntitiesByValueSchema, RawValidationDecisionsSchema
from anonymizer.engine.schemas.rewrite import EntityDispositionSchema
from anonymizer.interface.errors import AnonymizerWorkflowError
from anonymizer.logging import LoggingConfig, configure_logging
from anonymizer.telemetry import AnonymizerEvent, QueuedEvent, TaskEnum, TaskStatusEnum, TelemetryHandler

CANARY = "SENSITIVE_CONTENT_CANARY_e873"


class UnparseableEntities(dict):
    def get(self, key: object, default: object = None) -> object:
        raise RuntimeError(CANARY)


@pytest.fixture(params=["default", "debug"])
def captured_logs(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[io.StringIO]:
    stream = io.StringIO()
    loggers = [logging.getLogger(name) for name in ("anonymizer", "data_designer")]
    previous = [(logger.handlers[:], logger.level, logger.propagate) for logger in loggers]
    for field in ("_active_config", "_configured", "_anonymizer_handler"):
        monkeypatch.setattr(anonymizer_logging, field, getattr(anonymizer_logging, field))
    monkeypatch.setattr("sys.stderr", stream)
    configure_logging(getattr(LoggingConfig, request.param)())
    try:
        yield stream
    finally:
        for logger, (handlers, level, propagate) in zip(loggers, previous, strict=True):
            logger.handlers = handlers
            logger.setLevel(level)
            logger.propagate = propagate


def test_adapter_failure_keeps_public_cause_without_logging_content(captured_logs: io.StringIO) -> None:
    designer = Mock(spec=DataDesigner)
    designer._artifact_path = CANARY
    failure = RuntimeError(CANARY)
    designer.create.side_effect = failure
    adapter = NddAdapter(data_designer=designer)
    with pytest.raises(AnonymizerWorkflowError, match=CANARY) as caught:
        adapter.run_workflow(
            pd.DataFrame({CANARY: [CANARY]}),
            model_configs=[ModelConfig(alias=CANARY, model=CANARY, provider="stub")],
            columns=[LLMTextColumnConfig(name=CANARY + "_column", model_alias=CANARY, prompt=CANARY)],
            workflow_name=CANARY,
        )
    assert caught.value.__cause__ is failure
    assert str(caught.value) == f"Workflow failed: {CANARY}"
    assert "Workflow failed for 1 input record(s)" in captured_logs.getvalue()
    assert CANARY.lower() not in captured_logs.getvalue().lower()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("exhausted", [False, True])
def test_validator_failover_preserves_order_and_terminal_exception(
    captured_logs: io.StringIO, asynchronous: bool, exhausted: bool
) -> None:
    calls: list[int] = []
    first_failure = RuntimeError(CANARY + "first")
    last_failure = RuntimeError(CANARY + "last")
    output = RawValidationDecisionsSchema(decisions=[])

    def first(**kwargs: object) -> None:
        calls.append(1)
        raise first_failure

    def last(**kwargs: object) -> tuple[RawValidationDecisionsSchema, list[object]]:
        calls.append(2)
        if exhausted:
            raise last_failure
        return output, []

    facades = [(CANARY + "alias1", Mock()), (CANARY + "alias2", Mock())]
    for (_, facade), effect in zip(facades, (first, last), strict=True):
        facade.generate.side_effect = effect
        facade.agenerate = AsyncMock(side_effect=effect)

    def invoke() -> object:
        if asynchronous:
            return asyncio.run(
                _dispatch_chunk_async(facades=facades, prompt=CANARY, system_prompt=CANARY, chunk_index=0)
            )
        return _dispatch_chunk(facades=facades, prompt=CANARY, system_prompt=CANARY, chunk_index=0)

    if exhausted:
        with pytest.raises(RuntimeError) as caught:
            invoke()
        assert caught.value is last_failure
        assert "pool exhausted" in captured_logs.getvalue()
    else:
        assert invoke() is output
    assert calls == [1, 2]
    assert "failing over" in captured_logs.getvalue()
    assert CANARY.lower() not in captured_logs.getvalue().lower()


def test_judge_parse_failure_keeps_empty_fallback_without_traceback(captured_logs: io.StringIO) -> None:
    workflow = DetectionJudgeWorkflow(adapter=Mock())
    frame = pd.DataFrame({COL_TEXT: [CANARY], COL_ENTITIES_BY_VALUE: [UnparseableEntities()]})
    prepared = workflow.prepare(frame)
    assert workflow._passthrough_mask(prepared).tolist() == [True]
    assert prepared[COL_TEXT].tolist() == [CANARY]
    assert "Could not parse" in captured_logs.getvalue()
    assert CANARY.lower() not in captured_logs.getvalue().lower()
    assert "Traceback" not in captured_logs.getvalue()


def test_coverage_swallowed_failure_preserves_fallback(captured_logs: io.StringIO) -> None:
    adapter = Mock()
    adapter._attach_record_ids.side_effect = RuntimeError(CANARY)
    workflow = EntityCoverageWorkflow(adapter=adapter)
    frame = pd.DataFrame({COL_TEXT: [CANARY]})
    result, failures = workflow.run_non_critical(
        frame,
        model_configs=[],
        selected_models=EvaluateModelSelection(**dict.fromkeys(EvaluateModelSelection.model_fields, CANARY)),
    )
    assert result[COL_TEXT].tolist() == [CANARY]
    assert result[workflow.VALID_COL].tolist() == [None]
    assert result[workflow.INVALID_COL].tolist() == [[]]
    assert result[workflow.N_CANDIDATES_COL].tolist() == [None]
    assert failures == []
    assert "Entity coverage workflow failed" in captured_logs.getvalue()
    assert CANARY.lower() not in captured_logs.getvalue().lower()
    assert "Traceback" not in captured_logs.getvalue()


def test_replacement_collision_and_evaluator_normalization_keep_public_payloads(captured_logs: io.StringIO) -> None:
    parsed = EntitiesByValueSchema.model_validate({"entities_by_value": [{"value": CANARY, "labels": [CANARY]}]})
    result = _filter_replacement_map_to_input_entities(
        raw_map={"replacements": [{"original": CANARY, "label": CANARY, "synthetic": CANARY}]},
        parsed_entities=parsed,
        record_id=CANARY,
    )
    assert result["replacements"][0]["original"] == CANARY
    assert result["replacements"][0]["label"] == CANARY
    assert result["replacements"][0]["synthetic"] != CANARY
    normalized = _normalize_answer_items(
        [{"id": 987654321, "answer": CANARY}, {"id": 1, "answer": CANARY}, {"id": 1}],
        expected_ids=[1, 2],
        label=CANARY,
        default_item_factory=lambda item_id: {"id": item_id, "answer": None},
    )
    assert normalized == [{"id": 1, "answer": CANARY}, {"id": 2, "answer": None}]
    assert "repaired=1" in captured_logs.getvalue()
    assert "missing=1 duplicate=1 extra=1" in captured_logs.getvalue()
    assert CANARY.lower() not in captured_logs.getvalue().lower()
    assert "987654321" not in captured_logs.getvalue()


def test_rewrite_swallowed_failure_and_parse_fallback(captured_logs: io.StringIO) -> None:
    from anonymizer.engine.constants import COL_DETECTION_INVALID_ENTITIES, COL_DETECTION_VALID, COL_JUDGE_EVALUATION
    from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow, _detection_valid_fraction

    frame = pd.DataFrame({COL_TEXT: [CANARY]})
    workflow = RewriteWorkflow(adapter=Mock())
    workflow._judge_wf = Mock()
    workflow._judge_wf.columns.side_effect = RuntimeError(CANARY)
    result, failures = workflow._run_final_judge(
        frame,
        model_configs=[],
        selected_models=EvaluateModelSelection(**dict.fromkeys(EvaluateModelSelection.model_fields, CANARY)),
        privacy_goal=PrivacyGoal(protect="Protect all personal identities", preserve="Preserve the useful meaning"),
        preview_num_records=None,
    )
    assert result[COL_TEXT].tolist() == [CANARY]
    assert result[COL_JUDGE_EVALUATION].tolist() == [None]
    assert failures == []
    row = pd.Series(
        {
            COL_DETECTION_VALID: False,
            COL_DETECTION_INVALID_ENTITIES: [{"value": CANARY}],
            COL_ENTITIES_BY_VALUE: UnparseableEntities(),
        }
    )
    assert _detection_valid_fraction(row) is None
    assert "Could not parse" in captured_logs.getvalue()
    assert CANARY.lower() not in captured_logs.getvalue().lower()
    assert "Traceback" not in captured_logs.getvalue()


@pytest.mark.parametrize("preview", [None, 1])
def test_reader_keeps_source_and_renamed_schema_out_of_logs(
    captured_logs: io.StringIO, tmp_path: object, preview: int | None
) -> None:
    from pathlib import Path

    from anonymizer.config.anonymizer_config import AnonymizerInput
    from anonymizer.engine.io.reader import read_input

    source = Path(str(tmp_path)) / (CANARY + ".csv")
    pd.DataFrame({CANARY: [CANARY], CANARY + "_replaced": [CANARY]}).to_csv(source, index=False)
    result = read_input(AnonymizerInput(source=str(source), text_column=CANARY), nrows=preview)
    assert result.requested_text_column == CANARY
    assert result.dataframe[COL_TEXT].tolist() == [CANARY]
    assert result.dataframe[CANARY + "_replaced__input"].tolist() == [CANARY]
    assert "Renamed 1 input column(s)" in captured_logs.getvalue()
    assert CANARY.lower() not in captured_logs.getvalue().lower()


def test_replace_judges_keep_prepared_rows_on_swallowed_failure(captured_logs: io.StringIO) -> None:
    from anonymizer.engine.replace.replace_runner import ReplacementWorkflow

    adapter = Mock()
    adapter._attach_record_ids.side_effect = lambda frame: frame
    adapter.run_workflow.side_effect = RuntimeError(CANARY)
    judge = Mock()
    judge.prepare.side_effect = lambda frame: frame
    judge.postprocess.side_effect = lambda frame: frame.assign(safe_score=None)
    workflow = ReplacementWorkflow(adapter=adapter)
    frame = pd.DataFrame({COL_TEXT: [CANARY]})
    failures = []
    result = workflow._run_merged_judges(
        frame,
        is_substitute=False,
        entity_coverage_judge=judge,
        compute_detection_validity=False,
        model_configs=[],
        selected_models=EvaluateModelSelection(**dict.fromkeys(EvaluateModelSelection.model_fields, CANARY)),
        preview_num_records=None,
        failed_records=failures,
    )
    assert result[COL_TEXT].tolist() == [CANARY]
    assert result["safe_score"].tolist() == [None]
    assert failures == []
    adapter.run_workflow.assert_called_once()
    judge.postprocess.assert_called_once()
    assert CANARY.lower() not in captured_logs.getvalue().lower()
    assert "Traceback" not in captured_logs.getvalue()


def test_unexpected_replacement_type_keeps_empty_result_without_type_label(captured_logs: io.StringIO) -> None:
    raw = type(CANARY, (), {})()
    result = _filter_replacement_map_to_input_entities(
        raw_map=raw, parsed_entities=EntitiesByValueSchema(), record_id=CANARY
    )
    assert result == {"replacements": []}
    assert "unexpected type" in captured_logs.getvalue()
    assert CANARY.lower() not in captured_logs.getvalue().lower()


def test_detection_spec_retains_aliases_and_labels_without_logging_them(captured_logs: io.StringIO) -> None:
    from anonymizer.config.models import DetectionModelSelection
    from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
    from anonymizer.engine.workflow_columns.detection.config import ChunkedValidationConfig

    aliases = [CANARY + suffix for suffix in ("detector", "validator1", "validator2", "augmenter")]
    selection = DetectionModelSelection(
        entity_detector=aliases[0],
        entity_validator=aliases[1:3],
        entity_augmenter=aliases[3],
        latent_detector=aliases[3],
    )
    models, columns = EntityDetectionWorkflow(adapter=Mock())._build_detection_spec(
        model_configs=[ModelConfig(alias=alias, model=CANARY, provider="stub") for alias in aliases],
        selected_models=selection,
        gliner_detection_threshold=0.3,
        entity_labels=[CANARY],
    )
    assert [model.alias for model in models] == aliases
    validators = [column for column in columns if isinstance(column, ChunkedValidationConfig)]
    assert len(validators) == 1
    assert validators[0].pool == aliases[1:3]
    assert models[0].inference_parameters.extra_body is not None
    assert models[0].inference_parameters.extra_body["labels"] == [CANARY.lower()]
    assert "pool of 2 aliases" in captured_logs.getvalue()
    assert CANARY.lower() not in captured_logs.getvalue().lower()


def test_local_replacement_keeps_label_and_text_in_result_only(captured_logs: io.StringIO) -> None:
    from anonymizer.config.replace_strategies import Redact
    from anonymizer.engine.constants import COL_FINAL_ENTITIES, COL_REPLACED_TEXT, COL_REPLACEMENT_MAP
    from anonymizer.engine.replace.strategies import apply_local_replace_strategy

    frame = pd.DataFrame(
        {
            COL_TEXT: [CANARY],
            COL_FINAL_ENTITIES: [
                {"entities": [{"value": CANARY, "label": CANARY, "start_position": 0, "end_position": len(CANARY)}]}
            ],
        }
    )
    result = apply_local_replace_strategy(frame, strategy=Redact(normalize_label=False))
    assert result[COL_REPLACED_TEXT].tolist() == [f"[REDACTED_{CANARY}]"]
    assert result[COL_REPLACEMENT_MAP].iloc[0]["replacements"][0]["label"] == CANARY
    assert frame[COL_TEXT].tolist() == [CANARY]
    assert CANARY.lower() not in captured_logs.getvalue().lower()


def test_rewrite_detection_judge_failure_keeps_unavailable_scores(captured_logs: io.StringIO) -> None:
    from anonymizer.engine.constants import COL_DETECTION_VALID, COL_JUDGE_EVALUATION
    from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow

    workflow = RewriteWorkflow(adapter=Mock())
    workflow._detection_judge_wf = Mock()
    workflow._detection_judge_wf.evaluate.side_effect = RuntimeError(CANARY)
    workflow._judge_wf = Mock()
    workflow._judge_wf.columns.side_effect = RuntimeError(CANARY)
    frame = pd.DataFrame(
        {COL_TEXT: [CANARY], COL_ENTITIES_BY_VALUE: [{"entities_by_value": [{"value": CANARY, "labels": [CANARY]}]}]}
    )
    result = workflow.evaluate(
        frame,
        model_configs=[],
        selected_models=EvaluateModelSelection(**dict.fromkeys(EvaluateModelSelection.model_fields, CANARY)),
        privacy_goal=PrivacyGoal(protect="Protect all personal identities", preserve="Preserve the useful meaning"),
        compute_detection_validity=True,
    )
    assert result.dataframe[COL_TEXT].tolist() == [CANARY]
    assert result.dataframe[COL_DETECTION_VALID].tolist() == [None]
    assert result.dataframe[COL_JUDGE_EVALUATION].tolist() == [None]
    assert result.failed_records == []
    workflow._detection_judge_wf.evaluate.assert_called_once()
    assert CANARY.lower() not in captured_logs.getvalue().lower()
    assert "Traceback" not in captured_logs.getvalue()


def _assert_content_free_records(
    captured_logs: io.StringIO, records: list[logging.LogRecord], *, forbidden: tuple[str, ...] = (CANARY,)
) -> None:
    for secret in forbidden:
        assert secret.lower() not in captured_logs.getvalue().lower()
        for record in records:
            assert secret.lower() not in repr(record.args).lower()
            assert secret.lower() not in record.getMessage().lower()
    assert all(record.exc_info is None for record in records)
    assert all(record.stack_info is None for record in records)


@pytest.mark.parametrize("scenario", ["duplicate-keys", "duplicate-values", "excluded-examples", "overlap"])
def test_config_warnings_preserve_normalization_without_label_payloads(
    captured_logs: io.StringIO, caplog: pytest.LogCaptureFixture, scenario: str
) -> None:
    label = CANARY.lower()
    first, second, third = (CANARY + suffix for suffix in ("_first", "_second", "_third"))
    if scenario == "duplicate-keys":
        config = Detect(
            entity_labels=[CANARY],
            entity_label_examples={f" {CANARY} ": [f" {first} ", second], label: [third]},
        )
        assert config.entity_label_examples == {label: [first, second, third]}
        assert config.entity_labels == [label]
        assert config.excluded_entity_labels is None
    elif scenario == "duplicate-values":
        config = Detect(entity_labels=[CANARY], entity_label_examples={CANARY: [first, second, f" {first} "]})
        assert config.entity_label_examples == {label: [first, second]}
        assert config.entity_labels == [label]
        assert config.excluded_entity_labels is None
    elif scenario == "excluded-examples":
        config = Detect(entity_label_examples={CANARY: [first]}, excluded_entity_labels=[CANARY])
        assert config.entity_label_examples == {label: [first]}
        assert config.entity_labels is None
        assert config.excluded_entity_labels == [label]
    else:
        config = Detect(entity_labels=[CANARY, "email"], excluded_entity_labels=[CANARY])
        assert config.entity_labels == ["email", label]
        assert config.excluded_entity_labels == [label]
        assert config.entity_label_examples == {}
    records = [record for record in caplog.records if record.name.startswith("anonymizer")]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert records[0].args == (1,)
    _assert_content_free_records(captured_logs, records)


@pytest.mark.parametrize("scenario", ["unknown-example", "empty-effective-set"])
def test_config_errors_keep_public_details_without_logging_labels(
    captured_logs: io.StringIO, caplog: pytest.LogCaptureFixture, scenario: str
) -> None:
    if scenario == "unknown-example":
        expected = (
            "entity_label_examples contains labels outside the active label set: "
            f"{[CANARY.lower()]}. Add every example label to entity_labels "
            "(non-default labels require an explicit entity_labels set) or remove their examples."
        )
        with pytest.raises(ValidationError) as caught:
            Detect(entity_labels=["email"], entity_label_examples={CANARY: [CANARY]})
        assert CANARY.lower() in str(caught.value).lower()
    else:
        expected = "excluded_entity_labels entirely overlaps entity_labels, leaving an empty effective detection set."
        with pytest.raises(ValidationError) as caught:
            Detect(entity_labels=[CANARY], excluded_entity_labels=[CANARY])
    assert str(caught.value.errors()[0]["ctx"]["error"]) == expected
    _assert_content_free_records(captured_logs, caplog.records)


@pytest.mark.parametrize("method", ["replace", "generalize", "remove", "suppress_inference"])
def test_risk_promotion_keeps_all_entity_fields_without_logging_identity(
    captured_logs: io.StringIO, caplog: pytest.LogCaptureFixture, method: str
) -> None:
    payload = {
        "id": 987654321,
        "source": "tagged",
        "category": "direct_identifier",
        "sensitivity": "high",
        "entity_label": CANARY,
        "entity_value": CANARY,
        "protection_reason": "Protect the original sensitive value " + CANARY,
        "protection_method_suggestion": method,
        "combined_risk_level": "low",
    }
    entity = EntityDispositionSchema.model_validate(payload)
    assert entity.model_dump() == {**payload, "combined_risk_level": "medium"}
    assert entity.model_dump(mode="json") == {**payload, "combined_risk_level": "medium"}
    assert entity.needs_protection is True
    records = [record for record in caplog.records if record.name.startswith("anonymizer")]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    _assert_content_free_records(captured_logs, records, forbidden=(CANARY, "987654321"))


def test_high_risk_unprotected_entity_keeps_exact_public_error(
    captured_logs: io.StringIO, caplog: pytest.LogCaptureFixture
) -> None:
    with pytest.raises(ValidationError) as caught:
        EntityDispositionSchema.model_validate(
            {
                "id": 987654321,
                "source": "tagged",
                "category": "direct_identifier",
                "sensitivity": "high",
                "entity_label": CANARY,
                "entity_value": CANARY,
                "protection_reason": "Preserve the original sensitive value " + CANARY,
                "protection_method_suggestion": "leave_as_is",
                "combined_risk_level": "high",
            }
        )
    assert str(caught.value.errors()[0]["ctx"]["error"]) == (
        "Entity 987654321: combined_risk_level='high' cannot have protection_method_suggestion='leave_as_is'"
    )
    assert caught.value.errors()[0]["input"]["entity_label"] == CANARY
    assert caught.value.errors()[0]["input"]["entity_value"] == CANARY
    assert "987654321" in str(caught.value)
    _assert_content_free_records(captured_logs, caplog.records, forbidden=(CANARY, "987654321"))


def test_telemetry_flush_failure_keeps_pending_queues_and_closes_coroutine(
    captured_logs: io.StringIO, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    event = AnonymizerEvent(
        task=TaskEnum.BATCH,
        task_status=TaskStatusEnum.COMPLETED,
        transformation_type="redact",
        entity_detector_model=CANARY,
        entity_validator_model=CANARY,
        entity_augmenter_model=CANARY,
    )
    pending = QueuedEvent(event=event, timestamp=datetime(2026, 10, 15, tzinfo=timezone.utc))
    retry = QueuedEvent(event=event, timestamp=pending.timestamp, retry_count=1)
    handler = TelemetryHandler(source_client_version=CANARY, session_id=CANARY)
    handler._events = [pending]
    handler._dlq = [retry]
    pending_queue, retry_queue = handler._events, handler._dlq
    attempted: list[Coroutine[Any, Any, Any]] = []

    def fail_before_execution(coro: Coroutine[Any, Any, Any]) -> None:
        attempted.append(coro)
        coro.close()
        raise RuntimeError(CANARY)

    monkeypatch.setattr(handler, "_run_sync", fail_before_execution)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        with handler:
            pass
    assert len(attempted) == 1
    assert isinstance(attempted[0], CoroutineType)
    assert attempted[0].cr_frame is None
    assert handler._events is pending_queue and handler._events == [pending]
    assert handler._dlq is retry_queue and handler._dlq == [retry]
    assert pending.retry_count == 0 and retry.retry_count == 1
    assert not emitted
    records = [record for record in caplog.records if record.name.startswith("anonymizer")]
    assert len(records) == int(logging.getLogger("anonymizer").isEnabledFor(logging.DEBUG))
    _assert_content_free_records(captured_logs, records)
