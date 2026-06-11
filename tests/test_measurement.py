# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from data_designer.config.column_configs import CustomColumnConfig, LLMStructuredColumnConfig, LLMTextColumnConfig
from data_designer.config.column_types import ColumnConfigT
from data_designer.config.custom_column import custom_column_generator
from data_designer.config.models import ModelConfig
from data_designer.config.run_config import RunConfig
from data_designer.config.utils.constants import TRACE_COLUMN_POSTFIX
from data_designer.config.utils.trace_type import TraceType
from data_designer.interface.data_designer import DataDesigner

import anonymizer.measurement as measurement
from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, Detect
from anonymizer.config.models import DetectionModelSelection
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_FINAL_ENTITIES,
    COL_LEAKAGE_MASS,
    COL_NEEDS_HUMAN_REVIEW,
    COL_NEEDS_REPAIR,
    COL_REPAIR_ITERATIONS,
    COL_REPLACED_TEXT,
    COL_REPLACEMENT_MAP,
    COL_SEED_VALIDATION_CANDIDATES,
    COL_TEXT,
    COL_UTILITY_SCORE,
    COL_WEIGHTED_LEAKAGE_RATE,
)
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult, EntityDetectionWorkflow
from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN, NddAdapter, WorkflowRunResult
from anonymizer.engine.replace.replace_runner import ReplacementResult, ReplacementWorkflow
from anonymizer.engine.rewrite.rewrite_workflow import RewriteResult, RewriteWorkflow
from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.interface.errors import AnonymizerWorkflowError
from anonymizer.measurement import (
    DEFAULT_MEASUREMENT_ENV_PREFIX,
    MEASUREMENT_SCHEMA_VERSION,
    MeasurementCollector,
    MeasurementConfig,
    configured_measurement_session,
    estimate_llm_calls_by_stage,
    measurement_session,
    record_record_metrics,
    stage_timer,
)


class _FailingSink:
    def __init__(self, message: str) -> None:
        self.message = message

    def write_record(self, record: dict[str, Any]) -> None:
        _ = record
        raise OSError(self.message)

    def close(self) -> None:
        pass


@pytest.fixture
def trace_input_df() -> pd.DataFrame:
    return pd.DataFrame({"text": ["Alice works at Acme"], RECORD_ID_COLUMN: ["record-a"]})


def _trace_model_configs() -> list[ModelConfig]:
    return [ModelConfig(alias="alias", model="dummy-model", provider="provider")]


def _run_entity_detection_preview(
    adapter: NddAdapter,
    input_df: pd.DataFrame,
    columns: list[ColumnConfigT],
) -> WorkflowRunResult:
    return adapter.run_workflow(
        input_df,
        model_configs=_trace_model_configs(),
        columns=columns,
        workflow_name="entity-detection",
        preview_num_records=1,
    )


def _raw_detected_text_column() -> LLMTextColumnConfig:
    return LLMTextColumnConfig(name="raw_detected", prompt="{{ text }}", model_alias="alias")


def test_ndd_adapter_records_workflow_measurement_without_raw_text() -> None:
    input_df = pd.DataFrame(
        {
            "text": ["Alice works at Acme", "Bob works at Beta"],
            RECORD_ID_COLUMN: ["record-a", "record-b"],
        }
    )
    mock_dd = Mock(spec=DataDesigner)
    mock_dd.preview.return_value = SimpleNamespace(dataset=input_df.iloc[[0]].copy())
    adapter = NddAdapter(data_designer=mock_dd)
    collector = MeasurementCollector(record_hash_key="test-key")

    with measurement_session(collector):
        result = adapter.run_workflow(
            input_df,
            model_configs=[ModelConfig(alias="detector", model="dummy")],
            columns=[
                LLMTextColumnConfig(
                    name="raw_detected",
                    prompt="{{ text }}",
                    model_alias="detector",
                )
            ],
            workflow_name="entity-detection",
            preview_num_records=2,
        )

    assert len(result.failed_records) == 1
    records = [record for record in collector.records if record["record_type"] == "ndd_workflow"]
    assert len(records) == 1
    record = records[0]
    assert record["workflow_name"] == "entity-detection"
    assert record["model_aliases"] == ["detector"]
    assert record["input_row_count"] == 2
    assert record["seed_row_count"] == 2
    assert record["output_row_count"] == 1
    assert record["failed_record_count"] == 1
    assert record["elapsed_sec"] >= 0

    serialized = json.dumps(record)
    assert "Alice" not in serialized
    assert "Acme" not in serialized
    assert "Bob" not in serialized


def test_ndd_adapter_records_datadesigner_model_usage() -> None:
    input_df = pd.DataFrame(
        {
            "text": ["Alice works at Acme"],
            RECORD_ID_COLUMN: ["record-a"],
        }
    )

    class UsageStats:
        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return {
                "token_usage": {
                    "input_tokens": 12,
                    "output_tokens": 4,
                    "total_tokens": 16,
                },
                "request_usage": {
                    "successful_requests": 2,
                    "failed_requests": 1,
                    "total_requests": 3,
                },
            }

    class ModelRegistry:
        def get_model_usage_snapshot(self) -> dict[str, UsageStats]:
            return {"dummy-model": UsageStats()}

    class UsageDataDesigner:
        def _create_resource_provider(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(model_registry=ModelRegistry())

        def preview(self, _config_builder: object, *, num_records: int) -> SimpleNamespace:
            self._create_resource_provider("preview-dataset", _config_builder)
            return SimpleNamespace(dataset=input_df.iloc[:num_records].copy())

    adapter = NddAdapter(data_designer=cast(DataDesigner, UsageDataDesigner()))
    collector = MeasurementCollector(record_hash_key="test-key")

    with measurement_session(collector):
        adapter.run_workflow(
            input_df,
            model_configs=[ModelConfig(alias="detector", model="dummy")],
            columns=[
                LLMTextColumnConfig(
                    name="raw_detected",
                    prompt="{{ text }}",
                    model_alias="detector",
                )
            ],
            workflow_name="entity-detection",
            preview_num_records=1,
        )

    record = next(record for record in collector.records if record["record_type"] == "ndd_workflow")
    assert record["model_usage"]["dummy-model"]["token_usage"]["input_tokens"] == 12
    assert record["observed_input_tokens"] == 12
    assert record["observed_output_tokens"] == 4
    assert record["observed_total_tokens"] == 16
    assert record["observed_successful_requests"] == 2
    assert record["observed_failed_requests"] == 1
    assert record["observed_total_requests"] == 3
    assert record["input_rows_per_sec"] >= 0
    assert record["output_rows_per_sec"] >= 0
    assert record["observed_tokens_per_sec"] >= 0
    assert record["observed_requests_per_sec"] >= 0
    assert record["observed_tokens_per_successful_request"] == 8


def test_ndd_adapter_records_datadesigner_model_usage_by_alias_for_shared_model_names() -> None:
    input_df = pd.DataFrame(
        {
            "text": ["Alice works at Acme"],
            RECORD_ID_COLUMN: ["record-a"],
        }
    )

    class UsageStats:
        has_usage = True

        def __init__(self, *, input_tokens: int, output_tokens: int, successful: int, failed: int) -> None:
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens
            self.successful = successful
            self.failed = failed

        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return {
                "token_usage": {
                    "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens,
                    "total_tokens": self.input_tokens + self.output_tokens,
                },
                "request_usage": {
                    "successful_requests": self.successful,
                    "failed_requests": self.failed,
                    "total_requests": self.successful + self.failed,
                },
            }

    class ModelRegistry:
        def __init__(self) -> None:
            self._models = {
                "validator": SimpleNamespace(
                    model_alias="validator",
                    model_name="shared-model",
                    model_provider_name="local-vllm",
                    usage_stats=UsageStats(input_tokens=12, output_tokens=4, successful=2, failed=1),
                ),
                "augmenter": SimpleNamespace(
                    model_alias="augmenter",
                    model_name="shared-model",
                    model_provider_name="local-vllm",
                    usage_stats=UsageStats(input_tokens=20, output_tokens=8, successful=1, failed=0),
                ),
            }

        def get_model_usage_snapshot(self) -> dict[str, UsageStats]:
            return {
                "shared-model": UsageStats(input_tokens=999, output_tokens=999, successful=99, failed=99),
            }

    class UsageDataDesigner:
        def _create_resource_provider(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(model_registry=ModelRegistry())

        def preview(self, _config_builder: object, *, num_records: int) -> SimpleNamespace:
            self._create_resource_provider("preview-dataset", _config_builder)
            return SimpleNamespace(dataset=input_df.iloc[:num_records].copy())

    adapter = NddAdapter(data_designer=cast(DataDesigner, UsageDataDesigner()))
    collector = MeasurementCollector(record_hash_key="test-key")

    with measurement_session(collector):
        adapter.run_workflow(
            input_df,
            model_configs=[ModelConfig(alias="validator", model="shared-model")],
            columns=[
                LLMTextColumnConfig(
                    name="raw_detected",
                    prompt="{{ text }}",
                    model_alias="validator",
                )
            ],
            workflow_name="entity-detection",
            preview_num_records=1,
        )

    record = next(record for record in collector.records if record["record_type"] == "ndd_workflow")
    assert sorted(record["model_usage"]) == ["augmenter", "validator"]
    assert record["model_usage"]["validator"]["model_alias"] == "validator"
    assert record["model_usage"]["validator"]["model_name"] == "shared-model"
    assert record["model_usage"]["validator"]["model_provider_name"] == "local-vllm"
    assert record["model_usage"]["validator"]["token_usage"]["input_tokens"] == 12
    assert record["model_usage"]["augmenter"]["token_usage"]["input_tokens"] == 20
    assert record["observed_input_tokens"] == 32
    assert record["observed_output_tokens"] == 12
    assert record["observed_total_tokens"] == 44
    assert record["observed_successful_requests"] == 3
    assert record["observed_failed_requests"] == 1
    assert record["observed_total_requests"] == 4


def test_records_generic_model_workflow_usage_without_raw_text() -> None:
    collector = MeasurementCollector(record_hash_key="test-key")

    with measurement_session(collector):
        assert hasattr(measurement, "record_model_workflow")
        measurement.record_model_workflow(
            workflow_name="entity-detection-native-rules-router",
            model_aliases=["native-direct"],
            input_row_count=1,
            output_row_count=1,
            failed_record_count=0,
            elapsed_sec=0.25,
            model_usage={
                "native-direct": {
                    "model_alias": "native-direct",
                    "model_name": "nvidia/nemotron-3-super",
                    "model_provider_name": "local-vllm",
                    "request_usage": {
                        "successful_requests": 3,
                        "failed_requests": 0,
                        "total_requests": 3,
                    },
                    "token_usage": {
                        "input_tokens": 30,
                        "output_tokens": 12,
                        "total_tokens": 42,
                    },
                },
            },
        )

    records = [record for record in collector.records if record["record_type"] == "model_workflow"]
    assert len(records) == 1
    record = records[0]
    assert record["workflow_name"] == "entity-detection-native-rules-router"
    assert record["model_aliases"] == ["native-direct"]
    assert record["observed_total_requests"] == 3
    assert record["observed_input_tokens"] == 30
    assert record["observed_output_tokens"] == 12
    assert record["observed_total_tokens"] == 42
    assert record["observed_failed_request_rate"] == 0
    assert record["observed_tokens_per_successful_request"] == 14

    serialized = json.dumps(record)
    assert "Alice" not in serialized
    assert "sk-test" not in serialized


def test_anonymizer_records_per_record_measurement_without_raw_pii(tmp_path: Path) -> None:
    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Alice works at Acme"]}).to_csv(input_csv, index=False)
    final_entities = {
        "entities": [
            {"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5},
            {"value": "Acme", "label": "company_name", "start_position": 15, "end_position": 19},
        ]
    }
    validation_candidates = {
        "candidates": [
            {"value": "Alice", "label": "first_name"},
            {"value": "Acme", "label": "company_name"},
        ]
    }
    detection_workflow = Mock(spec=EntityDetectionWorkflow)
    detection_workflow.run.return_value = EntityDetectionResult(
        dataframe=pd.DataFrame(
            {
                COL_TEXT: ["Alice works at Acme"],
                COL_FINAL_ENTITIES: [final_entities],
                COL_SEED_VALIDATION_CANDIDATES: [validation_candidates],
            }
        ),
        failed_records=[],
    )
    replace_runner = Mock(spec=ReplacementWorkflow)
    replace_runner.run.return_value = ReplacementResult(
        dataframe=pd.DataFrame(
            {
                COL_TEXT: ["Alice works at Acme"],
                COL_REPLACED_TEXT: ["[REDACTED] works at [REDACTED]"],
                COL_FINAL_ENTITIES: [final_entities],
                COL_SEED_VALIDATION_CANDIDATES: [validation_candidates],
            }
        ),
        failed_records=[],
    )
    rewrite_runner = Mock(spec=RewriteWorkflow)
    rewrite_runner.run.return_value = RewriteResult(dataframe=pd.DataFrame(), failed_records=[])
    anonymizer = Anonymizer(
        detection_workflow=detection_workflow,
        replace_runner=replace_runner,
        rewrite_runner=rewrite_runner,
    )
    collector = MeasurementCollector(record_hash_key="test-key")

    with measurement_session(collector):
        anonymizer.run(
            config=AnonymizerConfig(replace=Redact(), detect=Detect(validation_max_entities_per_call=2)),
            data=AnonymizerInput(source=str(input_csv)),
        )

    record_metrics = [record for record in collector.records if record["record_type"] == "record"]
    assert len(record_metrics) == 1
    record = record_metrics[0]
    assert record["mode"] == "replace"
    assert record["strategy"] == "Redact"
    assert record["text_length_chars"] == len("Alice works at Acme")
    assert record["text_length_chars_bucket"] == "1-127"
    assert record["text_length_tokens"] > 0
    assert record["text_length_tokens_bucket"] == "1-127"
    assert record["final_entity_count"] == 2
    assert record["final_entity_label_counts"] == {"company_name": 1, "first_name": 1}
    assert record["detected_candidate_count"] == 2
    assert record["validation_chunk_count"] == 1
    assert record["original_value_leak_count"] == 0
    assert record["original_value_leak_label_counts"] == {}
    assert record["llm_calls_estimated_by_stage"] == {
        "entity_detection": 3,
        "replace_map_generation": 0,
    }
    assert record["llm_calls_estimated_total"] == 3
    assert len(record["record_hash"]) == 64

    stage_records = [record for record in collector.records if record["record_type"] == "stage"]
    assert any(record["stage"] == "Anonymizer._run_internal" for record in stage_records)
    assert any(record.get("input_rows_per_sec") is not None for record in stage_records)

    run_records = [record for record in collector.records if record["record_type"] == "run"]
    assert len(run_records) == 1
    run_record = run_records[0]
    assert run_record["mode"] == "replace"
    assert run_record["strategy"] == "Redact"
    assert run_record["input_row_count"] == 1
    assert run_record["input_source"] == {"kind": "local_file", "scheme": None, "suffix": ".csv"}
    assert run_record["input_text_column"] == "text"
    assert run_record["input_has_id_column"] is False
    assert run_record["input_has_data_summary"] is False
    assert run_record["detect"]["entity_label_source"] == "default"
    assert run_record["detect"]["entity_label_count"] > 0
    assert run_record["replace"]["strategy"] == "Redact"
    assert run_record["replace"]["normalize_label"] is True
    assert len(run_record["source_hash"]) == 64

    serialized = json.dumps(collector.records)
    assert "Alice" not in serialized
    assert "Acme" not in serialized
    assert str(input_csv) not in serialized


def test_anonymizer_measurement_config_writes_jsonl(tmp_path: Path) -> None:
    input_csv = tmp_path / "input.csv"
    output_jsonl = tmp_path / "measurements.jsonl"
    pd.DataFrame({"text": ["Alice works at Acme"]}).to_csv(input_csv, index=False)
    final_entities = {
        "entities": [
            {"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5},
        ]
    }
    detection_workflow = Mock(spec=EntityDetectionWorkflow)
    detection_workflow.run.return_value = EntityDetectionResult(
        dataframe=pd.DataFrame(
            {
                COL_TEXT: ["Alice works at Acme"],
                COL_FINAL_ENTITIES: [final_entities],
                COL_SEED_VALIDATION_CANDIDATES: [{"candidates": final_entities["entities"]}],
            }
        ),
        failed_records=[],
    )
    replace_runner = Mock(spec=ReplacementWorkflow)
    replace_runner.run.return_value = ReplacementResult(
        dataframe=pd.DataFrame(
            {
                COL_TEXT: ["Alice works at Acme"],
                COL_REPLACED_TEXT: ["[REDACTED] works at Acme"],
                COL_FINAL_ENTITIES: [final_entities],
                COL_SEED_VALIDATION_CANDIDATES: [{"candidates": final_entities["entities"]}],
            }
        ),
        failed_records=[],
    )
    anonymizer = Anonymizer(
        detection_workflow=detection_workflow,
        replace_runner=replace_runner,
        rewrite_runner=Mock(spec=RewriteWorkflow),
    )

    with configured_measurement_session(
        MeasurementConfig(
            output_path=output_jsonl,
            run_id="measurement-run",
            record_hash_key="test-key",
            run_tags={"config_id": "redact-default", "workload_id": "unit-small"},
        )
    ):
        anonymizer.run(
            config=AnonymizerConfig(replace=Redact()),
            data=AnonymizerInput(source=str(input_csv)),
        )

    records = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert {record["record_type"] for record in records} >= {"record", "run", "stage"}
    assert {record["run_id"] for record in records} == {"measurement-run"}
    assert {record["schema_version"] for record in records} == {MEASUREMENT_SCHEMA_VERSION}
    assert {record["run_tags"]["workload_id"] for record in records} == {"unit-small"}
    assert all(isinstance(record["timestamp_unix_sec"], float) for record in records)

    serialized = json.dumps(records)
    assert "Alice" not in serialized
    assert "Acme" not in serialized
    assert str(input_csv) not in serialized


def test_measurement_records_write_strict_json_safe_values(tmp_path: Path) -> None:
    output_jsonl = tmp_path / "measurements.jsonl"
    collector = MeasurementCollector(record_hash_key="test-key")
    collector.record("run", non_finite=float("nan"), mixed_set={1, "two"})

    collector.write_jsonl(output_jsonl)

    payload = json.loads(output_jsonl.read_text(encoding="utf-8"))
    assert payload["non_finite"] is None
    assert payload["mixed_set"] == [1, "two"]


def test_measurement_config_record_level_false_skips_record_rows(tmp_path: Path) -> None:
    output_json = tmp_path / "measurements.json"
    dataframe = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme"],
            COL_FINAL_ENTITIES: [{"entities": [{"value": "Alice", "label": "first_name"}]}],
        }
    )

    with configured_measurement_session(
        MeasurementConfig(
            output_path=output_json,
            output_format="json",
            record_level=False,
            run_id="stage-only",
            record_hash_key="test-key",
        )
    ):
        with stage_timer("example", input_row_count=1):
            pass
        record_record_metrics(
            dataframe,
            mode="replace",
            strategy="Redact",
            text_column=COL_TEXT,
            validation_max_entities_per_call=100,
        )

    records = json.loads(output_json.read_text(encoding="utf-8"))
    assert [record["record_type"] for record in records] == ["stage"]
    assert records[0]["run_id"] == "stage-only"
    assert records[0]["input_rows_per_sec"] >= 0


def test_measurement_config_from_env_returns_none_without_output_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = "ANON_TEST_EMPTY_MEASUREMENT_"
    monkeypatch.setenv(f"{prefix}RUN_ID", "env-run")

    assert MeasurementConfig.from_env(prefix=prefix) is None


def test_measurement_config_from_env_parses_supported_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prefix = "ANON_TEST_MEASUREMENT_"
    output_path = tmp_path / "measurements.json"
    monkeypatch.setenv(f"{prefix}OUTPUT_PATH", str(output_path))
    monkeypatch.setenv(f"{prefix}OUTPUT_FORMAT", "json")
    monkeypatch.setenv(f"{prefix}RECORD_LEVEL", "false")
    monkeypatch.setenv(f"{prefix}FAIL_ON_WRITE_ERROR", "true")
    monkeypatch.setenv(f"{prefix}RUN_ID", "env-run")
    monkeypatch.setenv(f"{prefix}RUN_TAGS", '{"config_id": "redact-default", "attempt": 2}')

    config = MeasurementConfig.from_env(prefix=prefix)

    assert config is not None
    assert config.output_path == str(output_path)
    assert config.output_format == "json"
    assert config.record_level is False
    assert config.fail_on_write_error is True
    assert config.run_id == "env-run"
    assert config.run_tags == {"config_id": "redact-default", "attempt": 2}


def test_measurement_config_from_sources_keeps_env_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prefix = "ANON_TEST_MEASUREMENT_"
    monkeypatch.setenv(f"{prefix}OUTPUT_PATH", str(tmp_path / "env.jsonl"))
    explicit = MeasurementConfig(output_path=tmp_path / "explicit.jsonl")

    assert MeasurementConfig.from_sources(env=False, prefix=prefix) is None
    assert MeasurementConfig.from_sources(explicit=explicit, env=True, prefix=prefix) is explicit


def test_measurement_config_from_env_reports_sanitized_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prefix = "ANON_TEST_MEASUREMENT_"
    secret_payload = "sk-secret-token-value"
    monkeypatch.setenv(f"{prefix}OUTPUT_PATH", str(tmp_path / "measurements.jsonl"))
    monkeypatch.setenv(f"{prefix}RUN_TAGS", secret_payload)

    with pytest.raises(ValueError) as exc_info:
        MeasurementConfig.from_env(prefix=prefix)

    message = str(exc_info.value)
    assert f"{prefix}RUN_TAGS" in message
    assert secret_payload not in message
    assert str(tmp_path) not in message


def test_default_measurement_env_prefix_is_anonymizer_scoped() -> None:
    assert DEFAULT_MEASUREMENT_ENV_PREFIX == "ANONYMIZER_MEASUREMENT_"


def test_measurement_config_write_errors_are_best_effort(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def raise_write_error(_self: MeasurementConfig, _collector: MeasurementCollector) -> None:
        raise OSError(f"cannot write {_self.output_path}")

    monkeypatch.setattr(MeasurementConfig, "write_collector", raise_write_error)
    caplog.set_level(logging.WARNING, logger="anonymizer.measurement")
    output_path = tmp_path / "secret-output-sk-live-value.jsonl"

    with configured_measurement_session(MeasurementConfig(output_path=output_path)) as collector:
        assert collector is not None
        collector.record("example")

    assert "Failed to write Anonymizer measurement records" in caplog.text
    assert str(output_path) not in caplog.text
    assert "sk-live-value" not in caplog.text


def test_measurement_config_strict_write_errors_can_fail_clean_body(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def raise_write_error(_self: MeasurementConfig, _collector: MeasurementCollector) -> None:
        raise OSError("cannot write")

    monkeypatch.setattr(MeasurementConfig, "write_collector", raise_write_error)

    with pytest.raises(OSError, match="cannot write"):
        with configured_measurement_session(
            MeasurementConfig(output_path=tmp_path / "measurements.jsonl", fail_on_write_error=True)
        ) as collector:
            assert collector is not None
            collector.record("example")


def test_measurement_config_strict_write_errors_still_close_collector(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    closed_run_ids: list[str] = []

    def raise_write_error(_self: MeasurementConfig, _collector: MeasurementCollector) -> None:
        raise OSError("cannot write")

    def close_collector(self: MeasurementCollector) -> None:
        closed_run_ids.append(self.run_id)

    monkeypatch.setattr(MeasurementConfig, "write_collector", raise_write_error)
    monkeypatch.setattr(MeasurementCollector, "close", close_collector)

    with pytest.raises(OSError, match="cannot write"):
        with configured_measurement_session(
            MeasurementConfig(
                output_path=tmp_path / "measurements.jsonl",
                fail_on_write_error=True,
                run_id="strict-write-run",
            )
        ) as collector:
            assert collector is not None
            collector.record("example")

    assert closed_run_ids == ["strict-write-run"]


def test_measurement_config_write_errors_do_not_mask_body_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def raise_write_error(_self: MeasurementConfig, _collector: MeasurementCollector) -> None:
        raise OSError("cannot write")

    monkeypatch.setattr(MeasurementConfig, "write_collector", raise_write_error)

    with pytest.raises(RuntimeError, match="body failed"):
        with configured_measurement_session(
            MeasurementConfig(output_path=tmp_path / "measurements.jsonl", fail_on_write_error=True)
        ) as collector:
            assert collector is not None
            collector.record("example")
            raise RuntimeError("body failed")


def test_streaming_measurement_session_writes_jsonl_without_retaining_records(tmp_path: Path) -> None:
    output_path = tmp_path / "measurements.jsonl"

    with configured_measurement_session(
        MeasurementConfig(output_path=output_path, streaming=True, keep_records=False)
    ) as collector:
        assert collector is not None
        collector.record("example", value=1)

        assert collector.records == []
        assert output_path.read_text(encoding="utf-8").count("\n") == 1

        collector.record("example", value=2)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["value"] for line in lines] == [1, 2]


def test_measurement_collector_close_attempts_all_sinks_after_failure() -> None:
    close_events: list[str] = []

    class FakeSink:
        def __init__(self, name: str, *, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        def write_record(self, record: dict[str, Any]) -> None:
            _ = record

        def close(self) -> None:
            close_events.append(self.name)
            if self.fail:
                raise OSError(f"{self.name} close failed")

    collector = MeasurementCollector(
        record_sink=FakeSink("records", fail=True),
        dd_trace_sink=FakeSink("dd-trace"),
        dd_task_trace_sink=FakeSink("dd-task-trace"),
    )

    with pytest.raises(OSError, match="records close failed"):
        collector.close()

    assert close_events == ["records", "dd-trace", "dd-task-trace"]


def test_streaming_measurement_requires_jsonl_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="streaming measurement output only supports jsonl"):
        MeasurementConfig(output_path=tmp_path / "measurements.json", output_format="json", streaming=True)


def test_dd_message_trace_requires_trace_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dd_trace_path is required"):
        MeasurementConfig(output_path=tmp_path / "measurements.jsonl", dd_trace="last_message")


@pytest.mark.parametrize("structured", [False, True])
def test_ndd_adapter_writes_native_dd_message_trace_and_strips_trace_columns(
    tmp_path: Path,
    *,
    structured: bool,
) -> None:
    input_df = pd.DataFrame(
        {
            "text": ["Alice works at Acme"],
            f"notes{TRACE_COLUMN_POSTFIX}": ["user supplied trace-looking column"],
            RECORD_ID_COLUMN: ["record-a"],
        }
    )
    original_column = (
        LLMStructuredColumnConfig(
            name="raw_detected",
            prompt="{{ text }}",
            model_alias="alias",
            output_format={"type": "object", "properties": {"entities": {"type": "array"}}},
        )
        if structured
        else LLMTextColumnConfig(
            name="raw_detected",
            prompt="{{ text }}",
            model_alias="alias",
        )
    )
    captured_columns: list[LLMTextColumnConfig] = []

    class TraceDataDesigner:
        def preview(self, config_builder: object, *, num_records: int) -> SimpleNamespace:
            traced_column = cast(Any, config_builder).get_column_config("raw_detected")
            captured_columns.append(traced_column)
            output = input_df.iloc[:num_records].copy()
            output["raw_detected"] = "[]"
            output[f"raw_detected{TRACE_COLUMN_POSTFIX}"] = [
                [
                    {"role": "system", "content": [{"type": "text", "text": "system secret"}]},
                    {"role": "user", "content": [{"type": "text", "text": "prompt secret"}]},
                    {"role": "assistant", "content": "secret response", "reasoning_content": "scratch"},
                ]
            ]
            return SimpleNamespace(dataset=output)

    adapter = NddAdapter(data_designer=cast(DataDesigner, TraceDataDesigner()))
    trace_path = tmp_path / "trace.jsonl"

    with configured_measurement_session(
        MeasurementConfig(
            output_path=tmp_path / "measurements.jsonl", dd_trace="last_message", dd_trace_path=trace_path
        )
    ):
        result = _run_entity_detection_preview(adapter, input_df, [original_column])

    assert original_column.with_trace == TraceType.NONE
    assert captured_columns[0].with_trace == TraceType.ALL_MESSAGES
    assert f"raw_detected{TRACE_COLUMN_POSTFIX}" not in result.dataframe.columns
    assert f"notes{TRACE_COLUMN_POSTFIX}" in result.dataframe.columns

    trace = json.loads(trace_path.read_text(encoding="utf-8").strip())
    assert trace["record_type"] == "dd_message_trace"
    assert trace["trace_source"] == "data_designer_column"
    assert trace["workflow_name"] == "entity-detection"
    assert trace["model_alias"] == "alias"
    assert trace["model_name"] == "dummy-model"
    assert trace["model_provider_name"] == "provider"
    assert trace["status"] == "completed"
    assert trace["messages"] == [{"role": "user", "content": [{"type": "text", "text": "prompt secret"}]}]
    assert trace["response"]["content"] == "secret response"
    assert trace["usage"] is None

    measurements = [json.loads(line) for line in (tmp_path / "measurements.jsonl").read_text().splitlines()]
    coverage = [record for record in measurements if record["record_type"] == "dd_trace_coverage"]
    assert len(coverage) == 1
    assert coverage[0]["traced_column_count"] == 1
    assert coverage[0]["unsupported_column_count"] == 0

    serialized_measurements = json.dumps(measurements)
    assert "prompt secret" not in serialized_measurements
    assert "secret response" not in serialized_measurements


class _TraceModelFacade:
    model_alias = "alias"
    model_name = "dummy-model"
    model_provider_name = "provider"
    model_provider = SimpleNamespace(endpoint="http://provider/v1")

    def generate(self, prompt: str, **_kwargs: Any) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return self.completion(messages).message.content

    def completion(self, _messages: list[dict[str, Any]]) -> SimpleNamespace:
        return SimpleNamespace(
            message=SimpleNamespace(content="custom response secret", reasoning_content="scratch", tool_calls=[]),
            usage=SimpleNamespace(input_tokens=3, output_tokens=5, total_tokens=8),
        )


class _TraceModelRegistry:
    def __init__(self, facade: Any | None = None) -> None:
        self._models = {"alias": facade or _TraceModelFacade()}

    def get_model(self, *, model_alias: str) -> _TraceModelFacade:
        return self._models[model_alias]


class _CustomTraceDataDesigner:
    def __init__(self, input_df: pd.DataFrame, *, facade: Any | None = None) -> None:
        self.input_df = input_df
        self.resource_provider = SimpleNamespace(model_registry=_TraceModelRegistry(facade))

    def _create_resource_provider(self, *_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return self.resource_provider

    def preview(self, config_builder: object, *, num_records: int) -> SimpleNamespace:
        resource_provider = self._create_resource_provider()
        model = resource_provider.model_registry.get_model(model_alias="alias")
        output = self.input_df.iloc[:num_records].copy()
        for column in cast(Any, config_builder).get_column_configs():
            for row_index, row in output.iterrows():
                generated = column.generator_function(
                    row.to_dict(),
                    generator_params=None,
                    models={"alias": model},
                )
                for key, value in generated.items():
                    output.loc[row_index, key] = value
        return SimpleNamespace(dataset=output)


class _TaskTraceDataDesigner:
    def __init__(
        self,
        input_df: pd.DataFrame,
        *,
        task_traces: list[Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.input_df = input_df
        self.task_traces = task_traces or []
        self.error = error
        self.run_config = RunConfig(async_trace=False)
        self.async_trace_values: list[bool] = []

    def set_run_config(self, run_config: RunConfig) -> None:
        self.async_trace_values.append(run_config.async_trace)
        self.run_config = run_config

    def preview(self, _config_builder: object, *, num_records: int) -> SimpleNamespace:
        assert self.run_config.async_trace is True
        if self.error is not None:
            raise self.error
        return SimpleNamespace(dataset=self.input_df.iloc[:num_records].copy(), task_traces=self.task_traces)


def _custom_trace_column(name: str, *, prompt: str, value: str) -> CustomColumnConfig:
    @custom_column_generator(required_columns=["text"], model_aliases=["alias"])
    def generator(
        row: dict[str, Any],
        generator_params: Any,
        models: dict[str, Any],
    ) -> dict[str, str]:
        _ = row, generator_params
        models["alias"].generate(prompt)
        return {name: value}

    return CustomColumnConfig(name=name, generator_function=generator)


def test_ndd_adapter_writes_custom_column_private_model_facade_dd_trace(
    tmp_path: Path,
    trace_input_df: pd.DataFrame,
) -> None:
    adapter = NddAdapter(data_designer=cast(DataDesigner, _CustomTraceDataDesigner(trace_input_df)))
    trace_path = tmp_path / "trace.jsonl"

    with configured_measurement_session(
        MeasurementConfig(
            output_path=tmp_path / "measurements.jsonl", dd_trace="all_messages", dd_trace_path=trace_path
        )
    ):
        _run_entity_detection_preview(
            adapter,
            trace_input_df,
            [
                _custom_trace_column("raw_detected", prompt="raw prompt secret", value="[]"),
                _custom_trace_column("quality_check", prompt="quality prompt secret", value="ok"),
            ],
        )

    traces = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    traces_by_column = {trace["column_name"]: trace for trace in traces}
    assert set(traces_by_column) == {"raw_detected", "quality_check"}

    raw_trace = traces_by_column["raw_detected"]
    assert raw_trace["record_type"] == "dd_message_trace"
    assert raw_trace["trace_source"] == "anonymizer_private_model_facade"
    assert raw_trace["workflow_name"] == "entity-detection"
    assert raw_trace["model_alias"] == "alias"
    assert raw_trace["model_name"] == "dummy-model"
    assert raw_trace["model_provider_name"] == "provider"
    assert raw_trace["model_provider_endpoint"] == "http://provider/v1"
    assert raw_trace["status"] == "completed"
    assert raw_trace["messages"] == [{"role": "user", "content": [{"type": "text", "text": "raw prompt secret"}]}]
    assert raw_trace["response"]["content"] == "custom response secret"
    assert raw_trace["response"]["reasoning_content"] == "scratch"
    assert raw_trace["usage"] == {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8}

    quality_trace = traces_by_column["quality_check"]
    assert quality_trace["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "quality prompt secret"}]}
    ]

    measurements = [json.loads(line) for line in (tmp_path / "measurements.jsonl").read_text().splitlines()]
    coverage = [record for record in measurements if record["record_type"] == "dd_trace_coverage"]
    assert len(coverage) == 1
    assert coverage[0]["trace_backend"] == "anonymizer_private_model_facade"
    assert coverage[0]["traced_column_count"] == 2
    assert coverage[0]["private_trace_column_count"] == 2
    assert coverage[0]["private_trace_column_names"] == ["raw_detected", "quality_check"]
    assert coverage[0]["unsupported_column_count"] == 0

    serialized_measurements = json.dumps(measurements)
    assert "raw prompt secret" not in serialized_measurements
    assert "quality prompt secret" not in serialized_measurements
    assert "custom response secret" not in serialized_measurements


def test_ndd_adapter_private_model_facade_trace_write_error_is_not_wrapped(
    trace_input_df: pd.DataFrame,
) -> None:
    adapter = NddAdapter(data_designer=cast(DataDesigner, _CustomTraceDataDesigner(trace_input_df)))
    collector = MeasurementCollector(
        dd_trace_mode="all_messages",
        dd_trace_sink=_FailingSink("private trace sidecar unavailable"),
        fail_on_write_error=True,
    )

    with measurement_session(collector), pytest.raises(OSError, match="private trace sidecar unavailable"):
        _run_entity_detection_preview(
            adapter,
            trace_input_df,
            [_custom_trace_column("raw_detected", prompt="raw prompt secret", value="[]")],
        )


def test_ndd_adapter_flushes_private_model_facade_error_trace_when_workflow_fails(
    tmp_path: Path,
    trace_input_df: pd.DataFrame,
) -> None:
    class FailingTraceModelFacade(_TraceModelFacade):
        def completion(self, _messages: list[dict[str, Any]]) -> SimpleNamespace:
            raise RuntimeError("custom model call failed")

    trace_path = tmp_path / "trace.jsonl"
    adapter = NddAdapter(
        data_designer=cast(DataDesigner, _CustomTraceDataDesigner(trace_input_df, facade=FailingTraceModelFacade()))
    )

    with pytest.raises(AnonymizerWorkflowError, match="Workflow failed"):
        with configured_measurement_session(
            MeasurementConfig(
                output_path=tmp_path / "measurements.jsonl",
                dd_trace="all_messages",
                dd_trace_path=trace_path,
            )
        ):
            _run_entity_detection_preview(
                adapter,
                trace_input_df,
                [_custom_trace_column("raw_detected", prompt="raw prompt secret", value="[]")],
            )

    traces = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert len(traces) == 1
    trace = traces[0]
    assert trace["record_type"] == "dd_message_trace"
    assert trace["trace_source"] == "anonymizer_private_model_facade"
    assert trace["workflow_name"] == "entity-detection"
    assert trace["column_name"] == "raw_detected"
    assert trace["status"] == "error"
    assert trace["error_type"] == "RuntimeError"
    assert trace["messages"] == [{"role": "user", "content": [{"type": "text", "text": "raw prompt secret"}]}]
    assert "custom model call failed" not in trace_path.read_text(encoding="utf-8")


def test_ndd_adapter_private_model_facade_trace_write_error_does_not_mask_workflow_failure(
    trace_input_df: pd.DataFrame,
) -> None:
    class FailingTraceModelFacade(_TraceModelFacade):
        def completion(self, _messages: list[dict[str, Any]]) -> SimpleNamespace:
            raise RuntimeError("custom model call failed")

    adapter = NddAdapter(
        data_designer=cast(DataDesigner, _CustomTraceDataDesigner(trace_input_df, facade=FailingTraceModelFacade()))
    )
    collector = MeasurementCollector(
        dd_trace_mode="all_messages",
        dd_trace_sink=_FailingSink("private trace sidecar unavailable"),
        fail_on_write_error=True,
    )

    with measurement_session(collector), pytest.raises(AnonymizerWorkflowError, match="Workflow failed"):
        _run_entity_detection_preview(
            adapter,
            trace_input_df,
            [_custom_trace_column("raw_detected", prompt="raw prompt secret", value="[]")],
        )


def test_ndd_adapter_writes_sanitized_dd_task_traces_and_restores_run_config(
    tmp_path: Path,
    trace_input_df: pd.DataFrame,
) -> None:
    task_trace = SimpleNamespace(
        column="raw_detected",
        row_group=0,
        row_index=7,
        task_type="llm",
        dispatched_at=10.0,
        slot_acquired_at=10.25,
        completed_at=12.0,
        status="error",
        error="raw secret token Alice",
    )
    data_designer = _TaskTraceDataDesigner(trace_input_df, task_traces=[task_trace])
    adapter = NddAdapter(data_designer=cast(DataDesigner, data_designer))
    task_trace_path = tmp_path / "task-trace.jsonl"

    with configured_measurement_session(
        MeasurementConfig(output_path=tmp_path / "measurements.jsonl", dd_task_trace_path=task_trace_path)
    ):
        _run_entity_detection_preview(
            adapter,
            trace_input_df,
            [_raw_detected_text_column()],
        )

    assert data_designer.async_trace_values == [True, False]
    assert data_designer.run_config.async_trace is False
    task_trace = json.loads(task_trace_path.read_text(encoding="utf-8").strip())
    assert task_trace["record_type"] == "dd_task_trace"
    assert task_trace["workflow_name"] == "entity-detection"
    assert task_trace["column"] == "raw_detected"
    assert task_trace["row_group"] == 0
    assert task_trace["row_index"] == 7
    assert task_trace["task_type"] == "llm"
    assert task_trace["status"] == "error"
    assert task_trace["error_present"] is True
    assert task_trace["dispatched_offset_sec"] == pytest.approx(0.0)
    assert task_trace["slot_acquired_offset_sec"] == pytest.approx(0.25)
    assert task_trace["completed_offset_sec"] == pytest.approx(2.0)
    assert task_trace["queue_wait_sec"] == pytest.approx(0.25)
    assert task_trace["execution_sec"] == pytest.approx(1.75)
    assert task_trace["total_sec"] == pytest.approx(2.0)
    assert "raw secret token Alice" not in task_trace_path.read_text(encoding="utf-8")
    assert "raw secret token Alice" not in (tmp_path / "measurements.jsonl").read_text(encoding="utf-8")


def test_ndd_adapter_task_trace_handles_mapping_and_invalid_timestamps(
    tmp_path: Path,
    trace_input_df: pd.DataFrame,
) -> None:
    task_traces = [
        {
            "column": "missing_timestamps",
            "row_group": 0,
            "row_index": 1,
            "task_type": "llm",
            "status": "completed",
            "error": None,
        },
        {
            "column": "nonpositive_timestamps",
            "row_group": 0,
            "row_index": 2,
            "task_type": "llm",
            "dispatched_at": 0.0,
            "slot_acquired_at": -1.0,
            "completed_at": -2.0,
            "status": "completed",
            "error": None,
        },
        {
            "column": "out_of_order_timestamps",
            "row_group": 0,
            "row_index": 3,
            "task_type": "llm",
            "dispatched_at": 20.0,
            "slot_acquired_at": 19.0,
            "completed_at": 18.0,
            "status": "completed",
            "error": None,
        },
    ]
    task_trace_path = tmp_path / "task-trace.jsonl"
    adapter = NddAdapter(
        data_designer=cast(DataDesigner, _TaskTraceDataDesigner(trace_input_df, task_traces=task_traces))
    )

    with configured_measurement_session(
        MeasurementConfig(output_path=tmp_path / "measurements.jsonl", dd_task_trace_path=task_trace_path)
    ):
        _run_entity_detection_preview(
            adapter,
            trace_input_df,
            [_raw_detected_text_column()],
        )

    traces = [json.loads(line) for line in task_trace_path.read_text(encoding="utf-8").splitlines()]
    traces_by_column = {trace["column"]: trace for trace in traces}

    missing = traces_by_column["missing_timestamps"]
    assert missing["dispatched_offset_sec"] is None
    assert missing["slot_acquired_offset_sec"] is None
    assert missing["completed_offset_sec"] is None
    assert missing["queue_wait_sec"] is None
    assert missing["execution_sec"] is None
    assert missing["total_sec"] is None

    nonpositive = traces_by_column["nonpositive_timestamps"]
    assert nonpositive["dispatched_offset_sec"] is None
    assert nonpositive["slot_acquired_offset_sec"] is None
    assert nonpositive["completed_offset_sec"] is None
    assert nonpositive["queue_wait_sec"] is None
    assert nonpositive["execution_sec"] is None
    assert nonpositive["total_sec"] is None

    out_of_order = traces_by_column["out_of_order_timestamps"]
    assert out_of_order["dispatched_offset_sec"] == pytest.approx(0.0)
    assert out_of_order["slot_acquired_offset_sec"] is None
    assert out_of_order["completed_offset_sec"] is None
    assert out_of_order["queue_wait_sec"] is None
    assert out_of_order["execution_sec"] is None
    assert out_of_order["total_sec"] is None


def test_ndd_adapter_task_trace_write_error_is_not_wrapped_as_workflow_error(
    trace_input_df: pd.DataFrame,
) -> None:
    task_trace = SimpleNamespace(
        column="raw_detected",
        row_group=0,
        row_index=7,
        task_type="llm",
        dispatched_at=10.0,
        slot_acquired_at=10.25,
        completed_at=12.0,
        status="completed",
        error=None,
    )
    adapter = NddAdapter(
        data_designer=cast(DataDesigner, _TaskTraceDataDesigner(trace_input_df, task_traces=[task_trace]))
    )
    collector = MeasurementCollector(
        dd_task_trace_sink=_FailingSink("task trace sidecar unavailable"),
        fail_on_write_error=True,
    )

    with measurement_session(collector), pytest.raises(OSError, match="task trace sidecar unavailable"):
        _run_entity_detection_preview(
            adapter,
            trace_input_df,
            [_raw_detected_text_column()],
        )


def test_ndd_adapter_trace_write_error_is_not_wrapped_as_workflow_error(
    trace_input_df: pd.DataFrame,
) -> None:
    class TraceDataDesigner:
        def preview(self, _config_builder: object, *, num_records: int) -> SimpleNamespace:
            output = trace_input_df.iloc[:num_records].copy()
            output["raw_detected"] = "[]"
            output[f"raw_detected{TRACE_COLUMN_POSTFIX}"] = [
                [
                    {"role": "user", "content": [{"type": "text", "text": "prompt secret"}]},
                    {"role": "assistant", "content": "secret response"},
                ]
            ]
            return SimpleNamespace(dataset=output)

    adapter = NddAdapter(data_designer=cast(DataDesigner, TraceDataDesigner()))
    collector = MeasurementCollector(
        dd_trace_mode="last_message",
        dd_trace_sink=_FailingSink("trace sidecar unavailable"),
        fail_on_write_error=True,
    )

    with measurement_session(collector), pytest.raises(OSError, match="trace sidecar unavailable"):
        _run_entity_detection_preview(
            adapter,
            trace_input_df,
            [_raw_detected_text_column()],
        )


def test_ndd_adapter_restores_run_config_when_task_traced_workflow_fails(
    tmp_path: Path,
    trace_input_df: pd.DataFrame,
) -> None:
    data_designer = _TaskTraceDataDesigner(trace_input_df, error=RuntimeError("raw secret failure"))
    adapter = NddAdapter(data_designer=cast(DataDesigner, data_designer))

    with pytest.raises(AnonymizerWorkflowError, match="Workflow failed"):
        with configured_measurement_session(
            MeasurementConfig(output_path=tmp_path / "measurements.jsonl", dd_task_trace_path=tmp_path / "task.jsonl")
        ):
            _run_entity_detection_preview(
                adapter,
                trace_input_df,
                [_raw_detected_text_column()],
            )

    assert data_designer.async_trace_values == [True, False]
    assert data_designer.run_config.async_trace is False


def test_record_metrics_capture_generic_counts_without_raw_values() -> None:
    final_entities = {
        "entities": [
            {"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5},
            {"value": "Acme", "label": "company_name", "start_position": 15, "end_position": 19},
        ]
    }
    ground_truth_entities = {
        "entities": [
            {"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5},
            {"value": "Beta", "label": "company_name", "start_position": 15, "end_position": 19},
        ]
    }
    replacement_map = {
        "replacements": [
            {"original": "Alice", "label": "first_name", "synthetic": "Maya"},
            {"original": "Acme", "label": "company_name", "synthetic": "Maya"},
        ]
    }
    dataframe = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme"],
            COL_FINAL_ENTITIES: [final_entities],
            "ground_truth_entities": [ground_truth_entities],
            COL_REPLACEMENT_MAP: [replacement_map],
            COL_SEED_VALIDATION_CANDIDATES: [{"candidates": final_entities["entities"]}],
            COL_REPAIR_ITERATIONS: [2],
            COL_UTILITY_SCORE: [0.82],
            COL_LEAKAGE_MASS: [0.2],
            COL_WEIGHTED_LEAKAGE_RATE: [0.1],
            COL_ANY_HIGH_LEAKED: [False],
            COL_NEEDS_HUMAN_REVIEW: [True],
            COL_NEEDS_REPAIR: [False],
        }
    )
    collector = MeasurementCollector(record_hash_key="test-key")

    with measurement_session(collector):
        record_record_metrics(
            dataframe,
            mode="rewrite",
            strategy="Rewrite",
            text_column=COL_TEXT,
            validation_max_entities_per_call=2,
        )

    record = collector.records[0]
    assert record["ground_truth_entity_count"] == 2
    assert record["ground_truth_entity_label_counts"] == {"company_name": 1, "first_name": 1}
    assert record["entity_true_positive_count"] == 1
    assert record["entity_false_positive_count"] == 1
    assert record["entity_false_negative_count"] == 1
    assert record["entity_precision"] == 0.5
    assert record["entity_recall"] == 0.5
    assert record["entity_f1"] == 0.5
    assert record["entity_relaxed_gt_found_count"] == 2
    assert record["entity_relaxed_detected_tp_count"] == 2
    assert record["entity_relaxed_label_compatible_gt_found_count"] == 2
    assert record["entity_relaxed_label_compatible_detected_tp_count"] == 2
    assert record["entity_relaxed_precision"] == 1.0
    assert record["entity_relaxed_recall"] == 1.0
    assert record["entity_relaxed_f1"] == 1.0
    assert record["entity_relaxed_label_compatible_precision"] == 1.0
    assert record["entity_relaxed_label_compatible_recall"] == 1.0
    assert record["entity_relaxed_label_compatible_f1"] == 1.0
    assert record["replacement_count"] == 2
    assert record["replacement_label_counts"] == {"company_name": 1, "first_name": 1}
    assert record["replacement_duplicate_value_count"] == 1
    assert record["replacement_missing_final_entity_count"] == 0
    assert record["replacement_missing_final_entity_label_counts"] == {}
    assert record["replacement_missing_final_value_count"] == 0
    assert record["replacement_synthetic_original_collision_count"] == 0
    assert record["replacement_synthetic_original_collision_label_counts"] == {}
    assert record["replacement_synthetic_original_collision_value_count"] == 0
    assert record["repair_iterations"] == 2
    assert record["utility_score"] == 0.82
    assert record["leakage_mass"] == 0.2
    assert record["weighted_leakage_rate"] == 0.1
    assert record["any_high_leaked"] is False
    assert record["needs_human_review"] is True
    assert record["needs_repair"] is False

    serialized = json.dumps(collector.records)
    assert "Alice" not in serialized
    assert "Acme" not in serialized
    assert "Beta" not in serialized
    assert "Maya" not in serialized


def test_record_metrics_counts_duplicate_ground_truth_entities_by_occurrence() -> None:
    final_entities = {
        "entities": [
            {"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5},
        ]
    }
    ground_truth_entities = {
        "entities": [
            {"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5},
            {"value": "Alice", "label": "first_name", "start_position": 18, "end_position": 23},
        ]
    }
    dataframe = pd.DataFrame(
        {
            COL_TEXT: ["Alice talked with Alice"],
            COL_FINAL_ENTITIES: [final_entities],
            "ground_truth_entities": [ground_truth_entities],
        }
    )
    collector = MeasurementCollector(record_hash_key="test-key")

    with measurement_session(collector):
        record_record_metrics(
            dataframe,
            mode="replace",
            strategy="Redact",
            text_column=COL_TEXT,
            validation_max_entities_per_call=2,
        )

    record = collector.records[0]
    assert record["ground_truth_entity_count"] == 2
    assert record["ground_truth_entity_label_counts"] == {"first_name": 2}
    assert record["entity_true_positive_count"] == 1
    assert record["entity_false_positive_count"] == 0
    assert record["entity_false_negative_count"] == 1
    assert record["entity_precision"] == 1.0
    assert record["entity_recall"] == 0.5
    assert record["entity_f1"] == pytest.approx(2 / 3)
    assert record["entity_relaxed_gt_found_count"] == 1
    assert record["entity_relaxed_detected_tp_count"] == 1
    assert record["entity_relaxed_precision"] == 1.0
    assert record["entity_relaxed_recall"] == 0.5


def test_record_metrics_capture_relaxed_gt_label_equivalence_without_raw_values() -> None:
    final_entities = {
        "entities": [
            {"value": "builduser42", "label": "user_name", "start_position": 4, "end_position": 15},
        ]
    }
    ground_truth_entities = {
        "entities": [
            {"value": "legacy-user", "label": "username", "start_position": 6, "end_position": 14},
        ]
    }
    dataframe = pd.DataFrame(
        {
            COL_TEXT: ["ssh builduser42@host"],
            COL_FINAL_ENTITIES: [final_entities],
            "ground_truth_entities": [ground_truth_entities],
        }
    )
    collector = MeasurementCollector(record_hash_key="test-key")

    with measurement_session(collector):
        record_record_metrics(
            dataframe,
            mode="replace",
            strategy="Redact",
            text_column=COL_TEXT,
            validation_max_entities_per_call=2,
        )

    record = collector.records[0]
    assert record["entity_true_positive_count"] == 0
    assert record["entity_false_positive_count"] == 1
    assert record["entity_false_negative_count"] == 1
    assert record["entity_relaxed_gt_found_count"] == 1
    assert record["entity_relaxed_detected_tp_count"] == 1
    assert record["entity_relaxed_label_compatible_gt_found_count"] == 1
    assert record["entity_relaxed_label_compatible_detected_tp_count"] == 1
    assert record["entity_relaxed_precision"] == 1.0
    assert record["entity_relaxed_recall"] == 1.0
    assert record["entity_relaxed_f1"] == 1.0
    assert record["entity_relaxed_label_compatible_precision"] == 1.0
    assert record["entity_relaxed_label_compatible_recall"] == 1.0
    assert record["entity_relaxed_label_compatible_f1"] == 1.0

    serialized = json.dumps(collector.records)
    assert "builduser42" not in serialized
    assert "legacy-user" not in serialized


def test_record_metrics_counts_missing_replacement_map_entries_without_raw_values() -> None:
    final_entities = {
        "entities": [
            {"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5},
            {"value": "2030-01-01", "label": "date", "start_position": 13, "end_position": 23},
            {"value": "2030-01-01", "label": "date", "start_position": 27, "end_position": 37},
        ]
    }
    replacement_map = {
        "replacements": [
            {"original": "Alice", "label": "first_name", "synthetic": "Maya"},
        ]
    }
    dataframe = pd.DataFrame(
        {
            COL_TEXT: ["Alice filed 2030-01-01 and 2030-01-01"],
            COL_FINAL_ENTITIES: [final_entities],
            COL_REPLACEMENT_MAP: [replacement_map],
        }
    )
    collector = MeasurementCollector(record_hash_key="test-key")

    with measurement_session(collector):
        record_record_metrics(
            dataframe,
            mode="replace",
            strategy="Substitute",
            text_column=COL_TEXT,
            validation_max_entities_per_call=100,
        )

    record = collector.records[0]
    assert record["replacement_missing_final_entity_count"] == 2
    assert record["replacement_missing_final_entity_label_counts"] == {"date": 2}
    assert record["replacement_missing_final_value_count"] == 1
    assert record["replacement_synthetic_original_collision_count"] == 0
    assert record["replacement_synthetic_original_collision_label_counts"] == {}
    assert record["replacement_synthetic_original_collision_value_count"] == 0

    serialized = json.dumps(collector.records)
    assert "Alice" not in serialized
    assert "2030-01-01" not in serialized
    assert "Maya" not in serialized


def test_record_metrics_counts_synthetic_original_collisions_without_raw_values() -> None:
    final_entities = {
        "entities": [
            {"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5},
            {"value": "2030-01-01", "label": "date", "start_position": 13, "end_position": 23},
        ]
    }
    replacement_map = {
        "replacements": [
            {"original": "Alice", "label": "first_name", "synthetic": "Maya"},
            {"original": "2029-12-01", "label": "date", "synthetic": "2030-01-01"},
        ]
    }
    dataframe = pd.DataFrame(
        {
            COL_TEXT: ["Alice filed 2030-01-01"],
            COL_FINAL_ENTITIES: [final_entities],
            COL_REPLACEMENT_MAP: [replacement_map],
        }
    )
    collector = MeasurementCollector(record_hash_key="test-key")

    with measurement_session(collector):
        record_record_metrics(
            dataframe,
            mode="replace",
            strategy="Substitute",
            text_column=COL_TEXT,
            validation_max_entities_per_call=100,
        )

    record = collector.records[0]
    assert record["replacement_synthetic_original_collision_count"] == 1
    assert record["replacement_synthetic_original_collision_label_counts"] == {"date": 1}
    assert record["replacement_synthetic_original_collision_value_count"] == 1

    serialized = json.dumps(collector.records)
    assert "Alice" not in serialized
    assert "2030-01-01" not in serialized
    assert "2029-12-01" not in serialized
    assert "Maya" not in serialized


def test_record_metrics_counts_original_value_replacement_leaks_without_raw_values() -> None:
    leaked_key = "sk-test-AAAAAAAAAAAAAAAAAAAAAAAA"
    dataframe = pd.DataFrame(
        {
            COL_TEXT: [f"token={leaked_key}"],
            COL_REPLACED_TEXT: [f"still token={leaked_key}"],
            COL_FINAL_ENTITIES: [{"entities": [{"value": leaked_key, "label": "api_key"}]}],
        }
    )
    collector = MeasurementCollector(record_hash_key="test-key")

    with measurement_session(collector):
        record_record_metrics(
            dataframe,
            mode="replace",
            strategy="Hash",
            text_column=COL_TEXT,
            validation_max_entities_per_call=100,
        )

    record = collector.records[0]
    assert record["original_value_leak_count"] == 1
    assert record["original_value_leak_label_counts"] == {"api_key": 1}
    assert leaked_key not in json.dumps(collector.records)


def test_record_metrics_ignores_short_value_inside_hash_replacement_token() -> None:
    dataframe = pd.DataFrame(
        {
            COL_TEXT: ["Alice is 34 years old."],
            COL_REPLACED_TEXT: ["Alice is <HASH_AGE_ab34cd> years old."],
            COL_FINAL_ENTITIES: [{"entities": [{"value": "34", "label": "age"}]}],
        }
    )
    collector = MeasurementCollector(record_hash_key="test-key")

    with measurement_session(collector):
        record_record_metrics(
            dataframe,
            mode="replace",
            strategy="Hash",
            text_column=COL_TEXT,
            validation_max_entities_per_call=100,
        )

    record = collector.records[0]
    assert record["original_value_leak_count"] == 0
    assert record["original_value_leak_label_counts"] == {}


def test_record_metrics_counts_standalone_short_value_replacement_leaks() -> None:
    dataframe = pd.DataFrame(
        {
            COL_TEXT: ["Alice is 34 years old."],
            COL_REPLACED_TEXT: ["Alice is 34 years old."],
            COL_FINAL_ENTITIES: [{"entities": [{"value": "34", "label": "age"}]}],
        }
    )
    collector = MeasurementCollector(record_hash_key="test-key")

    with measurement_session(collector):
        record_record_metrics(
            dataframe,
            mode="replace",
            strategy="Hash",
            text_column=COL_TEXT,
            validation_max_entities_per_call=100,
        )

    record = collector.records[0]
    assert record["original_value_leak_count"] == 1
    assert record["original_value_leak_label_counts"] == {"age": 1}


def test_record_metrics_normalizes_integral_row_index_types() -> None:
    dataframe = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme"],
            COL_FINAL_ENTITIES: [{"entities": [{"value": "Alice", "label": "first_name"}]}],
        },
        index=pd.Index([np.int64(7)]),
    )
    collector = MeasurementCollector(record_hash_key="test-key")

    with measurement_session(collector):
        record_record_metrics(
            dataframe,
            mode="replace",
            strategy="Redact",
            text_column=COL_TEXT,
            validation_max_entities_per_call=100,
        )

    assert collector.records[0]["row_index"] == 7


def test_record_hash_uses_run_scoped_secret_by_default() -> None:
    first = MeasurementCollector()
    second = MeasurementCollector()

    assert first.record_hash(row_index=0, text="Alice works at Acme") != second.record_hash(
        row_index=0,
        text="Alice works at Acme",
    )


def test_stage_timer_records_errors() -> None:
    workflow = EntityDetectionWorkflow(adapter=Mock(spec=NddAdapter))
    collector = MeasurementCollector(record_hash_key="test-key")

    with measurement_session(collector), pytest.raises(ValueError, match="privacy_goal is required"):
        workflow.run(
            pd.DataFrame({COL_TEXT: ["Alice"]}),
            model_configs=[],
            selected_models=DetectionModelSelection(
                entity_detector="detector",
                entity_validator=["validator"],
                entity_augmenter="augmenter",
                latent_detector="latent",
            ),
            gliner_detection_threshold=0.3,
            tag_latent_entities=True,
            privacy_goal=None,
        )

    stage_records = [record for record in collector.records if record["record_type"] == "stage"]
    assert len(stage_records) == 1
    record = stage_records[0]
    assert record["schema_version"] == MEASUREMENT_SCHEMA_VERSION
    assert record["record_type"] == "stage"
    assert record["run_id"] == collector.run_id
    assert record["run_tags"] == {}
    assert isinstance(record["timestamp_unix_sec"], float)
    assert record["stage"] == "EntityDetectionWorkflow.run"
    assert record["status"] == "error"
    assert record["elapsed_sec"] >= 0
    assert record["input_row_count"] == 1
    assert record["input_rows_per_sec"] >= 0
    assert record["output_rows_per_sec"] is None
    assert record["tag_latent_entities"] is True


def test_rewrite_llm_call_estimate_splits_by_stage() -> None:
    calls = estimate_llm_calls_by_stage(
        mode="rewrite",
        strategy="Rewrite",
        has_grouped_entities=True,
        validation_chunk_count=2,
        repair_iterations=2,
    )

    assert calls == {
        "entity_detection": 4,
        "latent_entity_detection": 1,
        "replace_map_generation": 1,
        "rewrite_pipeline": 5,
        "rewrite_evaluate": 9,
        "rewrite_repair": 2,
        "rewrite_final_judge": 1,
    }


def test_rewrite_llm_call_estimate_skips_rewrite_body_without_entities() -> None:
    calls = estimate_llm_calls_by_stage(
        mode="rewrite",
        strategy="Rewrite",
        has_grouped_entities=False,
        validation_chunk_count=0,
        repair_iterations=2,
    )

    assert calls == {
        "entity_detection": 2,
        "latent_entity_detection": 0,
        "replace_map_generation": 0,
        "rewrite_pipeline": 0,
        "rewrite_evaluate": 0,
        "rewrite_repair": 0,
        "rewrite_final_judge": 0,
    }
