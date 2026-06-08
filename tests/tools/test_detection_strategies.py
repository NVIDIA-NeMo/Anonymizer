# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pandas as pd

from anonymizer.engine.constants import (
    COL_DETECTED_ENTITIES,
    COL_RAW_DETECTED,
    COL_SEED_ENTITIES,
    COL_SEED_ENTITIES_JSON,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
)
from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
from anonymizer.engine.ndd.model_loader import load_default_model_selection
from anonymizer.engine.schemas import EntitiesSchema
from anonymizer.measurement import MeasurementCollector, measurement_session

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_tool(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def test_native_candidate_validate_no_augment_strategy_skips_data_designer_and_augmentation() -> None:
    tool = load_tool(
        "measurement_detection_strategies_native_candidate_validate",
        REPO_ROOT / "tools/measurement/detection_strategies.py",
    )

    class SequencedClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []
            self.outputs = [
                '{"entities": [{"value": "Alice", "label": "first_name", "reason": "name"}]}',
                '{"decisions": [{"id": "first_name_0_5", "decision": "keep", "reason": "real name"}]}',
            ]

        def complete(self, request):  # type: ignore[no-untyped-def]
            self.prompts.append(request.prompt)
            return SimpleNamespace(
                content=self.outputs.pop(0),
                elapsed_sec=0.1,
                usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            )

    adapter = Mock()
    client = SequencedClient()
    collector = MeasurementCollector(record_hash_key="test-key")

    with measurement_session(collector):
        with tool.experimental_detection_strategy_context(
            tool.ExperimentalDetectionStrategy.native_candidate_validate_no_augment,
            native_client=client,
        ):
            workflow = EntityDetectionWorkflow(adapter=adapter)
            result = workflow.detect_and_validate_entities(
                pd.DataFrame({COL_TEXT: ["Alice works at NVIDIA."]}),
                model_configs=[],
                selected_models=load_default_model_selection().detection,
                gliner_detection_threshold=0.3,
                validation_single_chunk_full_text=False,
                entity_labels=["first_name", "organization_name"],
            )

    adapter.run_workflow.assert_not_called()
    entities = EntitiesSchema.from_raw(result.dataframe[COL_DETECTED_ENTITIES].iloc[0]).entities
    assert [(entity.label, entity.value, entity.source) for entity in entities] == [
        ("first_name", "Alice", "direct_seed"),
    ]
    assert len(client.prompts) == 2
    assert all("Find additional sensitive entities" not in prompt for prompt in client.prompts)
    records = [record for record in collector.records if record["record_type"] == "model_workflow"]
    assert len(records) == 1
    record = records[0]
    assert record["workflow_name"] == "entity-detection-native-candidate-validate-no-augment"
    assert record["observed_total_requests"] == 2
    assert record["observed_total_tokens"] == 28


def test_detector_native_validate_no_augment_strategy_reuses_detector_seed_and_direct_validation() -> None:
    tool = load_tool(
        "measurement_detection_strategies_detector_native_validate",
        REPO_ROOT / "tools/measurement/detection_strategies.py",
    )
    text = "Alice works at NVIDIA."
    seed_row = {
        COL_TEXT: text,
        COL_RAW_DETECTED: "",
        COL_SEED_ENTITIES: EntitiesSchema(
            entities=[
                {
                    "id": "first_name_0_5",
                    "value": "Alice",
                    "label": "first_name",
                    "start_position": 0,
                    "end_position": 5,
                    "score": 0.9,
                    "source": "detector",
                }
            ]
        ).model_dump(mode="json"),
        COL_TAG_NOTATION: "xml",
    }
    tool.prepare_validation_inputs(seed_row)

    class ValidationClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete(self, request):  # type: ignore[no-untyped-def]
            self.prompts.append(request.prompt)
            return SimpleNamespace(
                content='{"decisions": [{"id": "first_name_0_5", "decision": "keep", "reason": "real name"}]}',
                elapsed_sec=0.1,
                usage={"prompt_tokens": 40, "completion_tokens": 8, "total_tokens": 48},
            )

    adapter = Mock()
    adapter.run_workflow.return_value = SimpleNamespace(dataframe=pd.DataFrame([seed_row]), failed_records=[])
    client = ValidationClient()
    collector = MeasurementCollector(record_hash_key="test-key")

    with measurement_session(collector):
        with tool.experimental_detection_strategy_context(
            tool.ExperimentalDetectionStrategy.detector_native_validate_no_augment,
            native_client=client,
        ):
            workflow = EntityDetectionWorkflow(adapter=adapter)
            result = workflow.detect_and_validate_entities(
                pd.DataFrame({COL_TEXT: [text]}),
                model_configs=[],
                selected_models=load_default_model_selection().detection,
                gliner_detection_threshold=0.3,
                validation_single_chunk_full_text=True,
                entity_labels=["first_name", "organization_name"],
            )

    adapter.run_workflow.assert_called_once()
    assert adapter.run_workflow.call_args.kwargs["workflow_name"] == (
        "entity-detection-detector-native-validate-no-augment-seed"
    )
    entities = EntitiesSchema.from_raw(result.dataframe[COL_DETECTED_ENTITIES].iloc[0]).entities
    assert [(entity.label, entity.value, entity.source) for entity in entities] == [
        ("first_name", "Alice", "detector"),
    ]
    assert len(client.prompts) == 1
    assert all("Find additional sensitive entities" not in prompt for prompt in client.prompts)
    records = [record for record in collector.records if record["record_type"] == "model_workflow"]
    assert len(records) == 1


def test_detector_native_validate_no_augment_ignores_invalid_reclass_labels() -> None:
    tool = load_tool(
        "measurement_detection_strategies_detector_native_validate_invalid_label",
        REPO_ROOT / "tools/measurement/detection_strategies.py",
    )
    text = "Alice works at NVIDIA."
    seed_row = {
        COL_TEXT: text,
        COL_RAW_DETECTED: "",
        COL_SEED_ENTITIES: EntitiesSchema(
            entities=[
                {
                    "id": "first_name_0_5",
                    "value": "Alice",
                    "label": "first_name",
                    "start_position": 0,
                    "end_position": 5,
                    "score": 0.9,
                    "source": "detector",
                }
            ]
        ).model_dump(mode="json"),
        COL_TAG_NOTATION: "xml",
    }
    tool.prepare_validation_inputs(seed_row)

    class ValidationClient:
        def complete(self, request):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                content=(
                    '{"decisions": ['
                    '{"id": "first_name_0_5", "decision": "reclass", '
                    '"proposed_label": "drop", "reason": "invalid label"}'
                    "]}"
                ),
                elapsed_sec=0.1,
                usage={"prompt_tokens": 40, "completion_tokens": 8, "total_tokens": 48},
            )

    adapter = Mock()
    adapter.run_workflow.return_value = SimpleNamespace(dataframe=pd.DataFrame([seed_row]), failed_records=[])

    with tool.experimental_detection_strategy_context(
        tool.ExperimentalDetectionStrategy.detector_native_validate_no_augment,
        native_client=ValidationClient(),
    ):
        workflow = EntityDetectionWorkflow(adapter=adapter)
        result = workflow.detect_and_validate_entities(
            pd.DataFrame({COL_TEXT: [text]}),
            model_configs=[],
            selected_models=load_default_model_selection().detection,
            gliner_detection_threshold=0.3,
            validation_single_chunk_full_text=True,
            entity_labels=["first_name", "organization_name"],
        )

    entities = EntitiesSchema.from_raw(result.dataframe[COL_DETECTED_ENTITIES].iloc[0]).entities
    assert [(entity.label, entity.value, entity.source) for entity in entities] == [
        ("first_name", "Alice", "detector"),
    ]


def test_detector_native_validate_native_augment_uses_direct_validation_and_augmentation() -> None:
    tool = load_tool(
        "measurement_detection_strategies_detector_native_augment",
        REPO_ROOT / "tools/measurement/detection_strategies.py",
    )
    text = "Alice works at NVIDIA."
    seed_row = {
        COL_TEXT: text,
        COL_RAW_DETECTED: "",
        COL_SEED_ENTITIES: EntitiesSchema(
            entities=[
                {
                    "id": "first_name_0_5",
                    "value": "Alice",
                    "label": "first_name",
                    "start_position": 0,
                    "end_position": 5,
                    "score": 0.9,
                    "source": "detector",
                }
            ]
        ).model_dump(mode="json"),
        COL_TAG_NOTATION: "xml",
    }
    tool.prepare_validation_inputs(seed_row)

    class DirectClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete(self, request):  # type: ignore[no-untyped-def]
            self.prompts.append(request.prompt)
            if "Find additional sensitive entities" in request.prompt:
                return SimpleNamespace(
                    content='{"entities": [{"value": "NVIDIA", "label": "organization_name"}]}',
                    elapsed_sec=0.2,
                    usage={"prompt_tokens": 50, "completion_tokens": 6, "total_tokens": 56},
                )
            return SimpleNamespace(
                content='{"decisions": [{"id": "first_name_0_5", "decision": "keep", "reason": "real name"}]}',
                elapsed_sec=0.1,
                usage={"prompt_tokens": 40, "completion_tokens": 8, "total_tokens": 48},
            )

    adapter = Mock()
    adapter.run_workflow.return_value = SimpleNamespace(dataframe=pd.DataFrame([seed_row]), failed_records=[])
    client = DirectClient()
    collector = MeasurementCollector(record_hash_key="test-key")

    with measurement_session(collector):
        with tool.experimental_detection_strategy_context(
            tool.ExperimentalDetectionStrategy.detector_native_validate_native_augment,
            native_client=client,
        ):
            workflow = EntityDetectionWorkflow(adapter=adapter)
            result = workflow.detect_and_validate_entities(
                pd.DataFrame({COL_TEXT: [text]}),
                model_configs=[],
                selected_models=load_default_model_selection().detection,
                gliner_detection_threshold=0.3,
                validation_single_chunk_full_text=True,
                entity_labels=["first_name", "organization_name"],
            )

    adapter.run_workflow.assert_called_once()
    assert adapter.run_workflow.call_args.kwargs["workflow_name"] == (
        "entity-detection-detector-native-validate-native-augment-seed"
    )
    entities = EntitiesSchema.from_raw(result.dataframe[COL_DETECTED_ENTITIES].iloc[0]).entities
    assert [(entity.label, entity.value, entity.source) for entity in entities] == [
        ("first_name", "Alice", "detector"),
        ("organization_name", "NVIDIA", "augmenter"),
    ]
    assert len(client.prompts) == 2
    records = [record for record in collector.records if record["record_type"] == "model_workflow"]
    assert len(records) == 1
    assert records[0]["workflow_name"] == "entity-detection-detector-native-validate-native-augment"
    assert records[0]["observed_total_requests"] == 2
    assert records[0]["observed_total_tokens"] == 104


def test_detector_native_validate_no_augment_parallel_rows_preserve_order_and_measurements() -> None:
    tool = load_tool(
        "measurement_detection_strategies_detector_native_validate_parallel",
        REPO_ROOT / "tools/measurement/detection_strategies.py",
    )

    def seed_row(text: str, value: str) -> dict[str, object]:
        row = {
            COL_TEXT: text,
            COL_RAW_DETECTED: "",
            COL_SEED_ENTITIES: EntitiesSchema(
                entities=[
                    {
                        "id": f"first_name_0_{len(value)}",
                        "value": value,
                        "label": "first_name",
                        "start_position": 0,
                        "end_position": len(value),
                        "score": 0.9,
                        "source": "detector",
                    }
                ]
            ).model_dump(mode="json"),
            COL_TAG_NOTATION: "xml",
        }
        tool.prepare_validation_inputs(row)
        return row

    class ValidationClient:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.active_count = 0
            self.max_active_count = 0

        def complete(self, request):  # type: ignore[no-untyped-def]
            with self.lock:
                self.active_count += 1
                self.max_active_count = max(self.max_active_count, self.active_count)
            time.sleep(0.05)
            with self.lock:
                self.active_count -= 1
            candidate_id = "first_name_0_3" if '"value":"Bob"' in request.prompt else "first_name_0_5"
            return SimpleNamespace(
                content=json.dumps({"decisions": [{"id": candidate_id, "decision": "keep", "reason": "real name"}]}),
                elapsed_sec=0.1,
                usage={"prompt_tokens": 40, "completion_tokens": 8, "total_tokens": 48},
            )

    seed_df = pd.DataFrame(
        [
            seed_row("Alice works at NVIDIA.", "Alice"),
            seed_row("Bob works at NVIDIA.", "Bob"),
        ],
        index=[10, 4],
    )
    adapter = Mock()
    adapter.run_workflow.return_value = SimpleNamespace(dataframe=seed_df, failed_records=[])
    client = ValidationClient()
    collector = MeasurementCollector(record_hash_key="test-key")

    with measurement_session(collector):
        with tool.experimental_detection_strategy_context(
            tool.ExperimentalDetectionStrategy.detector_native_validate_no_augment,
            native_client=client,
        ):
            workflow = EntityDetectionWorkflow(adapter=adapter)
            result = workflow.detect_and_validate_entities(
                pd.DataFrame({COL_TEXT: ["Alice works at NVIDIA.", "Bob works at NVIDIA."]}, index=[10, 4]),
                model_configs=[],
                selected_models=load_default_model_selection().detection,
                gliner_detection_threshold=0.3,
                validation_single_chunk_full_text=True,
                entity_labels=["first_name", "organization_name"],
            )

    assert client.max_active_count > 1
    assert list(result.dataframe.index) == [10, 4]
    entities_by_row = [
        [(entity.label, entity.value, entity.source) for entity in EntitiesSchema.from_raw(raw).entities]
        for raw in result.dataframe[COL_DETECTED_ENTITIES]
    ]
    assert entities_by_row == [
        [("first_name", "Alice", "detector")],
        [("first_name", "Bob", "detector")],
    ]
    records = [record for record in collector.records if record["record_type"] == "model_workflow"]
    assert [record["observed_total_requests"] for record in records] == [1, 1]
    assert [record["observed_total_tokens"] for record in records] == [48, 48]
    record = records[0]
    assert record["workflow_name"] == "entity-detection-detector-native-validate-no-augment"
    assert record["observed_total_requests"] == 1
    assert record["observed_total_tokens"] == 48


def test_gliner_native_validate_no_augment_strategy_bypasses_data_designer() -> None:
    tool = load_tool(
        "measurement_detection_strategies_gliner_native_validate",
        REPO_ROOT / "tools/measurement/detection_strategies.py",
    )
    text = "Alice works at NVIDIA."

    class GlinerSeedClient:
        def detect(self, request):  # type: ignore[no-untyped-def]
            assert request.text == text
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "entities": [
                            {
                                "text": "Alice",
                                "label": "first_name",
                                "start": 0,
                                "end": 5,
                                "score": 0.99,
                            }
                        ]
                    }
                ),
                elapsed_sec=0.2,
                usage={"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
            )

    class ValidationClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete(self, request):  # type: ignore[no-untyped-def]
            self.prompts.append(request.prompt)
            return SimpleNamespace(
                content='{"decisions": [{"id": "first_name_0_5", "decision": "keep", "reason": "real name"}]}',
                elapsed_sec=0.1,
                usage={"prompt_tokens": 40, "completion_tokens": 8, "total_tokens": 48},
            )

    adapter = Mock()
    seed_client = GlinerSeedClient()
    validation_client = ValidationClient()
    collector = MeasurementCollector(record_hash_key="test-key")

    with measurement_session(collector):
        with tool.experimental_detection_strategy_context(
            tool.ExperimentalDetectionStrategy.gliner_native_validate_no_augment,
            native_client=validation_client,
            gliner_seed_client=seed_client,
            native_runtime=tool.NativeDetectionRuntime(
                model="test/native",
                provider="test-native-provider",
                gliner_model="test/gliner",
                gliner_provider="test-gliner-provider",
            ),
        ):
            workflow = EntityDetectionWorkflow(adapter=adapter)
            result = workflow.detect_and_validate_entities(
                pd.DataFrame({COL_TEXT: [text]}),
                model_configs=[],
                selected_models=load_default_model_selection().detection,
                gliner_detection_threshold=0.3,
                validation_single_chunk_full_text=True,
                entity_labels=["first_name", "organization_name"],
            )

    adapter.run_workflow.assert_not_called()
    entities = EntitiesSchema.from_raw(result.dataframe[COL_DETECTED_ENTITIES].iloc[0]).entities
    assert [(entity.label, entity.value, entity.source) for entity in entities] == [
        ("first_name", "Alice", "detector"),
    ]
    assert len(validation_client.prompts) == 1
    assert all("Find additional sensitive entities" not in prompt for prompt in validation_client.prompts)
    records = [record for record in collector.records if record["record_type"] == "model_workflow"]
    assert len(records) == 1
    record = records[0]
    assert record["workflow_name"] == "entity-detection-gliner-native-validate-no-augment"
    assert record["observed_total_requests"] == 2
    assert record["observed_total_tokens"] == 73
    assert sorted(record["model_usage"]) == ["gliner-direct", "native-direct"]
    assert record["model_usage"]["gliner-direct"]["model_name"] == "test/gliner"
    assert record["model_usage"]["gliner-direct"]["model_provider_name"] == "test-gliner-provider"
    assert record["model_usage"]["gliner-direct"]["token_usage"]["total_tokens"] == 25
    assert record["model_usage"]["native-direct"]["model_name"] == "test/native"
    assert record["model_usage"]["native-direct"]["model_provider_name"] == "test-native-provider"
    assert record["model_usage"]["native-direct"]["token_usage"]["total_tokens"] == 48


def test_gliner_native_validate_no_augment_parallel_rows_preserve_order_and_measurements() -> None:
    tool = load_tool(
        "measurement_detection_strategies_gliner_native_parallel",
        REPO_ROOT / "tools/measurement/detection_strategies.py",
    )

    class GlinerSeedClient:
        def detect(self, request):  # type: ignore[no-untyped-def]
            value = str(request.text).split()[0]
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "entities": [
                            {
                                "text": value,
                                "label": "first_name",
                                "start": 0,
                                "end": len(value),
                                "score": 0.99,
                            }
                        ]
                    }
                ),
                elapsed_sec=0.2,
                usage={"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
            )

    class ValidationClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete(self, request):  # type: ignore[no-untyped-def]
            self.prompts.append(request.prompt)
            candidate_id = "first_name_0_3" if '"value":"Bob"' in request.prompt else "first_name_0_5"
            return SimpleNamespace(
                content=json.dumps({"decisions": [{"id": candidate_id, "decision": "keep", "reason": "real name"}]}),
                elapsed_sec=0.1,
                usage={"prompt_tokens": 40, "completion_tokens": 8, "total_tokens": 48},
            )

    adapter = Mock()
    collector = MeasurementCollector(record_hash_key="test-key")
    dataframe = pd.DataFrame({COL_TEXT: ["Alice works at NVIDIA.", "Bob works at NVIDIA."]}, index=[10, 4])

    with measurement_session(collector):
        with tool.experimental_detection_strategy_context(
            tool.ExperimentalDetectionStrategy.gliner_native_validate_no_augment,
            native_client=ValidationClient(),
            gliner_seed_client=GlinerSeedClient(),
        ):
            workflow = EntityDetectionWorkflow(adapter=adapter)
            result = workflow.detect_and_validate_entities(
                dataframe,
                model_configs=[],
                selected_models=load_default_model_selection().detection,
                gliner_detection_threshold=0.3,
                validation_single_chunk_full_text=True,
                entity_labels=["first_name", "organization_name"],
            )

    adapter.run_workflow.assert_not_called()
    assert list(result.dataframe.index) == [10, 4]
    entities_by_row = [
        [(entity.label, entity.value, entity.source) for entity in EntitiesSchema.from_raw(raw).entities]
        for raw in result.dataframe[COL_DETECTED_ENTITIES]
    ]
    assert entities_by_row == [
        [("first_name", "Alice", "detector")],
        [("first_name", "Bob", "detector")],
    ]
    records = [record for record in collector.records if record["record_type"] == "model_workflow"]
    assert [record["observed_total_requests"] for record in records] == [2, 2]
    assert [record["observed_total_tokens"] for record in records] == [73, 73]


def test_gliner_native_validate_no_augment_parallel_rows_keep_failed_records() -> None:
    tool = load_tool(
        "measurement_detection_strategies_gliner_native_parallel_failure",
        REPO_ROOT / "tools/measurement/detection_strategies.py",
    )

    class GlinerSeedClient:
        def detect(self, request):  # type: ignore[no-untyped-def]
            if str(request.text).startswith("Broken"):
                raise RuntimeError("seed unavailable")
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "entities": [
                            {
                                "text": "Alice",
                                "label": "first_name",
                                "start": 0,
                                "end": 5,
                                "score": 0.99,
                            }
                        ]
                    }
                ),
                elapsed_sec=0.2,
                usage={"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
            )

    class ValidationClient:
        def complete(self, _request):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                content='{"decisions": [{"id": "first_name_0_5", "decision": "keep", "reason": "real name"}]}',
                elapsed_sec=0.1,
                usage={"prompt_tokens": 40, "completion_tokens": 8, "total_tokens": 48},
            )

    collector = MeasurementCollector(record_hash_key="test-key")
    dataframe = pd.DataFrame({COL_TEXT: ["Alice works at NVIDIA.", "Broken row"]}, index=[0, 1])

    with measurement_session(collector):
        with tool.experimental_detection_strategy_context(
            tool.ExperimentalDetectionStrategy.gliner_native_validate_no_augment,
            native_client=ValidationClient(),
            gliner_seed_client=GlinerSeedClient(),
        ):
            workflow = EntityDetectionWorkflow(adapter=Mock())
            result = workflow.detect_and_validate_entities(
                dataframe,
                model_configs=[],
                selected_models=load_default_model_selection().detection,
                gliner_detection_threshold=0.3,
                validation_single_chunk_full_text=True,
                entity_labels=["first_name"],
            )

    assert list(result.dataframe.index) == [0]
    entities = EntitiesSchema.from_raw(result.dataframe[COL_DETECTED_ENTITIES].iloc[0]).entities
    assert [(entity.label, entity.value) for entity in entities] == [("first_name", "Alice")]
    assert [(failed.record_id, failed.step) for failed in result.failed_records] == [
        ("1", "entity-detection-gliner-native-validate-no-augment")
    ]
    assert "seed unavailable" in result.failed_records[0].reason
    records = [record for record in collector.records if record["record_type"] == "model_workflow"]
    assert len(records) == 1
    assert records[0]["observed_total_requests"] == 2


def test_gliner_native_validate_native_augment_strategy_bypasses_data_designer() -> None:
    tool = load_tool(
        "measurement_detection_strategies_gliner_native_augment",
        REPO_ROOT / "tools/measurement/detection_strategies.py",
    )
    text = "Alice works at NVIDIA."

    class GlinerSeedClient:
        def detect(self, request):  # type: ignore[no-untyped-def]
            assert request.text == text
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "entities": [
                            {
                                "text": "Alice",
                                "label": "first_name",
                                "start": 0,
                                "end": 5,
                                "score": 0.99,
                            }
                        ]
                    }
                ),
                elapsed_sec=0.2,
                usage={"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
            )

    class NativeClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []
            self.outputs = [
                '{"decisions": [{"id": "first_name_0_5", "decision": "keep", "reason": "real name"}]}',
                '{"entities": [{"value": "NVIDIA", "label": "organization_name", "reason": "employer"}]}',
            ]

        def complete(self, request):  # type: ignore[no-untyped-def]
            self.prompts.append(request.prompt)
            return SimpleNamespace(
                content=self.outputs.pop(0),
                elapsed_sec=0.1,
                usage={"prompt_tokens": 40, "completion_tokens": 8, "total_tokens": 48},
            )

    adapter = Mock()
    seed_client = GlinerSeedClient()
    native_client = NativeClient()
    collector = MeasurementCollector(record_hash_key="test-key")

    with measurement_session(collector):
        with tool.experimental_detection_strategy_context(
            tool.ExperimentalDetectionStrategy.gliner_native_validate_native_augment,
            native_client=native_client,
            gliner_seed_client=seed_client,
        ):
            workflow = EntityDetectionWorkflow(adapter=adapter)
            result = workflow.detect_and_validate_entities(
                pd.DataFrame({COL_TEXT: [text]}),
                model_configs=[],
                selected_models=load_default_model_selection().detection,
                gliner_detection_threshold=0.3,
                validation_single_chunk_full_text=True,
                entity_labels=["first_name", "organization_name"],
            )

    adapter.run_workflow.assert_not_called()
    entities = EntitiesSchema.from_raw(result.dataframe[COL_DETECTED_ENTITIES].iloc[0]).entities
    assert [(entity.label, entity.value, entity.source) for entity in entities] == [
        ("first_name", "Alice", "detector"),
        ("organization_name", "NVIDIA", "augmenter"),
    ]
    assert any("Find additional sensitive entities" in prompt for prompt in native_client.prompts)
    records = [record for record in collector.records if record["record_type"] == "model_workflow"]
    assert len(records) == 1
    record = records[0]
    assert record["workflow_name"] == "entity-detection-gliner-native-validate-native-augment"
    assert record["observed_total_requests"] == 3
    assert record["observed_total_tokens"] == 121
    assert sorted(record["model_usage"]) == ["gliner-direct", "native-direct"]
    assert record["model_usage"]["gliner-direct"]["token_usage"]["total_tokens"] == 25
    assert record["model_usage"]["native-direct"]["token_usage"]["total_tokens"] == 96


def test_native_single_pass_strategy_runs_one_direct_call_without_data_designer() -> None:
    tool = load_tool(
        "measurement_detection_strategies_native_single_pass",
        REPO_ROOT / "tools/measurement/detection_strategies.py",
    )

    class SinglePassClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete(self, request):  # type: ignore[no-untyped-def]
            self.prompts.append(request.prompt)
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "entities": [
                            {"value": "Alice", "label": "first_name", "start": 0, "end": 5},
                            {"value": "NVIDIA", "label": "organization_name", "start": 15, "end": 21},
                        ]
                    }
                ),
                elapsed_sec=0.1,
                usage={},
            )

    adapter = Mock()
    client = SinglePassClient()

    with tool.experimental_detection_strategy_context(
        tool.ExperimentalDetectionStrategy.native_single_pass,
        native_client=client,
    ):
        workflow = EntityDetectionWorkflow(adapter=adapter)
        result = workflow.detect_and_validate_entities(
            pd.DataFrame({COL_TEXT: ["Alice works at NVIDIA."]}),
            model_configs=[],
            selected_models=load_default_model_selection().detection,
            gliner_detection_threshold=0.3,
            entity_labels=["first_name", "organization_name"],
        )

    adapter.run_workflow.assert_not_called()
    entities = EntitiesSchema.from_raw(result.dataframe[COL_DETECTED_ENTITIES].iloc[0]).entities
    assert [(entity.label, entity.value, entity.source) for entity in entities] == [
        ("first_name", "Alice", "direct_single_pass"),
        ("organization_name", "NVIDIA", "direct_single_pass"),
    ]
    assert len(client.prompts) == 1
    assert '"start"' in client.prompts[0]
    assert '"end"' in client.prompts[0]


def test_native_single_pass_recall_strategy_uses_label_examples() -> None:
    tool = load_tool(
        "measurement_detection_strategies_native_single_pass_recall",
        REPO_ROOT / "tools/measurement/detection_strategies.py",
    )

    class SinglePassClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete(self, request):  # type: ignore[no-untyped-def]
            self.prompts.append(request.prompt)
            return SimpleNamespace(
                content='{"entities": [{"value": "Alice", "label": "person", "start": 0, "end": 5}]}',
                elapsed_sec=0.1,
                usage={},
            )

    client = SinglePassClient()

    with tool.experimental_detection_strategy_context(
        tool.ExperimentalDetectionStrategy.native_single_pass_recall,
        native_client=client,
    ):
        workflow = EntityDetectionWorkflow(adapter=Mock())
        workflow.detect_and_validate_entities(
            pd.DataFrame({COL_TEXT: ["Alice works at NVIDIA."]}),
            model_configs=[],
            selected_models=load_default_model_selection().detection,
            gliner_detection_threshold=0.3,
            entity_labels=["person", "email"],
        )

    assert len(client.prompts) == 1
    assert "- person" in client.prompts[0]
    assert "- email:" in client.prompts[0]
    assert "Bias toward high recall" in client.prompts[0]


def test_native_single_pass_values_strategy_uses_value_only_prompt() -> None:
    tool = load_tool(
        "measurement_detection_strategies_native_single_pass_values",
        REPO_ROOT / "tools/measurement/detection_strategies.py",
    )

    class SinglePassClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete(self, request):  # type: ignore[no-untyped-def]
            self.prompts.append(request.prompt)
            return SimpleNamespace(
                content='{"entities": [{"value": "Alice", "label": "person"}]}',
                elapsed_sec=0.1,
                usage={},
            )

    client = SinglePassClient()

    with tool.experimental_detection_strategy_context(
        tool.ExperimentalDetectionStrategy.native_single_pass_values,
        native_client=client,
    ):
        workflow = EntityDetectionWorkflow(adapter=Mock())
        result = workflow.detect_and_validate_entities(
            pd.DataFrame({COL_TEXT: ["Alice met Alice."]}),
            model_configs=[],
            selected_models=load_default_model_selection().detection,
            gliner_detection_threshold=0.3,
            entity_labels=["person"],
        )

    entities = EntitiesSchema.from_raw(result.dataframe[COL_DETECTED_ENTITIES].iloc[0]).entities
    assert [(entity.value, entity.start_position, entity.end_position) for entity in entities] == [
        ("Alice", 0, 5),
        ("Alice", 10, 15),
    ]
    assert len(client.prompts) == 1
    assert '"start"' not in client.prompts[0]
    assert '"end"' not in client.prompts[0]
    assert '{"entities": [{"value": "exact substring", "label": "one_allowed_label"' in client.prompts[0]


def test_native_single_pass_strategy_records_direct_model_usage() -> None:
    tool = load_tool(
        "measurement_detection_strategies_native_single_pass_usage",
        REPO_ROOT / "tools/measurement/detection_strategies.py",
    )

    class SinglePassClient:
        def complete(self, _request):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                content='{"entities": [{"value": "Alice", "label": "first_name", "start": 0, "end": 5}]}',
                elapsed_sec=0.1,
                usage={"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27},
            )

    collector = MeasurementCollector(record_hash_key="test-key")

    with measurement_session(collector):
        with tool.experimental_detection_strategy_context(
            tool.ExperimentalDetectionStrategy.native_single_pass,
            native_client=SinglePassClient(),
        ):
            workflow = EntityDetectionWorkflow(adapter=Mock())
            workflow.detect_and_validate_entities(
                pd.DataFrame({COL_TEXT: ["Alice works at NVIDIA."]}),
                model_configs=[],
                selected_models=load_default_model_selection().detection,
                gliner_detection_threshold=0.3,
                entity_labels=["first_name"],
            )

    records = [record for record in collector.records if record["record_type"] == "model_workflow"]
    assert len(records) == 1
    record = records[0]
    assert record["workflow_name"] == "entity-detection-native-single-pass"
    assert record["observed_total_requests"] == 1
    assert record["observed_successful_requests"] == 1
    assert record["observed_failed_requests"] == 0
    assert record["observed_input_tokens"] == 20
    assert record["observed_output_tokens"] == 7
    assert record["observed_total_tokens"] == 27


def test_native_single_pass_strategy_uses_only_native_spans() -> None:
    tool = load_tool(
        "measurement_detection_strategies_native_single_pass_native_spans",
        REPO_ROOT / "tools/measurement/detection_strategies.py",
    )
    text = "Alice logged in.\nPassword: SuperSecret123!\n"

    class SinglePassClient:
        def complete(self, _request):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                content='{"entities": [{"value": "Alice", "label": "person", "start": 0, "end": 5}]}',
                elapsed_sec=0.1,
                usage={},
            )

    with tool.experimental_detection_strategy_context(
        tool.ExperimentalDetectionStrategy.native_single_pass,
        native_client=SinglePassClient(),
    ):
        workflow = EntityDetectionWorkflow(adapter=Mock())
        result = workflow.detect_and_validate_entities(
            pd.DataFrame({COL_TEXT: [text]}),
            model_configs=[],
            selected_models=load_default_model_selection().detection,
            gliner_detection_threshold=0.3,
            entity_labels=["person", "password"],
        )

    entities = EntitiesSchema.from_raw(result.dataframe[COL_DETECTED_ENTITIES].iloc[0]).entities
    assert [(entity.label, entity.value, entity.source) for entity in entities] == [
        ("person", "Alice", "direct_single_pass"),
    ]


def test_native_single_pass_strategy_records_parser_errors_as_failures() -> None:
    tool = load_tool(
        "measurement_detection_strategies_native_single_pass_parser_error",
        REPO_ROOT / "tools/measurement/detection_strategies.py",
    )

    class InvalidJsonClient:
        def complete(self, _request):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                content="not json",
                elapsed_sec=0.1,
                usage={"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
            )

    collector = MeasurementCollector(record_hash_key="test-key")

    with measurement_session(collector):
        with tool.experimental_detection_strategy_context(
            tool.ExperimentalDetectionStrategy.native_single_pass,
            native_client=InvalidJsonClient(),
        ):
            workflow = EntityDetectionWorkflow(adapter=Mock())
            result = workflow.detect_and_validate_entities(
                pd.DataFrame({COL_TEXT: ["Alice works at NVIDIA."]}),
                model_configs=[],
                selected_models=load_default_model_selection().detection,
                gliner_detection_threshold=0.3,
                entity_labels=["first_name"],
            )

    assert result.dataframe.empty
    assert len(result.failed_records) == 1
    assert result.failed_records[0].step == "entity-detection-native-single-pass"
    records = [record for record in collector.records if record["record_type"] == "model_workflow"]
    assert len(records) == 1
    record = records[0]
    assert record["workflow_name"] == "entity-detection-native-single-pass"
    assert record["status"] == "error"
    assert record["failed_record_count"] == 1
    assert record["observed_successful_requests"] == 1
    assert record["observed_failed_requests"] == 0


def test_detector_only_strategy_finalizes_gliner_spans_without_validation_or_augmentation() -> None:
    tool = load_tool(
        "measurement_detection_strategies_detector_only",
        REPO_ROOT / "tools/measurement/detection_strategies.py",
    )
    adapter = Mock()
    text = "Alice met Alice at the lab."
    start = text.index("Alice")

    def fake_run_workflow(dataframe: pd.DataFrame, *, columns: list, **kwargs: object) -> SimpleNamespace:
        assert [column.name for column in columns] == [
            COL_RAW_DETECTED,
            COL_SEED_ENTITIES,
            COL_SEED_ENTITIES_JSON,
            COL_DETECTED_ENTITIES,
        ]
        assert kwargs["workflow_name"] == "entity-detection-detector-only"
        row = {
            COL_TEXT: dataframe[COL_TEXT].iloc[0],
            COL_RAW_DETECTED: json.dumps(
                {
                    "entities": [
                        {
                            "text": "Alice",
                            "label": "person",
                            "start": start,
                            "end": start + len("Alice"),
                            "score": 0.99,
                        }
                    ]
                }
            ),
        }
        row = columns[1].generator_function(row)
        row = columns[2].generator_function(row)
        seed_entities = json.loads(row[COL_SEED_ENTITIES_JSON])
        assert [(entity["label"], entity["value"]) for entity in seed_entities] == [("person", "Alice")]
        row = columns[3].generator_function(row)
        return SimpleNamespace(dataframe=pd.DataFrame([row]), failed_records=[])

    adapter.run_workflow.side_effect = fake_run_workflow

    with tool.experimental_detection_strategy_context(tool.ExperimentalDetectionStrategy.detector_only):
        workflow = EntityDetectionWorkflow(adapter=adapter)
        result = workflow.detect_and_validate_entities(
            pd.DataFrame({COL_TEXT: [text]}),
            model_configs=[],
            selected_models=load_default_model_selection().detection,
            gliner_detection_threshold=0.3,
            entity_labels=["person"],
        )

    entities = EntitiesSchema.from_raw(result.dataframe[COL_DETECTED_ENTITIES].iloc[0]).entities
    assert [(entity.label, entity.value) for entity in entities] == [("person", "Alice"), ("person", "Alice")]
    assert "<person>Alice</person>" in result.dataframe[COL_TAGGED_TEXT].iloc[0]


def test_compact_validation_strategy_disables_full_text_single_chunk_validation() -> None:
    tool = load_tool(
        "measurement_detection_strategies_compact_validation",
        REPO_ROOT / "tools/measurement/detection_strategies.py",
    )
    original = EntityDetectionWorkflow.detect_and_validate_entities
    calls = []

    def fake_original(
        self: EntityDetectionWorkflow,
        dataframe: pd.DataFrame,
        **kwargs: object,
    ) -> object:
        calls.append(kwargs)
        return SimpleNamespace(
            dataframe=pd.DataFrame(
                [
                    {
                        COL_TEXT: dataframe[COL_TEXT].iloc[0],
                        COL_DETECTED_ENTITIES: EntitiesSchema(entities=[]).model_dump(mode="json"),
                    }
                ]
            ),
            failed_records=[],
        )

    EntityDetectionWorkflow.detect_and_validate_entities = fake_original  # type: ignore[method-assign]
    try:
        with tool.experimental_detection_strategy_context(tool.ExperimentalDetectionStrategy.compact_validation):
            workflow = EntityDetectionWorkflow(adapter=Mock())
            workflow.detect_and_validate_entities(
                pd.DataFrame({COL_TEXT: ["Alice works at Acme."]}),
                model_configs=[],
                selected_models=load_default_model_selection().detection,
                gliner_detection_threshold=0.3,
                entity_labels=["first_name"],
            )
    finally:
        EntityDetectionWorkflow.detect_and_validate_entities = original  # type: ignore[method-assign]

    assert EntityDetectionWorkflow.detect_and_validate_entities is original
    assert calls[0]["validation_single_chunk_full_text"] is False


def test_prose_augment_focus_extends_and_restores_augment_prompt() -> None:
    tool = load_tool(
        "measurement_detection_strategies_prose_augment_focus",
        REPO_ROOT / "tools/measurement/detection_strategies.py",
    )
    before = tool.dw._get_augment_prompt(data_summary=None, labels=["organization_name"], strict_labels=True)

    with tool.experimental_detection_strategy_context(tool.ExperimentalDetectionStrategy.prose_augment_focus):
        inside = tool.dw._get_augment_prompt(data_summary=None, labels=["organization_name"], strict_labels=True)

    after = tool.dw._get_augment_prompt(data_summary=None, labels=["organization_name"], strict_labels=True)
    assert "Contextual prose recall focus" not in before
    assert "Contextual prose recall focus" in inside
    assert "organization and institution names" in inside
    assert after == before
