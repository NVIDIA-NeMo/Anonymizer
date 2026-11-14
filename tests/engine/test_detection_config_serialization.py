# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.config.seed import PartitionBlock, SamplingStrategy
from data_designer.config.seed_source import LocalFileSeedSource
from data_designer.engine.testing.utils import assert_valid_plugin
from data_designer.interface.data_designer import DataDesigner
from data_designer.plugins import Plugin

from anonymizer.config.models import DetectionModelSelection
from anonymizer.config.regex import BuiltinRegex, RegexCandidate, RegexRule
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import (
    COL_ADMISSION_RESULTS,
    COL_ADMITTED_CANDIDATES,
    COL_AUGMENTED_CANDIDATES,
    COL_AUGMENTED_ENTITIES,
    COL_DETECTED_ENTITIES,
    COL_FINAL_ENTITIES,
    COL_RAW_DETECTED,
    COL_RECONCILED_CANDIDATES,
    COL_REGEX_ACCEPTED_ENTITIES,
    COL_REGEX_DETECTION_RESULTS,
    COL_REGEX_VALIDATION_TRACE,
    COL_REPLACED_TEXT,
    COL_SEED_ENTITIES,
    COL_TEXT,
    COL_VALIDATION_DECISIONS,
    DEFAULT_ENTITY_LABELS,
    RECORD_ID_COLUMN,
)
from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.ndd.model_loader import parse_model_configs
from anonymizer.engine.replace.strategies import apply_local_replace_strategy
from anonymizer.engine.workflow_columns.detection.config import (
    AugmentationConfig,
    ChunkedValidationConfig,
    DetectionTransformConfig,
    DetectionTransformOperation,
    RegexDetectionConfig,
)
from anonymizer.engine.workflow_columns.detection.impl import AugmentationGenerator
from anonymizer.engine.workflow_columns.detection.plugins import (
    augmentation_plugin,
    chunked_validation_plugin,
    detection_transform_plugin,
    regex_detection_plugin,
)


@pytest.mark.parametrize(
    "plugin", [detection_transform_plugin, chunked_validation_plugin, augmentation_plugin, regex_detection_plugin]
)
def test_detection_plugin_satisfies_data_designer_contract(plugin: Plugin) -> None:
    assert_valid_plugin(plugin)


def test_regex_constrained_augmentation_retains_async_only_entrypoint() -> None:
    config = AugmentationConfig(
        name=COL_AUGMENTED_ENTITIES,
        model_alias="stub",
        prompt_template="Synthetic fixture",
        enabled=False,
        regex_constrained_entity_labels=["url"],
    )
    assert config.required_columns == [RECORD_ID_COLUMN, COL_TEXT, COL_ADMITTED_CANDIDATES]
    generator = AugmentationGenerator(config, resource_provider=Mock())
    with pytest.raises(NotImplementedError, match="async-only; use agenerate"):
        generator.generate({COL_TEXT: "url:https://example.org/private"})


def test_regex_validation_plugin_dependencies_do_not_require_downstream_results() -> None:
    config = ChunkedValidationConfig(
        name=COL_VALIDATION_DECISIONS,
        pool=["stub"],
        max_entities_per_call=5,
        excerpt_window_chars=100,
        prompt_template="Synthetic fixture",
    )
    assert COL_REGEX_DETECTION_RESULTS in config.required_columns
    assert COL_RECONCILED_CANDIDATES not in config.required_columns
    assert COL_AUGMENTED_CANDIDATES not in config.required_columns


def test_detection_builder_round_trips_through_native_data_designer_config(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.parquet"
    pd.DataFrame({COL_TEXT: ["Alice", "Bob", "Carol"]}).to_parquet(seed_path, index=False)

    parsed_models = parse_model_configs(None)
    workflow = EntityDetectionWorkflow(adapter=NddAdapter(data_designer=cast(DataDesigner, Mock())))
    builder = workflow.build_detection_builder_for_seed(
        seed_path=seed_path,
        model_configs=parsed_models.model_configs,
        selected_models=parsed_models.selected_models.detection,
        gliner_detection_threshold=0.42,
        validation_max_entities_per_call=7,
        validation_excerpt_window_chars=321,
        entity_labels=["first_name", "email"],
        excluded_entity_labels=["email"],
        entity_label_examples={
            "first_name": ["Alicia"],
            "email": ["configured@example.test"],
        },
        data_summary="Customer support messages",
        job_index=1,
        num_jobs=3,
    )

    direct_augmenter = next(column for column in builder.get_column_configs() if column.name == COL_AUGMENTED_ENTITIES)
    assert isinstance(direct_augmenter, AugmentationConfig)

    payload = builder.get_builder_config().to_json()
    assert payload is not None
    restored = DataDesignerConfigBuilder.from_config(payload)

    assert restored.get_builder_config().to_dict() == builder.get_builder_config().to_dict()

    seed_config = restored.get_seed_config()
    assert seed_config is not None
    assert isinstance(seed_config.source, LocalFileSeedSource)
    assert seed_config.source.path == str(seed_path)
    assert seed_config.sampling_strategy == SamplingStrategy.ORDERED
    assert seed_config.selection_strategy == PartitionBlock(index=1, num_partitions=3)

    columns = restored.get_column_configs()
    assert len(columns) == 11
    regex = next(column for column in columns if isinstance(column, RegexDetectionConfig))
    assert regex.rules == []
    assert regex.side_effect_columns == [
        COL_REGEX_ACCEPTED_ENTITIES,
        COL_REGEX_VALIDATION_TRACE,
        COL_REGEX_DETECTION_RESULTS,
    ]
    assert all(column.column_type != "custom" for column in columns)

    transforms = [column for column in columns if isinstance(column, DetectionTransformConfig)]
    assert {DetectionTransformOperation(column.operation) for column in transforms} == set(DetectionTransformOperation)
    merge_transform = next(
        column
        for column in transforms
        if DetectionTransformOperation(column.operation) == DetectionTransformOperation.MERGE_AND_BUILD_CANDIDATES
    )
    seed_validation_transform = next(
        column
        for column in transforms
        if DetectionTransformOperation(column.operation)
        == DetectionTransformOperation.APPLY_VALIDATION_TO_SEED_ENTITIES
    )
    finalize_transform = next(
        column
        for column in transforms
        if DetectionTransformOperation(column.operation) == DetectionTransformOperation.FINALIZE_ENTITIES
    )
    assert seed_validation_transform.excluded_entity_labels == []
    assert merge_transform.excluded_entity_labels == []
    assert merge_transform.required_columns == [COL_TEXT, COL_ADMITTED_CANDIDATES, COL_AUGMENTED_CANDIDATES]
    assert finalize_transform.excluded_entity_labels == []
    assert finalize_transform.allowed_entity_labels is None
    assert finalize_transform.required_columns == [COL_TEXT, COL_RECONCILED_CANDIDATES]

    validation = next(column for column in columns if column.name == COL_VALIDATION_DECISIONS)
    assert isinstance(validation, ChunkedValidationConfig)
    assert validation.max_entities_per_call == 7
    assert validation.excerpt_window_chars == 321
    admission = next(column for column in columns if column.name == COL_ADMISSION_RESULTS)
    assert isinstance(admission, DetectionTransformConfig)
    assert admission.allowed_entity_labels == validation.allowed_entity_labels == ["first_name"]
    assert admission.excluded_entity_labels == validation.excluded_entity_labels == ["email"]
    assert validation.pool == parsed_models.selected_models.detection.entity_validator
    assert "Customer support messages" in validation.prompt_template
    assert "Alicia" in validation.prompt_template
    assert "configured@example.test" not in validation.prompt_template

    augmenter = next(column for column in columns if column.name == COL_AUGMENTED_ENTITIES)
    assert isinstance(augmenter, AugmentationConfig)
    assert "first_name: Alicia" in augmenter.prompt_template
    assert "Michael, Isabella, Carlos, Wei" not in augmenter.prompt_template
    assert "configured@example.test" not in augmenter.prompt_template

    serialized = json.loads(payload)
    serialized_text = json.dumps(serialized)
    assert "anonymizer-detection-transform" in serialized_text
    assert "anonymizer-chunked-validation" in serialized_text
    assert "anonymizer-augmentation" in serialized_text
    assert "generator_function" not in serialized_text
    assert "generator_params" not in serialized_text


def _get_gliner_labels_from_builder(builder: DataDesignerConfigBuilder) -> list[str]:
    payload = builder.get_builder_config().to_json()
    assert payload is not None
    serialized = json.loads(payload)
    model_configs = serialized["data_designer"]["model_configs"]
    gliner = next(m for m in model_configs if m.get("alias") == "gliner-pii-detector")
    return gliner["inference_parameters"]["extra_body"]["labels"]


def test_build_detection_builder_for_seed_respects_excluded_entity_labels(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.parquet"
    pd.DataFrame({COL_TEXT: ["Alice"]}).to_parquet(seed_path, index=False)

    parsed_models = parse_model_configs(None)
    workflow = EntityDetectionWorkflow(adapter=NddAdapter(data_designer=cast(DataDesigner, Mock())))
    builder = workflow.build_detection_builder_for_seed(
        seed_path=seed_path,
        model_configs=parsed_models.model_configs,
        selected_models=parsed_models.selected_models.detection,
        gliner_detection_threshold=0.3,
        entity_labels=["first_name", "email", "city"],
        excluded_entity_labels=["email"],
    )

    labels = _get_gliner_labels_from_builder(builder)
    assert "email" not in labels
    assert "first_name" in labels
    assert "city" in labels


def test_build_detection_config_respects_excluded_entity_labels(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.parquet"
    input_df = pd.DataFrame({COL_TEXT: ["Alice"]})
    input_df.to_parquet(seed_path, index=False)

    parsed_models = parse_model_configs(None)
    workflow = EntityDetectionWorkflow(adapter=NddAdapter(data_designer=cast(DataDesigner, Mock())))
    builder = workflow.build_detection_config(
        input_df,
        seed_path=seed_path,
        model_configs=parsed_models.model_configs,
        selected_models=parsed_models.selected_models.detection,
        gliner_detection_threshold=0.3,
        entity_labels=["first_name", "email", "city"],
        excluded_entity_labels=["email"],
    )

    augmenter = next(column for column in builder.get_column_configs() if column.name == COL_AUGMENTED_ENTITIES)
    assert isinstance(augmenter, AugmentationConfig)
    labels = _get_gliner_labels_from_builder(builder)
    assert "email" not in labels
    assert "first_name" in labels
    assert "city" in labels


def test_exported_builder_includes_explicit_non_default_example_label(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.parquet"
    pd.DataFrame({COL_TEXT: ["Credential acme_live_abc123"]}).to_parquet(seed_path, index=False)

    parsed_models = parse_model_configs(None)
    workflow = EntityDetectionWorkflow(adapter=NddAdapter(data_designer=cast(DataDesigner, Mock())))
    builder = workflow.build_detection_builder_for_seed(
        seed_path=seed_path,
        model_configs=parsed_models.model_configs,
        selected_models=parsed_models.selected_models.detection,
        gliner_detection_threshold=0.3,
        entity_labels=[*DEFAULT_ENTITY_LABELS, "vendor_api_key"],
        entity_label_examples={"vendor_api_key": ["acme_live_abc123"]},
    )

    assert "vendor_api_key" in _get_gliner_labels_from_builder(builder)
    columns = builder.get_column_configs()
    validation = next(column for column in columns if column.name == COL_VALIDATION_DECISIONS)
    augmenter = next(column for column in columns if column.name == COL_AUGMENTED_ENTITIES)
    assert isinstance(validation, ChunkedValidationConfig)
    assert isinstance(augmenter, AugmentationConfig)
    finalize = next(
        column
        for column in columns
        if isinstance(column, DetectionTransformConfig)
        and column.operation == DetectionTransformOperation.FINALIZE_ENTITIES
    )
    assert "- vendor_api_key: acme_live_abc123" in validation.prompt_template
    assert "vendor_api_key: acme_live_abc123" in augmenter.prompt_template
    assert "Use ONLY labels from this list" in augmenter.prompt_template
    assert finalize.allowed_entity_labels is None
    assert validation.allowed_entity_labels == [*DEFAULT_ENTITY_LABELS, "vendor_api_key"]


def test_fresh_process_discovers_plugins_when_loading_native_config(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.parquet"
    pd.DataFrame({COL_TEXT: ["Alice"]}).to_parquet(seed_path, index=False)

    parsed_models = parse_model_configs(None)
    workflow = EntityDetectionWorkflow(adapter=NddAdapter(data_designer=cast(DataDesigner, Mock())))
    builder = workflow.build_detection_builder_for_seed(
        seed_path=seed_path,
        model_configs=parsed_models.model_configs,
        selected_models=parsed_models.selected_models.detection,
        gliner_detection_threshold=0.3,
    )
    config_path = tmp_path / "detection.json"
    output_path = tmp_path / "restored-columns.json"
    builder.get_builder_config().to_json(config_path)

    script = """
import json
import sys
from pathlib import Path

from data_designer.config.config_builder import DataDesignerConfigBuilder

builder = DataDesignerConfigBuilder.from_config(Path(sys.argv[1]))
columns = [
    {
        "name": column.name,
        "column_type": column.column_type,
        "class_name": type(column).__name__,
    }
    for column in builder.get_column_configs()
]
Path(sys.argv[2]).write_text(json.dumps(columns))
"""
    subprocess.run(
        [sys.executable, "-c", script, str(config_path), str(output_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    restored_columns = json.loads(output_path.read_text())
    restored_types = {(column["column_type"], column["class_name"]) for column in restored_columns}
    assert ("anonymizer-detection-transform", "DetectionTransformConfig") in restored_types
    assert ("anonymizer-chunked-validation", "ChunkedValidationConfig") in restored_types
    assert ("anonymizer-augmentation", "AugmentationConfig") in restored_types
    assert ("anonymizer-regex-detection", "RegexDetectionConfig") in restored_types


@pytest.mark.parametrize("validate_matches", [False, True])
@pytest.mark.parametrize("additional_matches", [False, True])
def test_regex_rule_controls_round_trip_independently(
    tmp_path: Path, validate_matches: bool, additional_matches: bool
) -> None:
    pd.DataFrame({COL_TEXT: ["TKT-123"]}).to_parquet(tmp_path / "seed.parquet", index=False)
    parsed_models = parse_model_configs(None)
    workflow = EntityDetectionWorkflow(adapter=NddAdapter(data_designer=cast(DataDesigner, Mock())))
    builder = workflow.build_detection_builder_for_seed(
        seed_path=tmp_path / "seed.parquet",
        model_configs=parsed_models.model_configs,
        selected_models=parsed_models.selected_models.detection,
        gliner_detection_threshold=0.3,
        entity_labels=["ticket"],
        builtin_regexes=False,
        regex_rules=[
            RegexRule(
                label="ticket",
                pattern=r"TKT-\d+",
                validate_matches_with_llm=validate_matches,
                detect_additional_matches=additional_matches,
            )
        ],
    )
    payload = builder.get_builder_config().to_json()
    assert payload is not None
    restored = DataDesignerConfigBuilder.from_config(payload)
    columns = restored.get_column_configs()
    regex = next(column for column in columns if isinstance(column, RegexDetectionConfig))
    augmenter = next(column for column in columns if isinstance(column, AugmentationConfig))
    admission = next(
        column
        for column in columns
        if isinstance(column, DetectionTransformConfig)
        and column.operation == DetectionTransformOperation.ADMIT_VALIDATION_DECISIONS
    )
    assert regex.rules[0].validate_matches_with_llm is validate_matches
    assert regex.rules[0].detect_additional_matches is additional_matches
    assert _get_gliner_labels_from_builder(restored) == (["ticket"] if additional_matches else [])
    assert augmenter.enabled is additional_matches
    assert (
        augmenter.regex_constrained_entity_labels
        == admission.regex_constrained_entity_labels
        == ([] if additional_matches else ["ticket"])
    )


def test_regex_constrained_detection_config_round_trips_with_model_columns_skipped(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.parquet"
    pd.DataFrame({COL_TEXT: ["TKT-123"]}).to_parquet(seed_path, index=False)

    parsed_models = parse_model_configs(None)
    workflow = EntityDetectionWorkflow(adapter=NddAdapter(data_designer=cast(DataDesigner, Mock())))
    builder = workflow.build_detection_builder_for_seed(
        seed_path=seed_path,
        model_configs=parsed_models.model_configs,
        selected_models=parsed_models.selected_models.detection,
        gliner_detection_threshold=0.3,
        entity_labels=["ticket"],
        regex_rules=[
            RegexRule(
                label="ticket",
                pattern=r"TKT-\d+",
                validate_matches_with_llm=False,
                detect_additional_matches=False,
            )
        ],
    )

    payload = builder.get_builder_config().to_json()
    assert payload is not None
    restored = DataDesignerConfigBuilder.from_config(payload)
    columns = restored.get_column_configs()

    assert _get_gliner_labels_from_builder(builder) == []
    assert next(column for column in columns if column.name == COL_RAW_DETECTED).skip is not None
    augmenter = next(column for column in columns if column.name == COL_AUGMENTED_ENTITIES)
    assert isinstance(augmenter, AugmentationConfig)
    assert augmenter.enabled is False
    assert next(column for column in columns if column.name == COL_SEED_ENTITIES).propagate_skip is False


@pytest.mark.parametrize("preview_num_records", [2, None], ids=["preview", "create"])
@pytest.mark.parametrize(
    "workload",
    [
        "custom-ticket",
        "builtin-card-with-code",
        "all-builtins-100-rows",
        "network-urls",
        "colon-url",
        "builtin-invalid",
        "unsupported-urls",
    ],
)
def test_regex_constrained_matches_survive_skipped_model_columns_in_data_designer(
    tmp_path: Path,
    preview_num_records: int | None,
    workload: str,
) -> None:
    card_with_code = workload == "builtin-card-with-code"
    value = "4111 1111 1111 1111" if card_with_code else "TKT-123"
    label = "credit_debit_card" if card_with_code else "ticket"
    suffix = " 123" if card_with_code else ""
    rule = (
        BuiltinRegex(label=label, validate_matches_with_llm=False, detect_additional_matches=False)
        if card_with_code
        else RegexRule(
            label=label, pattern=r"TKT-\d+", validate_matches_with_llm=False, detect_additional_matches=False
        )
    )
    input_df = pd.DataFrame({COL_TEXT: [value + suffix, "nothing"]})
    values = [(value, label)]
    rules: list[BuiltinRegex | RegexRule] = [rule]
    labels = [label]
    if workload == "colon-url":
        value = "https://example.org/private/alice"
        label = "url"
        values = [(value, label)]
        labels = [label]
        rules = [BuiltinRegex(label=label, validate_matches_with_llm=False, detect_additional_matches=False)]
        input_df = pd.DataFrame({COL_TEXT: [f"url:{value}.", f"url: {value}.", f"URL:{value}."]})
    if workload in {"builtin-invalid", "unsupported-urls"}:
        labels = ["credit_debit_card", "email", "ipv4", "ipv6", "mac_address", "url"]
        values = []
        rules = [
            BuiltinRegex(label=label, validate_matches_with_llm=False, detect_additional_matches=False)
            for label in labels
        ]
        text = " | ".join(
            [
                "4111 1111 1111 1112",
                "a" * 65 + "@example.org",
                "256.0.2.10",
                "2001::db8::1",
                "00:1A-2B:3C:4D:5E",
                "https://example.org:65536/private",
            ]
        )
        if workload == "unsupported-urls":
            labels = ["url"]
            rules = [BuiltinRegex(label="url", validate_matches_with_llm=False, detect_additional_matches=False)]
            text = " | ".join(
                [
                    "example.org/private",
                    "//example.org/private",
                    "file:///private/report.csv",
                    "s3://www.example.org/private",
                    "git+https://example.org/private",
                    r"https:\/\/example.org/private",
                ]
            )
        input_df = pd.DataFrame({COL_TEXT: [text, "nothing"]})
    if workload in {"all-builtins-100-rows", "network-urls"}:
        values = [
            ("4111 1111 1111 1111", "credit_debit_card"),
            ("user@example.org", "email"),
            ("192.0.2.10", "ipv4"),
            ("2001:db8::1", "ipv6"),
            ("00:1A:2B:3C:4D:5E", "mac_address"),
            ("https://example.org/path", "url"),
        ]
        if workload == "network-urls":
            values = [
                ("http://localhost:8080/private", "url"),
                ("https://buildbox/private", "url"),
                ("ftp://user:password@example.org:21/report.csv", "url"),
                ("ftps://例子.公司/路径", "url"),
                ("sftp://internal-host:2222/private", "url"),
                ("ws://192.0.2.10:8080/session", "url"),
                ("wss://[2001:db8::1]/session", "url"),
            ]
        labels = sorted({label for _, label in values})
        rules = [
            BuiltinRegex(label=label, validate_matches_with_llm=False, detect_additional_matches=False)
            for label in labels
        ]
        text = "tool output: completed normally; " * 300 + " | ".join(value for value, _ in values)
        records = 100 if workload == "all-builtins-100-rows" else 2
        input_df = pd.DataFrame({COL_TEXT: [f"Record {index}: {text}" for index in range(records)]})
    model_configs = [
        ModelConfig(
            alias="known",
            model="stub-model",
            provider="stub",
            skip_health_check=True,
        )
    ]
    selected_models = DetectionModelSelection(
        entity_detector="known",
        entity_validator="known",
        entity_augmenter="known",
        latent_detector="known",
    )
    build_workflow = EntityDetectionWorkflow(adapter=NddAdapter(data_designer=cast(DataDesigner, Mock())))
    builder = build_workflow.build_detection_config(
        input_df,
        seed_path=tmp_path / "seed.parquet",
        model_configs=model_configs,
        selected_models=selected_models,
        gliner_detection_threshold=0.3,
        entity_labels=labels,
        builtin_regexes=workload != "custom-ticket",
        regex_rules=rules,
    )
    payload = builder.get_builder_config().to_json()
    assert payload is not None
    restored = DataDesignerConfigBuilder.from_config(payload)

    provider = ModelProvider(
        name="stub",
        endpoint="http://127.0.0.1:9/v1",
        provider_type="openai",
        api_key="EMPTY",
    )
    data_designer = DataDesigner(
        artifact_path=tmp_path / "artifacts",
        model_providers=[provider],
        auto_configure_logging=False,
    )
    adapter = NddAdapter(data_designer=data_designer)
    with (
        patch("httpx.Client.send", side_effect=AssertionError("unexpected synchronous model request")),
        patch("httpx.AsyncClient.send", side_effect=AssertionError("unexpected asynchronous model request")),
    ):
        result = adapter.run_workflow(
            input_df,
            model_configs=restored.model_configs,
            columns=restored.get_column_configs(),
            workflow_name=f"regex-constrained-{'preview' if preview_num_records is not None else 'create'}",
            preview_num_records=len(input_df) if preview_num_records is not None else None,
        )

    assert result.failed_records == []
    assert len(result.dataframe) == len(input_df)
    detected = result.dataframe[COL_DETECTED_ENTITIES].tolist()

    replacement_input = result.dataframe[[COL_TEXT]].copy()
    replacement_input[COL_FINAL_ENTITIES] = detected
    replaced = apply_local_replace_strategy(replacement_input, strategy=Redact())
    expected_replaced = []
    for text in input_df[COL_TEXT]:
        for value, label in values:
            text = text.replace(value, f"[REDACTED_{label.upper()}]")
        expected_replaced.append(text)
    assert replaced[COL_REPLACED_TEXT].tolist() == expected_replaced

    assert [
        [
            (
                entity["value"],
                entity["label"],
                entity["start_position"],
                entity["end_position"],
            )
            for entity in detection["entities"]
        ]
        for detection in detected
    ] == [
        [(value, label, text.index(value), text.index(value) + len(value)) for value, label in values if value in text]
        for text in input_df[COL_TEXT]
    ]


@pytest.mark.parametrize("label", ["credit_debit_card", "email", "ipv4", "ipv6", "mac_address", "url"])
@pytest.mark.parametrize("validate_matches", [False, True])
@pytest.mark.parametrize("additional_matches", [False, True])
@pytest.mark.parametrize("excluded", [False, True])
def test_builtin_controls_and_validators_survive_config_export(
    tmp_path: Path, label: str, validate_matches: bool, additional_matches: bool, excluded: bool
) -> None:
    seed_path = tmp_path / "seed.parquet"
    pd.DataFrame({COL_TEXT: ["Synthetic fixture"]}).to_parquet(seed_path, index=False)
    models = parse_model_configs(None)
    # Keep a second builtin active when excluding the target, so this tests
    # exclusion rather than the documented empty-effective-label-set error.
    control = "ipv4" if label == "email" else "email"
    workflow = EntityDetectionWorkflow(adapter=NddAdapter(data_designer=cast(DataDesigner, Mock())))
    builder = workflow.build_detection_builder_for_seed(
        seed_path=seed_path,
        model_configs=models.model_configs,
        selected_models=models.selected_models.detection,
        gliner_detection_threshold=0.3,
        entity_labels=[label, control] if excluded else [label],
        excluded_entity_labels=[label] if excluded else [],
        builtin_regexes=True,
        regex_rules=[
            BuiltinRegex(
                label=label,
                validate_matches_with_llm=validate_matches,
                detect_additional_matches=additional_matches,
            ),
            *(
                [BuiltinRegex(label=control, validate_matches_with_llm=False, detect_additional_matches=False)]
                if excluded
                else []
            ),
        ],
    )
    payload = builder.get_builder_config().to_json()
    assert payload is not None
    restored = DataDesignerConfigBuilder.from_config(payload)
    assert restored.get_builder_config().to_dict() == builder.get_builder_config().to_dict()
    columns = restored.get_column_configs()
    regex = next(column for column in columns if isinstance(column, RegexDetectionConfig))
    augmenter = next(column for column in columns if isinstance(column, AugmentationConfig))
    assert _get_gliner_labels_from_builder(restored) == ([label] if additional_matches and not excluded else [])
    assert augmenter.enabled is (additional_matches and not excluded)
    assert len(regex.rules) == 1
    rule = regex.rules[0]
    assert rule.label == (control if excluded else label)
    assert rule.source == "regex_builtin"
    assert rule.validator_id is not None
    assert rule.validator_id.startswith("nemo-anonymizer.")
    assert rule.validate_matches_with_llm is (False if excluded else validate_matches)
    assert rule.detect_additional_matches is (False if excluded else additional_matches)


@pytest.mark.parametrize("builder_kind", ["dataframe", "seed"])
@pytest.mark.parametrize("rule_scope", ["excluded", "disabled", "out_of_scope", "active"])
def test_export_checks_callable_validators_only_for_active_rules(
    tmp_path: Path, builder_kind: str, rule_scope: str
) -> None:
    def validator(candidate: RegexCandidate) -> bool:
        return bool(candidate.value)

    seed_path = tmp_path / "seed.parquet"
    dataframe = pd.DataFrame({COL_TEXT: ["Alice"]})
    dataframe.to_parquet(seed_path, index=False)
    models = parse_model_configs(None)
    workflow = EntityDetectionWorkflow(adapter=NddAdapter(data_designer=cast(DataDesigner, Mock())))
    kwargs: dict[str, Any] = dict(
        seed_path=seed_path,
        model_configs=models.model_configs,
        selected_models=models.selected_models.detection,
        gliner_detection_threshold=0.3,
        entity_labels=["first_name"] if rule_scope == "out_of_scope" else ["first_name", "ticket"],
        excluded_entity_labels=["ticket"] if rule_scope == "excluded" else None,
        builtin_regexes=False,
        regex_rules=[
            RegexRule(label="ticket", pattern=r"TKT-\d+", validator=validator, enabled=rule_scope != "disabled")
        ],
    )
    if rule_scope == "active":
        with pytest.raises(ValueError, match="registered validator names"):
            if builder_kind == "dataframe":
                workflow.build_detection_config(dataframe, **kwargs)
            else:
                workflow.build_detection_builder_for_seed(**kwargs)
    else:
        builder = (
            workflow.build_detection_config(dataframe, **kwargs)
            if builder_kind == "dataframe"
            else workflow.build_detection_builder_for_seed(**kwargs)
        )
        payload = builder.get_builder_config().to_json()
        assert payload is not None
        restored = DataDesignerConfigBuilder.from_config(payload)
        assert (
            next(column for column in restored.get_column_configs() if isinstance(column, RegexDetectionConfig)).rules
            == []
        )
