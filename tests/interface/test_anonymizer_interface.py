# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from data_designer.config.models import ModelConfig

from anonymizer import RunConfig
from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, EvaluateConfig, Rewrite
from anonymizer.config.models import ModelSelection, ReplaceModelSelection
from anonymizer.config.replace_strategies import Redact, Substitute
from anonymizer.engine.constants import (
    COL_DETECTED_ENTITIES,
    COL_DETECTION_VALID,
    COL_ENTITIES_BY_VALUE,
    COL_FINAL_ENTITIES,
    COL_JUDGE_EVALUATION,
    COL_REPLACED_TEXT,
    COL_REPLACEMENT_MAP,
    COL_REWRITTEN_TEXT,
    COL_TAGGED_TEXT,
    COL_TEXT,
)
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult, EntityDetectionWorkflow
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.engine.ndd.model_loader import load_default_model_providers, validate_model_alias_references
from anonymizer.engine.replace.replace_runner import ReplacementResult, ReplacementWorkflow
from anonymizer.engine.rewrite.rewrite_workflow import RewriteResult, RewriteWorkflow
from anonymizer.interface.anonymizer import Anonymizer, _resolve_model_providers
from anonymizer.interface.errors import InvalidConfigError, InvalidInputError


@pytest.fixture
def stub_input(tmp_path: Path) -> AnonymizerInput:
    csv_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Alice"]}).to_csv(csv_path, index=False)
    return AnonymizerInput(source=str(csv_path))


def _make_anonymizer(
    detection_return: EntityDetectionResult | None = None,
    replace_return: ReplacementResult | None = None,
    rewrite_return: RewriteResult | None = None,
) -> tuple[Anonymizer, Mock, Mock, Mock]:
    detection_workflow = Mock(spec=EntityDetectionWorkflow)
    detection_workflow.run.return_value = detection_return or EntityDetectionResult(
        dataframe=pd.DataFrame({COL_TEXT: ["Alice works at Acme"], COL_FINAL_ENTITIES: [{"entities": []}]}),
        failed_records=[],
    )
    _replace_df = pd.DataFrame(
        {COL_TEXT: ["Alice works at Acme"], COL_REPLACED_TEXT: ["[REDACTED] works at [REDACTED]"]}
    )
    replace_runner = Mock(spec=ReplacementWorkflow)
    replace_runner.run.return_value = replace_return or ReplacementResult(
        dataframe=_replace_df,
        failed_records=[],
    )
    _rewrite_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme"],
            COL_REWRITTEN_TEXT: ["Beth works at Globex"],
            "utility_score": [0.85],
            "leakage_mass": [0.3],
            "weighted_leakage_rate": [0.23],
            "any_high_leaked": [False],
            "needs_human_review": [False],
        }
    )
    rewrite_runner = Mock(spec=RewriteWorkflow)
    rewrite_runner.run.return_value = rewrite_return or RewriteResult(
        dataframe=_rewrite_df,
        failed_records=[],
    )
    anonymizer = Anonymizer(
        detection_workflow=detection_workflow,
        replace_runner=replace_runner,
        rewrite_runner=rewrite_runner,
    )
    return anonymizer, detection_workflow, replace_runner, rewrite_runner


def test_run_merges_failed_records_from_both_stages(
    stub_anonymizer_config: AnonymizerConfig,
    stub_input: AnonymizerInput,
) -> None:
    detection_failures = [FailedRecord(record_id="r1", step="detection", reason="timeout")]
    replace_failures = [FailedRecord(record_id="r2", step="replace", reason="parse error")]

    detection_result = EntityDetectionResult(
        dataframe=pd.DataFrame({COL_TEXT: ["Alice"], COL_FINAL_ENTITIES: [{"entities": []}]}),
        failed_records=detection_failures,
    )
    _replace_df = pd.DataFrame({COL_TEXT: ["Alice"], COL_REPLACED_TEXT: ["Alice"]})
    replace_return = ReplacementResult(dataframe=_replace_df, failed_records=replace_failures)

    anonymizer, _, _, _ = _make_anonymizer(detection_return=detection_result, replace_return=replace_return)
    result = anonymizer.run(config=stub_anonymizer_config, data=stub_input)

    assert len(result.failed_records) == 2
    assert result.failed_records[0].step == "detection"
    assert result.failed_records[1].step == "replace"


def test_run_enables_latent_detection_when_rewrite_configured(stub_input: AnonymizerInput) -> None:
    config = AnonymizerConfig(rewrite=Rewrite())
    anonymizer, detection_wf, _, _ = _make_anonymizer()
    anonymizer.run(config=config, data=stub_input)

    assert detection_wf.run.call_args.kwargs["tag_latent_entities"] is True


def test_run_disables_latent_detection_without_rewrite(
    stub_anonymizer_config: AnonymizerConfig,
    stub_input: AnonymizerInput,
) -> None:
    anonymizer, detection_wf, _, _ = _make_anonymizer()
    anonymizer.run(config=stub_anonymizer_config, data=stub_input)

    assert detection_wf.run.call_args.kwargs["tag_latent_entities"] is False


def test_run_passes_detect_entity_labels_to_detection_workflow(stub_input: AnonymizerInput) -> None:
    config = AnonymizerConfig(detect={"entity_labels": ["server_name"]}, replace=Redact())
    anonymizer, detection_wf, _, _ = _make_anonymizer()

    anonymizer.run(config=config, data=stub_input)

    assert detection_wf.run.call_args.kwargs["entity_labels"] == ["server_name"]


def test_resolve_model_providers_raises_on_invalid_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text("not_providers: []")

    with pytest.raises(ValueError, match="providers"):
        _resolve_model_providers(yaml_path)


def test_anonymizer_default_passes_bundled_providers_to_data_designer() -> None:
    bundled = load_default_model_providers()
    with patch("anonymizer.interface.anonymizer.DataDesigner") as mock_data_designer:
        Anonymizer(
            detection_workflow=Mock(),
            replace_runner=Mock(),
            rewrite_runner=Mock(),
        )
    mock_data_designer.assert_called_once()
    passed_providers = mock_data_designer.call_args.kwargs["model_providers"]
    assert {provider.name for provider in passed_providers} == {provider.name for provider in bundled}


def test_anonymizer_custom_model_providers_override_bundled_defaults() -> None:
    from anonymizer import ModelProvider

    # Bundled model configs reference provider name "nvidia"; override the endpoint, not the name.
    custom_providers = [ModelProvider(name="nvidia", endpoint="https://example.com/v1")]
    with patch("anonymizer.interface.anonymizer.DataDesigner") as mock_data_designer:
        Anonymizer(
            model_providers=custom_providers,
            detection_workflow=Mock(),
            replace_runner=Mock(),
            rewrite_runner=Mock(),
        )
    passed_providers = mock_data_designer.call_args.kwargs["model_providers"]
    assert passed_providers is custom_providers


def test_anonymizer_applies_data_designer_run_config_to_managed_instance() -> None:
    run_config = RunConfig(buffer_size=20, max_in_flight_tasks=64)

    with patch("anonymizer.interface.anonymizer.DataDesigner") as mock_data_designer:
        Anonymizer(
            data_designer_run_config=run_config,
            detection_workflow=Mock(),
            replace_runner=Mock(),
            rewrite_runner=Mock(),
        )

    mock_data_designer.return_value.set_run_config.assert_called_once_with(run_config)


def test_anonymizer_applies_data_designer_run_config_to_supplied_instance() -> None:
    from data_designer.interface.data_designer import DataDesigner

    data_designer = Mock(spec=DataDesigner)
    run_config = RunConfig(buffer_size=20, max_in_flight_tasks=64)

    Anonymizer(
        data_designer=data_designer,
        data_designer_run_config=run_config,
        detection_workflow=Mock(),
        replace_runner=Mock(),
        rewrite_runner=Mock(),
    )

    data_designer.set_run_config.assert_called_once_with(run_config)


def test_anonymizer_rejects_missing_provider_as_invalid_config_error() -> None:
    yaml_str = """
model_configs:
  - alias: custom-detector
    model: test/model
"""
    with pytest.raises(InvalidConfigError, match="missing required field 'provider'"):
        Anonymizer(
            model_configs=yaml_str,
            detection_workflow=Mock(),
            replace_runner=Mock(),
            rewrite_runner=Mock(),
        )


def test_anonymizer_rejects_unknown_model_provider_as_invalid_config_error() -> None:
    from anonymizer import ModelProvider

    yaml_str = """
model_configs:
  - alias: custom-detector
    model: test/model
    provider: unknown-provider
"""
    providers = [ModelProvider(name="nvidia", endpoint="https://example.com/v1")]
    with pytest.raises(InvalidConfigError, match="unknown-provider"):
        Anonymizer(
            model_configs=yaml_str,
            model_providers=providers,
            detection_workflow=Mock(),
            replace_runner=Mock(),
            rewrite_runner=Mock(),
        )


def test_anonymizer_rejects_empty_model_providers_list() -> None:
    with pytest.raises(InvalidConfigError, match="at least one provider"):
        Anonymizer(
            model_providers=[],
            detection_workflow=Mock(),
            replace_runner=Mock(),
            rewrite_runner=Mock(),
        )


def test_anonymizer_skips_provider_validation_with_supplied_data_designer() -> None:
    from data_designer.interface.data_designer import DataDesigner

    # Provider not in bundled defaults; would raise if validated against them.
    yaml_str = """
model_configs:
  - alias: custom-detector
    model: test/model
    provider: my-own-provider
"""
    anonymizer = Anonymizer(
        model_configs=yaml_str,
        data_designer=Mock(spec=DataDesigner),
        detection_workflow=Mock(),
        replace_runner=Mock(),
        rewrite_runner=Mock(),
    )
    assert {config.provider for config in anonymizer._model_configs} == {"my-own-provider"}


def test_run_exposes_trace_dataframe_and_filters_internal_columns(
    stub_anonymizer_config: AnonymizerConfig,
    stub_input: AnonymizerInput,
) -> None:
    _replace_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice"],
            COL_REPLACED_TEXT: ["[REDACTED]"],
            COL_TAGGED_TEXT: ["<first_name>Alice</first_name>"],
            COL_DETECTED_ENTITIES: [{"entities": [{"value": "Alice", "label": "first_name"}]}],
            COL_REPLACEMENT_MAP: [{"replacements": []}],
        }
    )
    replace_return = ReplacementResult(dataframe=_replace_df, failed_records=[])
    anonymizer, _, _, _ = _make_anonymizer(replace_return=replace_return)

    result = anonymizer.run(config=stub_anonymizer_config, data=stub_input)

    assert COL_DETECTED_ENTITIES in result.trace_dataframe.columns
    assert COL_REPLACEMENT_MAP in result.trace_dataframe.columns
    assert COL_DETECTED_ENTITIES not in result.dataframe.columns
    assert COL_REPLACEMENT_MAP not in result.dataframe.columns
    assert set(result.dataframe.columns) == {"text", "text_replaced", "text_with_spans"}


def test_preview_exposes_trace_dataframe_for_display(
    stub_anonymizer_config: AnonymizerConfig,
    stub_input: AnonymizerInput,
) -> None:
    _replace_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice"],
            COL_REPLACED_TEXT: ["[REDACTED]"],
            COL_DETECTED_ENTITIES: [{"entities": [{"value": "Alice", "label": "first_name"}]}],
            COL_REPLACEMENT_MAP: [{"replacements": []}],
        }
    )
    replace_return = ReplacementResult(dataframe=_replace_df, failed_records=[])
    anonymizer, _, _, _ = _make_anonymizer(replace_return=replace_return)

    preview = anonymizer.preview(
        config=stub_anonymizer_config,
        data=stub_input,
        num_records=1,
    )

    assert COL_DETECTED_ENTITIES in preview.trace_dataframe.columns
    assert COL_DETECTED_ENTITIES not in preview.dataframe.columns


def test_run_restores_original_text_column_names_for_user_dataframe(
    stub_anonymizer_config: AnonymizerConfig,
    tmp_path: Path,
) -> None:
    replace_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice bio text"],
            COL_REPLACED_TEXT: ["[REDACTED] bio text"],
            COL_TAGGED_TEXT: ["<first_name>Alice</first_name> bio text"],
            COL_DETECTED_ENTITIES: [{"entities": [{"value": "Alice", "label": "first_name"}]}],
        }
    )
    replace_return = ReplacementResult(dataframe=replace_df, failed_records=[])
    anonymizer, _, _, _ = _make_anonymizer(replace_return=replace_return)

    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"bio": ["Alice bio text"]}).to_csv(input_csv, index=False)

    result = anonymizer.run(
        config=stub_anonymizer_config,
        data=AnonymizerInput(source=str(input_csv), text_column="bio"),
    )

    assert "bio" in result.dataframe.columns
    assert "bio_replaced" in result.dataframe.columns
    assert "bio_with_spans" in result.dataframe.columns
    assert "text" not in result.dataframe.columns
    assert "replaced_text" not in result.dataframe.columns
    assert result.dataframe["bio"].iloc[0] == "Alice bio text"
    assert result.dataframe["bio_replaced"].iloc[0] == "[REDACTED] bio text"
    assert result.resolved_text_column == "bio"


def test_run_ignores_workflow_output_attrs_for_text_column_resolution(
    stub_anonymizer_config: AnonymizerConfig,
    tmp_path: Path,
) -> None:
    """Behavioral guarantee: the orchestrator's text-column resolution comes
    from :class:`ResolvedInput`, never from workflow output ``DataFrame.attrs``.

    Construct workflow mocks whose returned dataframes carry deliberately
    misleading ``attrs`` (a key that historical implementations might have
    relied on). If the orchestrator silently consulted ``attrs`` it would
    produce a column named ``WRONG_COL`` instead of ``bio``, and
    ``display_record`` would fail to find the text. We assert the user-facing
    dataframe and the display HTML both reflect the user's actual
    ``text_column`` request.
    """
    detection_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice bio text"],
            COL_FINAL_ENTITIES: [{"entities": [{"value": "Alice", "label": "first_name"}]}],
        }
    )
    detection_df.attrs = {"resolved_text_column": "WRONG_COL", "requested_text_column": "WRONG_COL"}
    replace_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice bio text"],
            COL_REPLACED_TEXT: ["[REDACTED] bio text"],
            COL_TAGGED_TEXT: ["<first_name>Alice</first_name> bio text"],
            COL_DETECTED_ENTITIES: [
                {"entities": [{"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5}]}
            ],
            COL_REPLACEMENT_MAP: [
                {"replacements": [{"original": "Alice", "label": "first_name", "synthetic": "[REDACTED]"}]}
            ],
        }
    )
    replace_df.attrs = {"resolved_text_column": "WRONG_COL", "extra": "noise"}

    anonymizer, _, _, _ = _make_anonymizer(
        detection_return=EntityDetectionResult(dataframe=detection_df, failed_records=[]),
        replace_return=ReplacementResult(dataframe=replace_df, failed_records=[]),
    )

    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"bio": ["Alice bio text"]}).to_csv(input_csv, index=False)
    result = anonymizer.run(
        config=stub_anonymizer_config,
        data=AnonymizerInput(source=str(input_csv), text_column="bio"),
    )

    assert result.resolved_text_column == "bio"
    assert "bio" in result.dataframe.columns
    assert "bio_replaced" in result.dataframe.columns
    assert "WRONG_COL" not in result.dataframe.columns
    assert "WRONG_COL" not in result.trace_dataframe.columns

    from anonymizer.interface.display import render_record_html

    rendered = render_record_html(
        result.trace_dataframe.iloc[0],
        record_index=0,
        resolved_text_column=result.resolved_text_column,
    )
    assert "Alice" in rendered
    assert "bio text" in rendered
    assert "[REDACTED]" in rendered
    assert "WRONG_COL" not in rendered


def test_run_with_colliding_internal_text_column_raises(
    stub_anonymizer_config: AnonymizerConfig,
    tmp_path: Path,
) -> None:
    input_csv = tmp_path / "input.csv"
    pd.DataFrame({COL_TEXT: ["internal"], "bio": ["Alice bio text"]}).to_csv(input_csv, index=False)
    anonymizer, _, _, _ = _make_anonymizer()

    with pytest.raises(InvalidInputError, match="reserved internal column"):
        anonymizer.run(
            config=stub_anonymizer_config,
            data=AnonymizerInput(source=str(input_csv), text_column="bio"),
        )


def test_run_with_colliding_output_column_renames_input_and_warns(
    stub_anonymizer_config: AnonymizerConfig,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Colliding input columns are renamed (not blocked) and a warning is logged.

    We use a custom ``replace_return`` that propagates the renamed input column
    so we can verify that it survives into the trace dataframe alongside the
    fresh pipeline output. This documents the lossless behaviour: the user's
    original ``bio_replaced`` data ends up under ``bio_replaced__input`` while
    the pipeline writes its own ``bio_replaced``.
    """
    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"bio": ["Alice bio text"], "bio_replaced": ["pre-existing"]}).to_csv(input_csv, index=False)

    replace_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice bio text"],
            COL_REPLACED_TEXT: ["[REDACTED] bio text"],
            "bio_replaced__input": ["pre-existing"],
        }
    )
    anonymizer, _, _, _ = _make_anonymizer(
        replace_return=ReplacementResult(dataframe=replace_df, failed_records=[]),
    )

    with caplog.at_level("WARNING", logger="anonymizer"):
        result = anonymizer.run(
            config=stub_anonymizer_config,
            data=AnonymizerInput(source=str(input_csv), text_column="bio"),
        )

    assert any("collide with Anonymizer output column names" in rec.message for rec in caplog.records)
    assert result.dataframe["bio_replaced"].iloc[0] == "[REDACTED] bio text"
    assert result.trace_dataframe["bio_replaced__input"].iloc[0] == "pre-existing"


def test_run_with_text_column_matching_static_output_preserves_both_columns(
    stub_anonymizer_config: AnonymizerConfig,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Setting ``text_column`` to a fixed output name (e.g. ``final_entities``)
    must not produce duplicate columns in the trace/user dataframes.

    The reader renames the input column to ``final_entities__input`` so the
    pipeline's own ``final_entities`` output (the detected entities dict) and
    the user's original text live side-by-side as distinct columns.
    """
    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"final_entities": ["Alice bio text"]}).to_csv(input_csv, index=False)

    entities_payload = {"entities": [{"value": "Alice", "label": "first_name"}]}
    replace_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice bio text"],
            COL_REPLACED_TEXT: ["[REDACTED] bio text"],
            COL_TAGGED_TEXT: ["<first_name>Alice</first_name> bio text"],
            COL_FINAL_ENTITIES: [entities_payload],
        }
    )
    anonymizer, _, _, _ = _make_anonymizer(
        replace_return=ReplacementResult(dataframe=replace_df, failed_records=[]),
    )

    with caplog.at_level("WARNING", logger="anonymizer"):
        result = anonymizer.run(
            config=stub_anonymizer_config,
            data=AnonymizerInput(source=str(input_csv), text_column="final_entities"),
        )

    assert any("collide with Anonymizer output column names" in rec.message for rec in caplog.records)

    for frame_name, frame in (("dataframe", result.dataframe), ("trace_dataframe", result.trace_dataframe)):
        assert list(frame.columns).count("final_entities") == 1, (
            f"{frame_name} has duplicate 'final_entities' columns: {list(frame.columns)}"
        )
        assert "final_entities__input" in frame.columns, f"{frame_name} missing renamed user text column"
        assert frame["final_entities__input"].iloc[0] == "Alice bio text"
        assert frame["final_entities"].iloc[0] == entities_payload


def test_validate_config_passes_for_valid_replace_config(stub_anonymizer_config: AnonymizerConfig) -> None:
    anonymizer, _, _, _ = _make_anonymizer()
    anonymizer.validate_config(stub_anonymizer_config)


def test_validate_config_passes_for_valid_substitute_config() -> None:
    anonymizer, _, _, _ = _make_anonymizer()
    anonymizer.validate_config(AnonymizerConfig(replace=Substitute()))


def test_validate_config_passes_for_valid_rewrite_config() -> None:
    anonymizer, _, _, _ = _make_anonymizer()
    anonymizer.validate_config(AnonymizerConfig(rewrite=Rewrite()))


def test_validate_config_raises_on_unknown_detection_alias(
    stub_anonymizer_config: AnonymizerConfig,
    stub_known_model_configs: list[ModelConfig],
    stub_slim_model_selection: ModelSelection,
) -> None:
    anonymizer, _, _, _ = _make_anonymizer()
    anonymizer._model_configs = stub_known_model_configs
    anonymizer._selected_models = stub_slim_model_selection
    anonymizer._selected_models = anonymizer._selected_models.model_copy(
        update={
            "detection": anonymizer._selected_models.detection.model_copy(
                update={"entity_detector": "bad-detection-alias"}
            )
        }
    )

    with pytest.raises(InvalidConfigError, match="bad-detection-alias"):
        anonymizer.validate_config(stub_anonymizer_config)


def test_validate_config_raises_on_unknown_replace_alias_for_substitute(
    stub_known_model_configs: list[ModelConfig],
    stub_slim_model_selection: ModelSelection,
) -> None:
    anonymizer, _, _, _ = _make_anonymizer()
    anonymizer._model_configs = stub_known_model_configs
    anonymizer._selected_models = stub_slim_model_selection
    anonymizer._selected_models = anonymizer._selected_models.model_copy(
        update={
            "replace": ReplaceModelSelection(
                replacement_generator="bad-replace-alias",
                detection_judge="known",
                type_fidelity_judge="known",
                relational_consistency_judge="known",
                attribute_fidelity_judge="known",
            )
        }
    )

    with pytest.raises(InvalidConfigError, match="bad-replace-alias"):
        anonymizer.validate_config(AnonymizerConfig(replace=Substitute()))


def test_validate_config_skips_replace_alias_for_non_substitute(
    stub_anonymizer_config: AnonymizerConfig,
    stub_known_model_configs: list[ModelConfig],
    stub_slim_model_selection: ModelSelection,
) -> None:
    anonymizer, _, _, _ = _make_anonymizer()
    anonymizer._model_configs = stub_known_model_configs
    anonymizer._selected_models = stub_slim_model_selection
    anonymizer._selected_models = anonymizer._selected_models.model_copy(
        update={
            "replace": ReplaceModelSelection(
                replacement_generator="bad-replace-alias",
                detection_judge="known",
                type_fidelity_judge="known",
                relational_consistency_judge="known",
                attribute_fidelity_judge="known",
            )
        }
    )

    anonymizer.validate_config(stub_anonymizer_config)


def test_validate_config_raises_on_unknown_rewrite_alias(
    stub_known_model_configs: list[ModelConfig],
    stub_slim_model_selection: ModelSelection,
) -> None:
    anonymizer, _, _, _ = _make_anonymizer()
    anonymizer._model_configs = stub_known_model_configs
    anonymizer._selected_models = stub_slim_model_selection
    anonymizer._selected_models = anonymizer._selected_models.model_copy(
        update={"rewrite": anonymizer._selected_models.rewrite.model_copy(update={"rewriter": "bad-rewrite-alias"})}
    )

    with pytest.raises(InvalidConfigError, match="bad-rewrite-alias"):
        anonymizer.validate_config(AnonymizerConfig(rewrite=Rewrite()))


def test_validate_config_skips_latent_detector_without_rewrite(
    stub_anonymizer_config: AnonymizerConfig,
    stub_known_model_configs: list[ModelConfig],
    stub_slim_model_selection: ModelSelection,
) -> None:
    anonymizer, _, _, _ = _make_anonymizer()
    anonymizer._model_configs = stub_known_model_configs
    anonymizer._selected_models = stub_slim_model_selection.model_copy(
        update={"rewrite": stub_slim_model_selection.rewrite.model_copy(update={"rewriter": "bad-rewrite-alias"})}
    )

    anonymizer.validate_config(stub_anonymizer_config)


def test_validate_config_raises_on_unknown_latent_detector_in_rewrite(
    stub_known_model_configs: list[ModelConfig],
    stub_slim_model_selection: ModelSelection,
) -> None:
    anonymizer, _, _, _ = _make_anonymizer()
    anonymizer._model_configs = stub_known_model_configs
    anonymizer._selected_models = stub_slim_model_selection.model_copy(
        update={
            "detection": stub_slim_model_selection.detection.model_copy(update={"latent_detector": "bad-latent-alias"})
        }
    )

    with pytest.raises(InvalidConfigError, match="bad-latent-alias"):
        anonymizer.validate_config(AnonymizerConfig(rewrite=Rewrite()))


def test_validate_config_skips_rewrite_alias_without_rewrite(
    stub_anonymizer_config: AnonymizerConfig,
    stub_known_model_configs: list[ModelConfig],
    stub_slim_model_selection: ModelSelection,
) -> None:
    anonymizer, _, _, _ = _make_anonymizer()
    anonymizer._model_configs = stub_known_model_configs
    anonymizer._selected_models = stub_slim_model_selection
    anonymizer._selected_models = anonymizer._selected_models.model_copy(
        update={"rewrite": anonymizer._selected_models.rewrite.model_copy(update={"rewriter": "bad-rewrite-alias"})}
    )

    anonymizer.validate_config(stub_anonymizer_config)


def test_run_raises_invalid_config_before_workflows(
    stub_input: AnonymizerInput,
    stub_known_model_configs: list[ModelConfig],
    stub_slim_model_selection: ModelSelection,
) -> None:
    anonymizer, detection_wf, replace_runner, _ = _make_anonymizer()
    anonymizer._model_configs = stub_known_model_configs
    anonymizer._selected_models = stub_slim_model_selection
    anonymizer._selected_models = anonymizer._selected_models.model_copy(
        update={
            "replace": ReplaceModelSelection(
                replacement_generator="bad-replace-alias",
                detection_judge="known",
                type_fidelity_judge="known",
                relational_consistency_judge="known",
                attribute_fidelity_judge="known",
            )
        }
    )

    with pytest.raises(InvalidConfigError, match="bad-replace-alias"):
        anonymizer.run(config=AnonymizerConfig(replace=Substitute()), data=stub_input)

    detection_wf.run.assert_not_called()
    replace_runner.run.assert_not_called()


def test_preview_raises_invalid_config_before_workflows(
    stub_input: AnonymizerInput,
    stub_known_model_configs: list[ModelConfig],
    stub_slim_model_selection: ModelSelection,
) -> None:
    anonymizer, detection_wf, replace_runner, _ = _make_anonymizer()
    anonymizer._model_configs = stub_known_model_configs
    anonymizer._selected_models = stub_slim_model_selection
    anonymizer._selected_models = anonymizer._selected_models.model_copy(
        update={"rewrite": anonymizer._selected_models.rewrite.model_copy(update={"rewriter": "bad-rewrite-alias"})}
    )

    with pytest.raises(InvalidConfigError, match="bad-rewrite-alias"):
        anonymizer.preview(config=AnonymizerConfig(rewrite=Rewrite()), data=stub_input, num_records=1)

    detection_wf.run.assert_not_called()
    replace_runner.run.assert_not_called()


# ---------------------------------------------------------------------------
# Rewrite wiring tests
# ---------------------------------------------------------------------------


def test_run_rewrite_calls_rewrite_runner(stub_input: AnonymizerInput) -> None:
    config = AnonymizerConfig(rewrite=Rewrite())
    anonymizer, _, _, rewrite_runner = _make_anonymizer()

    anonymizer.run(config=config, data=stub_input)

    rewrite_runner.run.assert_called_once()
    call_kwargs = rewrite_runner.run.call_args.kwargs
    assert call_kwargs["privacy_goal"] == config.rewrite.privacy_goal
    assert call_kwargs["evaluation"] == config.rewrite.evaluation


def test_run_rewrite_output_columns(stub_input: AnonymizerInput) -> None:
    config = AnonymizerConfig(rewrite=Rewrite())
    anonymizer, _, _, _ = _make_anonymizer()

    result = anonymizer.run(config=config, data=stub_input)

    assert "text_rewritten" in result.dataframe.columns
    assert "utility_score" in result.dataframe.columns
    assert "leakage_mass" in result.dataframe.columns
    assert "weighted_leakage_rate" in result.dataframe.columns
    assert "any_high_leaked" in result.dataframe.columns
    assert "needs_human_review" in result.dataframe.columns


def test_run_rewrite_internal_columns_only_in_trace(stub_input: AnonymizerInput) -> None:
    rewrite_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme"],
            COL_REWRITTEN_TEXT: ["Beth works at Globex"],
            "_domain": ["general"],
            "_sensitivity_disposition": [None],
            "utility_score": [0.85],
            "leakage_mass": [0.3],
            "weighted_leakage_rate": [0.23],
            "any_high_leaked": [False],
            "needs_human_review": [False],
        }
    )
    rewrite_return = RewriteResult(dataframe=rewrite_df, failed_records=[])
    config = AnonymizerConfig(rewrite=Rewrite())
    anonymizer, _, _, _ = _make_anonymizer(rewrite_return=rewrite_return)

    result = anonymizer.run(config=config, data=stub_input)

    assert "_domain" in result.trace_dataframe.columns
    assert "_domain" not in result.dataframe.columns
    assert "_sensitivity_disposition" in result.trace_dataframe.columns
    assert "_sensitivity_disposition" not in result.dataframe.columns


def test_run_rewrite_merges_failed_records(stub_input: AnonymizerInput) -> None:
    detection_failures = [FailedRecord(record_id="r1", step="detection", reason="timeout")]
    rewrite_failures = [FailedRecord(record_id="r2", step="rewrite", reason="parse error")]

    detection_result = EntityDetectionResult(
        dataframe=pd.DataFrame({COL_TEXT: ["Alice"], COL_FINAL_ENTITIES: [{"entities": []}]}),
        failed_records=detection_failures,
    )
    _rewrite_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice"],
            COL_REWRITTEN_TEXT: ["Beth"],
            "utility_score": [0.85],
            "leakage_mass": [0.3],
            "weighted_leakage_rate": [0.23],
            "any_high_leaked": [False],
            "needs_human_review": [False],
        }
    )
    rewrite_return = RewriteResult(dataframe=_rewrite_df, failed_records=rewrite_failures)

    config = AnonymizerConfig(rewrite=Rewrite())
    anonymizer, _, _, _ = _make_anonymizer(detection_return=detection_result, rewrite_return=rewrite_return)
    result = anonymizer.run(config=config, data=stub_input)

    assert len(result.failed_records) == 2
    assert result.failed_records[0].step == "detection"
    assert result.failed_records[1].step == "rewrite"


def test_run_rewrite_passes_compute_grouped_entities(stub_input: AnonymizerInput) -> None:
    config = AnonymizerConfig(rewrite=Rewrite())
    anonymizer, detection_wf, _, _ = _make_anonymizer()

    anonymizer.run(config=config, data=stub_input)

    assert detection_wf.run.call_args.kwargs["compute_grouped_entities"] is True


def test_validate_config_raises_on_unknown_replace_alias_in_rewrite_mode(
    stub_known_model_configs: list[ModelConfig],
    stub_slim_model_selection: ModelSelection,
) -> None:
    """Rewrite mode uses replace aliases for the replacement map; validate them up front."""
    anonymizer, _, _, _ = _make_anonymizer()
    anonymizer._model_configs = stub_known_model_configs
    anonymizer._selected_models = stub_slim_model_selection
    anonymizer._selected_models = anonymizer._selected_models.model_copy(
        update={
            "replace": ReplaceModelSelection(
                replacement_generator="bad-replace-alias",
                detection_judge="known",
                type_fidelity_judge="known",
                relational_consistency_judge="known",
                attribute_fidelity_judge="known",
            )
        }
    )

    with pytest.raises(InvalidConfigError, match="bad-replace-alias"):
        anonymizer.validate_config(AnonymizerConfig(rewrite=Rewrite()))


def test_evaluate_raises_value_error_on_legacy_result_without_replace_method() -> None:
    """A pickled result from before `replace_method` existed should surface the
    actionable ValueError, not an AttributeError from the missing attribute."""
    anonymizer, _, _, _ = _make_anonymizer()
    legacy_result = SimpleNamespace(
        dataframe=pd.DataFrame(),
        trace_dataframe=pd.DataFrame(),
        resolved_text_column="text",
    )

    with pytest.raises(ValueError, match="replace_method"):
        anonymizer.evaluate(legacy_result)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests: Anonymizer.evaluate() for rewrite results
# ---------------------------------------------------------------------------


def test_run_rewrite_does_not_include_judge_in_user_dataframe(stub_input: AnonymizerInput) -> None:
    """run() output must not include COL_JUDGE_EVALUATION — it only appears after evaluate()."""
    config = AnonymizerConfig(rewrite=Rewrite())
    anonymizer, _, _, _ = _make_anonymizer()

    result = anonymizer.run(config=config, data=stub_input)

    assert COL_JUDGE_EVALUATION not in result.dataframe.columns


def test_evaluate_rewrite_result_adds_judge_columns(stub_input: AnonymizerInput) -> None:
    """anonymizer.evaluate() on a rewrite result must add COL_JUDGE_EVALUATION."""
    config = AnonymizerConfig(rewrite=Rewrite())
    anonymizer, _, _, rewrite_runner = _make_anonymizer()

    run_result = anonymizer.run(config=config, data=stub_input)

    eval_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme"],
            COL_REWRITTEN_TEXT: ["Beth works at Globex"],
            "utility_score": [0.85],
            "leakage_mass": [0.3],
            "weighted_leakage_rate": [0.23],
            "any_high_leaked": [False],
            "needs_human_review": [False],
            COL_JUDGE_EVALUATION: [{"privacy": {"score": "high"}}],
            COL_DETECTION_VALID: [1.0],
        }
    )
    rewrite_runner.evaluate.return_value = RewriteResult(dataframe=eval_df, failed_records=[])

    evaluated = anonymizer.evaluate(run_result)

    assert COL_JUDGE_EVALUATION in evaluated.dataframe.columns


def test_evaluate_rewrite_result_adds_detection_valid(stub_input: AnonymizerInput) -> None:
    """anonymizer.evaluate() on a rewrite result must add COL_DETECTION_VALID."""
    config = AnonymizerConfig(rewrite=Rewrite())
    anonymizer, _, _, rewrite_runner = _make_anonymizer()

    run_result = anonymizer.run(config=config, data=stub_input)

    eval_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme"],
            COL_REWRITTEN_TEXT: ["Beth works at Globex"],
            "utility_score": [0.85],
            "leakage_mass": [0.3],
            "weighted_leakage_rate": [0.23],
            "any_high_leaked": [False],
            "needs_human_review": [False],
            COL_JUDGE_EVALUATION: [None],
            COL_DETECTION_VALID: [0.9],
            COL_ENTITIES_BY_VALUE: [{}],
        }
    )
    rewrite_runner.evaluate.return_value = RewriteResult(dataframe=eval_df, failed_records=[])

    evaluated = anonymizer.evaluate(run_result, config=EvaluateConfig(compute_detection_validity=True))

    assert COL_DETECTION_VALID in evaluated.dataframe.columns


def test_evaluate_rewrite_raises_without_rewrite_config() -> None:
    """evaluate() must raise ValueError when result has no rewrite_config and no replace_method."""
    anonymizer, _, _, _ = _make_anonymizer()
    bare_result = SimpleNamespace(
        dataframe=pd.DataFrame(),
        trace_dataframe=pd.DataFrame(),
        resolved_text_column="text",
        rewrite_config=None,
    )

    with pytest.raises(ValueError):
        anonymizer.evaluate(bare_result)  # type: ignore[arg-type]


def test_evaluate_rewrite_raises_on_bad_rewrite_judge_alias(
    stub_input: AnonymizerInput,
    stub_known_model_configs: list[ModelConfig],
    stub_slim_model_selection: ModelSelection,
) -> None:
    """evaluate() on a rewrite result must raise InvalidConfigError for an unknown rewrite_judge alias.

    This is the integration-level counterpart to the unit tests on
    validate_model_alias_references — it guards the path from Anonymizer.evaluate()
    through to the validator so that a misconfigured evaluate.rewrite_judge is caught
    before any LLM call is made.
    """
    config = AnonymizerConfig(rewrite=Rewrite())
    anonymizer, _, _, _ = _make_anonymizer()
    anonymizer._model_configs = stub_known_model_configs
    anonymizer._selected_models = stub_slim_model_selection.model_copy(
        update={
            "evaluate": stub_slim_model_selection.evaluate.model_copy(
                update={"rewrite_judge": "bad-rewrite-judge-alias"}
            )
        }
    )

    run_result = anonymizer.run(config=config, data=stub_input)

    with pytest.raises(InvalidConfigError, match="bad-rewrite-judge-alias"):
        anonymizer.evaluate(run_result)


def test_evaluate_rewrite_calls_validate_with_check_rewrite_false(stub_input: AnonymizerInput) -> None:
    """evaluate() on a rewrite result must NOT validate rewrite pipeline model aliases.

    Passing check_rewrite=True would require domain-classifier / rewrite-generator
    aliases that are irrelevant for post-hoc evaluation. This test asserts the call
    uses check_rewrite=False so users with evaluate-only configs are not blocked.
    """
    from unittest.mock import patch as _patch

    config = AnonymizerConfig(rewrite=Rewrite())
    anonymizer, _, _, rewrite_runner = _make_anonymizer()

    run_result = anonymizer.run(config=config, data=stub_input)

    eval_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme"],
            COL_REWRITTEN_TEXT: ["Beth works at Globex"],
            "utility_score": [0.85],
            "leakage_mass": [0.3],
            "weighted_leakage_rate": [0.23],
            "any_high_leaked": [False],
            "needs_human_review": [False],
            COL_JUDGE_EVALUATION: [None],
            COL_DETECTION_VALID: [1.0],
        }
    )
    rewrite_runner.evaluate.return_value = RewriteResult(dataframe=eval_df, failed_records=[])

    with _patch(
        "anonymizer.interface.anonymizer.validate_model_alias_references",
        wraps=validate_model_alias_references,
    ) as mock_validate:
        anonymizer.evaluate(run_result)

    rewrite_eval_calls = [call for call in mock_validate.call_args_list if call.kwargs.get("check_evaluate") is True]
    assert rewrite_eval_calls, "validate_model_alias_references was not called with check_evaluate=True"
    for call in rewrite_eval_calls:
        assert call.kwargs.get("check_rewrite") is False, (
            "evaluate() on a rewrite result must pass check_rewrite=False to avoid "
            "requiring rewrite pipeline model aliases that are unused during evaluation"
        )


# ---------------------------------------------------------------------------
# Tests: strict_entity_protection flows config -> result -> coverage judge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strict", [True, False])
def test_run_persists_strict_entity_protection(stub_input: AnonymizerInput, strict: bool) -> None:
    """run() must copy config.rewrite.strict_entity_protection onto the result."""
    config = AnonymizerConfig(rewrite=Rewrite(strict_entity_protection=strict))
    anonymizer, _, _, _ = _make_anonymizer()

    result = anonymizer.run(config=config, data=stub_input)

    assert result.strict_entity_protection is strict


@pytest.mark.parametrize("strict", [True, False])
def test_preview_persists_strict_entity_protection(stub_input: AnonymizerInput, strict: bool) -> None:
    """preview() must copy config.rewrite.strict_entity_protection onto the PreviewResult."""
    config = AnonymizerConfig(rewrite=Rewrite(strict_entity_protection=strict))
    anonymizer, _, _, _ = _make_anonymizer()

    preview = anonymizer.preview(config=config, data=stub_input, num_records=1)

    assert preview.strict_entity_protection is strict


def test_evaluate_passes_strict_entity_protection_to_coverage_judge(stub_input: AnonymizerInput) -> None:
    """evaluate() must forward the result's strict flag into EntityCoverageWorkflow.

    Regression: strict_entity_protection was dropped after run()/preview(), so the
    entity-coverage judge always scored in non-strict mode regardless of config.
    """
    config = AnonymizerConfig(rewrite=Rewrite(strict_entity_protection=True))
    anonymizer, _, _, rewrite_runner = _make_anonymizer()

    run_result = anonymizer.run(config=config, data=stub_input)

    eval_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme"],
            COL_REWRITTEN_TEXT: ["Beth works at Globex"],
            "utility_score": [0.85],
            "leakage_mass": [0.3],
            "weighted_leakage_rate": [0.23],
            "any_high_leaked": [False],
            "needs_human_review": [False],
            COL_JUDGE_EVALUATION: [None],
            COL_DETECTION_VALID: [1.0],
        }
    )
    rewrite_runner.evaluate.return_value = RewriteResult(dataframe=eval_df, failed_records=[])

    with patch("anonymizer.interface.anonymizer.EntityCoverageWorkflow") as mock_coverage_wf:
        mock_coverage_wf.return_value.run_non_critical.return_value = (eval_df, [])
        anonymizer.evaluate(run_result)

    assert mock_coverage_wf.call_args is not None, "EntityCoverageWorkflow was not constructed"
    assert mock_coverage_wf.call_args.kwargs["strict_entity_protection"] is True
