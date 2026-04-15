# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from data_designer.config.models import ModelConfig

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, Rewrite
from anonymizer.config.models import ModelSelection, ReplaceModelSelection
from anonymizer.config.replace_strategies import Redact, Substitute
from anonymizer.engine.constants import (
    COL_DETECTED_ENTITIES,
    COL_FINAL_ENTITIES,
    COL_REPLACED_TEXT,
    COL_REPLACEMENT_MAP,
    COL_REWRITTEN_TEXT,
    COL_TAGGED_TEXT,
    COL_TEXT,
)
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult, EntityDetectionWorkflow
from anonymizer.engine.ndd.adapter import FailedRecord
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
    _replace_df.attrs["original_text_column"] = "text"
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
    _rewrite_df.attrs["original_text_column"] = "text"
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
    _replace_df.attrs["original_text_column"] = "text"
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
    _replace_df.attrs["original_text_column"] = "text"
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
    _replace_df.attrs["original_text_column"] = "text"
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
    replace_df.attrs["original_text_column"] = "bio"
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


def test_run_with_colliding_output_column_raises(
    stub_anonymizer_config: AnonymizerConfig,
    tmp_path: Path,
) -> None:
    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"bio": ["Alice bio text"], "bio_replaced": ["pre-existing"]}).to_csv(input_csv, index=False)
    anonymizer, _, _, _ = _make_anonymizer()

    with pytest.raises(InvalidInputError, match="collide with Anonymizer output column names"):
        anonymizer.run(
            config=stub_anonymizer_config,
            data=AnonymizerInput(source=str(input_csv), text_column="bio"),
        )


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
        update={"replace": ReplaceModelSelection(replacement_generator="bad-replace-alias")}
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
        update={"replace": ReplaceModelSelection(replacement_generator="bad-replace-alias")}
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
        update={"replace": ReplaceModelSelection(replacement_generator="bad-replace-alias")}
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
    rewrite_df.attrs["original_text_column"] = "text"
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
    _rewrite_df.attrs["original_text_column"] = "text"
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
        update={"replace": ReplaceModelSelection(replacement_generator="bad-replace-alias")}
    )

    with pytest.raises(InvalidConfigError, match="bad-replace-alias"):
        anonymizer.validate_config(AnonymizerConfig(rewrite=Rewrite()))
