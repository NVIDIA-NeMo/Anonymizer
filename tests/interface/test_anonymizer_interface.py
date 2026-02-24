# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput
from anonymizer.config.replace_strategies import RedactReplace
from anonymizer.config.rewrite import RewriteParams
from anonymizer.engine.constants import (
    COL_DETECTED_ENTITIES,
    COL_REPLACED_TEXT,
    COL_REPLACEMENT_MAP,
    COL_TAGGED_TEXT,
    COL_TEXT,
)
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult, EntityDetectionWorkflow
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.engine.replace.replace_runner import ReplaceRunner
from anonymizer.interface.anonymizer import Anonymizer, _resolve_model_providers
from anonymizer.interface.errors import InvalidInputError


@pytest.fixture
def stub_input(tmp_path: Path) -> AnonymizerInput:
    csv_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Alice"]}).to_csv(csv_path, index=False)
    return AnonymizerInput(source=str(csv_path))


def _make_anonymizer(
    detection_return: EntityDetectionResult | None = None,
    replace_return: tuple | None = None,
) -> tuple[Anonymizer, Mock, Mock]:
    detection_workflow = Mock(spec=EntityDetectionWorkflow)
    detection_workflow.run.return_value = detection_return or EntityDetectionResult(
        dataframe=pd.DataFrame({COL_TEXT: ["Alice works at Acme"], COL_DETECTED_ENTITIES: [[]]}),
        failed_records=[],
    )
    replace_runner = Mock(spec=ReplaceRunner)
    replace_runner.run.return_value = replace_return or (
        pd.DataFrame({COL_TEXT: ["Alice works at Acme"], COL_REPLACED_TEXT: ["[REDACTED] works at [REDACTED]"]}),
        [],
    )
    anonymizer = Anonymizer(detection_workflow=detection_workflow, replace_runner=replace_runner)
    return anonymizer, detection_workflow, replace_runner


def test_run_merges_failed_records_from_both_stages(
    stub_anonymizer_config: AnonymizerConfig,
    stub_input: AnonymizerInput,
) -> None:
    detection_failures = [FailedRecord(record_id="r1", step="detection", reason="timeout")]
    replace_failures = [FailedRecord(record_id="r2", step="replace", reason="parse error")]

    detection_result = EntityDetectionResult(
        dataframe=pd.DataFrame({COL_TEXT: ["Alice"], COL_DETECTED_ENTITIES: [[]]}),
        failed_records=detection_failures,
    )
    replace_return = (pd.DataFrame({COL_TEXT: ["Alice"], COL_REPLACED_TEXT: ["Alice"]}), replace_failures)

    anonymizer, _, _ = _make_anonymizer(detection_return=detection_result, replace_return=replace_return)
    result = anonymizer.run(config=stub_anonymizer_config, data=stub_input)

    assert len(result.failed_records) == 2
    assert result.failed_records[0].step == "detection"
    assert result.failed_records[1].step == "replace"


def test_run_enables_latent_detection_when_rewrite_configured(stub_input: AnonymizerInput) -> None:
    config = AnonymizerConfig(replace=RedactReplace(), rewrite=RewriteParams())
    anonymizer, detection_wf, _ = _make_anonymizer()
    anonymizer.run(config=config, data=stub_input)

    assert detection_wf.run.call_args.kwargs["tag_latent_entities"] is True


def test_run_disables_latent_detection_without_rewrite(
    stub_anonymizer_config: AnonymizerConfig,
    stub_input: AnonymizerInput,
) -> None:
    anonymizer, detection_wf, _ = _make_anonymizer()
    anonymizer.run(config=stub_anonymizer_config, data=stub_input)

    assert detection_wf.run.call_args.kwargs["tag_latent_entities"] is False


def test_resolve_model_providers_raises_on_invalid_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text("not_providers: []")

    with pytest.raises(ValueError, match="providers"):
        _resolve_model_providers(yaml_path)


def test_run_exposes_trace_dataframe_and_filters_internal_columns(
    stub_anonymizer_config: AnonymizerConfig,
    stub_input: AnonymizerInput,
) -> None:
    replace_return = (
        pd.DataFrame(
            {
                COL_TEXT: ["Alice"],
                COL_REPLACED_TEXT: ["[REDACTED]"],
                COL_TAGGED_TEXT: ["<first_name>Alice</first_name>"],
                COL_DETECTED_ENTITIES: [[{"value": "Alice", "label": "first_name"}]],
                COL_REPLACEMENT_MAP: [{"replacements": []}],
            }
        ),
        [],
    )
    anonymizer, _, _ = _make_anonymizer(replace_return=replace_return)

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
    replace_return = (
        pd.DataFrame(
            {
                COL_TEXT: ["Alice"],
                COL_REPLACED_TEXT: ["[REDACTED]"],
                COL_DETECTED_ENTITIES: [[{"value": "Alice", "label": "first_name"}]],
                COL_REPLACEMENT_MAP: [{"replacements": []}],
            }
        ),
        [],
    )
    anonymizer, _, _ = _make_anonymizer(replace_return=replace_return)

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
            COL_DETECTED_ENTITIES: [[{"value": "Alice", "label": "first_name"}]],
        }
    )
    replace_df.attrs["original_text_column"] = "bio"
    replace_return = (replace_df, [])
    anonymizer, _, _ = _make_anonymizer(replace_return=replace_return)

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
    anonymizer, _, _ = _make_anonymizer()

    with pytest.raises(InvalidInputError, match="reserved internal column"):
        anonymizer.run(
            config=stub_anonymizer_config,
            data=AnonymizerInput(source=str(input_csv), text_column="bio"),
        )
