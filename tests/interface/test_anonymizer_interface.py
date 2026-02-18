# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from anonymizer.config.anonymizer_config import AnonymizerConfig
from anonymizer.config.replace_strategies import RedactReplace
from anonymizer.config.rewrite import RewriteParams
from anonymizer.engine.detection.constants import COL_DETECTED_ENTITIES, COL_TEXT
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult, EntityDetectionWorkflow
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.engine.replace.replace_runner import ReplaceRunner
from anonymizer.interface.anonymizer import Anonymizer, _resolve_model_providers


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
        pd.DataFrame({COL_TEXT: ["Alice works at Acme"], "replaced_text": ["[REDACTED] works at [REDACTED]"]}),
        [],
    )
    anonymizer = Anonymizer(detection_workflow=detection_workflow, replace_runner=replace_runner)
    return anonymizer, detection_workflow, replace_runner


def test_run_merges_failed_records_from_both_stages(stub_anonymizer_config: AnonymizerConfig) -> None:
    detection_failures = [FailedRecord(record_id="r1", step="detection", reason="timeout")]
    replace_failures = [FailedRecord(record_id="r2", step="replace", reason="parse error")]

    detection_result = EntityDetectionResult(
        dataframe=pd.DataFrame({COL_TEXT: ["Alice"], COL_DETECTED_ENTITIES: [[]]}),
        failed_records=detection_failures,
    )
    replace_return = (pd.DataFrame({COL_TEXT: ["Alice"], "replaced_text": ["Alice"]}), replace_failures)

    anonymizer, _, _ = _make_anonymizer(detection_return=detection_result, replace_return=replace_return)
    result = anonymizer.run(config=stub_anonymizer_config, data=pd.DataFrame({COL_TEXT: ["Alice"]}))

    assert len(result.failed_records) == 2
    assert result.failed_records[0].step == "detection"
    assert result.failed_records[1].step == "replace"


def test_run_enables_latent_detection_when_rewrite_configured() -> None:
    config = AnonymizerConfig(replace=RedactReplace(), rewrite=RewriteParams())
    anonymizer, detection_wf, _ = _make_anonymizer()
    anonymizer.run(config=config, data=pd.DataFrame({COL_TEXT: ["Alice"]}))

    assert detection_wf.run.call_args.kwargs["tag_latent_entities"] is True


def test_run_disables_latent_detection_without_rewrite(stub_anonymizer_config: AnonymizerConfig) -> None:
    anonymizer, detection_wf, _ = _make_anonymizer()
    anonymizer.run(config=stub_anonymizer_config, data=pd.DataFrame({COL_TEXT: ["Alice"]}))

    assert detection_wf.run.call_args.kwargs["tag_latent_entities"] is False


def test_resolve_model_providers_raises_on_invalid_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text("not_providers: []")

    with pytest.raises(ValueError, match="providers"):
        _resolve_model_providers(yaml_path)
