# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``run`` and ``preview`` expose the default resolved detection scope."""

from __future__ import annotations

from unittest.mock import Mock

import pandas as pd

from anonymizer.config.anonymizer_config import AnonymizerConfig, TextRecord, TextRecordsInput
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import COL_FINAL_ENTITIES, COL_TEXT, DEFAULT_ENTITY_LABELS
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult, EntityDetectionWorkflow
from anonymizer.interface.anonymizer import Anonymizer
from tests.detector_results import returns_detector_result


def test_run_and_preview_expose_the_default_detection_scope() -> None:
    data = TextRecordsInput(records=[TextRecord(id="r1", text="Alice")])
    detector = returns_detector_result(Mock(spec=EntityDetectionWorkflow))
    frame = pd.DataFrame({COL_TEXT: ["Alice"], COL_FINAL_ENTITIES: [{"entities": []}]})
    detector.run.return_value = EntityDetectionResult(dataframe=frame, failed_records=[])
    anonymizer = Anonymizer(detection_workflow=detector)
    config = AnonymizerConfig(replace=Redact())
    scopes = [
        anonymizer.run(config=config, data=data).detection_scope,
        anonymizer.preview(config=config, data=data, num_records=1).detection_scope,
    ]
    for scope in scopes:
        assert scope is not None
        assert scope.effective_labels == tuple(sorted(DEFAULT_ENTITY_LABELS))
        assert scope.excluded_labels == ()
        assert scope.gliner_threshold == 0.3
        assert scope.allowed_labels is None
