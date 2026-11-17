# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-32 declared detector facts and the resolved detection scope (E32.1-E32.2)."""

from __future__ import annotations

from unittest.mock import Mock

import pandas as pd

from anonymizer.config.anonymizer_config import AnonymizerConfig, Detect, TextRecord, TextRecordsInput
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import COL_FINAL_ENTITIES, COL_TEXT, DEFAULT_ENTITY_LABELS
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult, EntityDetectionWorkflow
from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.interface.errors import InvalidConfigError
from tests.spec_programs._builder import Checks


def program() -> list[dict[str, str]]:
    c = Checks("SP-32")
    data = TextRecordsInput(records=[TextRecord(id="r1", text="Alice")])

    # E32.1 an effective label outside a declared fixed label set is rejected before the detector runs.
    def e1() -> tuple[bool, str]:
        detector = Mock(spec=EntityDetectionWorkflow)
        detect = Detect(
            entity_labels=["first_name", "email"], detector_labels=["first_name"], detector_labels_fixed=True
        )
        try:
            Anonymizer(detection_workflow=detector).run(
                config=AnonymizerConfig(detect=detect, replace=Redact()), data=data
            )
        except InvalidConfigError:
            return detector.run.call_count == 0, f"detector calls={detector.run.call_count}"
        return False, "no InvalidConfigError"

    c.guard("E32.1", e1)

    # E32.2 run and preview expose the default resolved scope.
    def e2() -> tuple[bool, str]:
        detector = Mock(spec=EntityDetectionWorkflow)
        frame = pd.DataFrame({COL_TEXT: ["Alice"], COL_FINAL_ENTITIES: [{"entities": []}]})
        detector.run.return_value = EntityDetectionResult(dataframe=frame, failed_records=[])
        anonymizer = Anonymizer(detection_workflow=detector)
        config = AnonymizerConfig(replace=Redact())
        scopes = [anonymizer.run(config=config, data=data).detection_scope]
        scopes.append(anonymizer.preview(config=config, data=data, num_records=1).detection_scope)
        ok = all(
            s is not None
            and s.effective_labels == tuple(sorted(DEFAULT_ENTITY_LABELS))
            and s.excluded_labels == ()
            and s.gliner_threshold == 0.3
            and s.allowed_labels is None
            for s in scopes
        )
        return ok, f"scopes set={[s is not None for s in scopes]}"

    c.guard("E32.2", e2)
    return c.lines
