# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, Rewrite
from anonymizer.engine.constants import COL_ENTITIES_BY_VALUE, COL_REWRITTEN_TEXT, RECORD_ID_COLUMN
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult, EntityDetectionWorkflow
from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow
from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.interface.errors import InvalidInputError
from tests.engine.test_rewrite_workflow import _PRIVACY_GOAL, _REPLACE_PATCH, _ScriptedAdapter


@pytest.mark.parametrize("method", ["run", "preview"])
def test_input_column_named_like_the_rewritten_text_is_rejected(method: str, tmp_path: Path) -> None:
    """An input column named ``_rewritten_text`` used to be released as the rewritten text, reported ``met``."""
    entities = {"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]}
    detector = Mock(spec=EntityDetectionWorkflow)
    detector.run.side_effect = lambda frame, **_: EntityDetectionResult(
        dataframe=frame.assign(**{RECORD_ID_COLUMN: ["rec"], COL_ENTITIES_BY_VALUE: [entities]}),
        failed_records=[],
        source_positions=(0,),
    )
    csv_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Alice works at Acme"], COL_REWRITTEN_TEXT: ["note"]}).to_csv(csv_path, index=False)
    adapter = _ScriptedAdapter({"rec": 0})
    anonymizer = Anonymizer(detection_workflow=detector, rewrite_runner=RewriteWorkflow(adapter=Mock(wraps=adapter)))

    with patch(_REPLACE_PATCH) as mock_replace_cls, pytest.raises(InvalidInputError, match=COL_REWRITTEN_TEXT):
        mock_replace_cls.return_value.generate_map_only.side_effect = lambda frame, **_: Mock(
            dataframe=frame, failed_records=[], source_positions=tuple(range(len(frame)))
        )
        getattr(anonymizer, method)(
            config=AnonymizerConfig(
                rewrite=Rewrite(privacy_goal=_PRIVACY_GOAL, risk_tolerance="minimal"), emit_telemetry=False
            ),
            data=AnonymizerInput(source=str(csv_path), text_column="text"),
        )

    detector.run.assert_not_called()
    assert adapter.workflow_names == []
