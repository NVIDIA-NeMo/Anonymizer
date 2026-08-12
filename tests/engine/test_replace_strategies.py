# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock, patch

import pandas as pd

from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import (
    COL_REPLACED_TEXT,
    COL_REPLACEMENT_APPLICATION,
    COL_REPLACEMENT_MAP,
)
from anonymizer.engine.replace.strategies import apply_local_replace_strategy, apply_replacement_map


def test_apply_replacement_map_preserves_row_order_and_custom_columns() -> None:
    dataframe = pd.DataFrame(
        {
            "source": ["Bob", "Alice"],
            "detected": [
                {"entities": [{"value": "Bob", "label": "first_name", "start_position": 0, "end_position": 3}]},
                {"entities": [{"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5}]},
            ],
            "maps": [
                {"replacements": [{"original": "Bob", "label": "first_name", "synthetic": "Robert"}]},
                {"replacements": [{"original": "Alice", "label": "first_name", "synthetic": "Maria"}]},
            ],
        },
        index=pd.Index([9, 3]),
    )

    result = apply_replacement_map(
        dataframe,
        text_column="source",
        entities_column="detected",
        replacement_map_column="maps",
    )

    assert result.index.tolist() == [9, 3]
    assert result[COL_REPLACED_TEXT].tolist() == ["Robert", "Maria"]
    assert [metrics["applied_span_count"] for metrics in result[COL_REPLACEMENT_APPLICATION]] == [1, 1]


def test_apply_local_replace_strategy_records_progress_for_each_entity_value() -> None:
    dataframe = pd.DataFrame(
        {
            "source": ["Alice", "No PII"],
            "detected": [
                {"entities": [{"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5}]},
                {"entities": []},
            ],
        }
    )
    tracker = Mock()

    with patch("anonymizer.engine.replace.strategies.ProgressTracker", return_value=tracker):
        result = apply_local_replace_strategy(
            dataframe,
            strategy=Redact(format_template="***"),
            text_column="source",
            entities_column="detected",
        )

    assert tracker.record_success.call_count == len(dataframe)
    assert tracker.log_final.call_count == 1
    assert result[COL_REPLACED_TEXT].tolist() == ["***", "No PII"]
    assert result[COL_REPLACEMENT_MAP].iloc[0]["replacements"][0]["synthetic"] == "***"
