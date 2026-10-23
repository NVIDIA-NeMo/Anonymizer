# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import (
    COL_REPLACED_TEXT,
    COL_REPLACEMENT_MAP,
)
from anonymizer.engine.detection.postprocess import EntitySpan
from anonymizer.engine.replace.strategies import (
    ReplacementEntry,
    apply_entity_spans,
    apply_local_replace_strategy,
    apply_replacements_to_spans,
)
from anonymizer.engine.schemas import EntitiesSchema, EntitySchema


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

    with patch("anonymizer.engine.replace.route.ProgressTracker", return_value=tracker):
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


@pytest.mark.parametrize("fallback", [False, True])
def test_typed_and_legacy_rendering_share_overlap_and_fallback_policy(fallback: bool) -> None:
    text = "Alice met Bob."
    spans = (
        EntitySpan("a", "Alice", "first_name", 0, 5, 0.9, "detector"),
        EntitySpan("overlap", "Alice met", "person", 0, 9, 0.8, "detector"),
        EntitySpan("b", "Bob", "last_name", 10, 13, 0.7, "detector"),
    )
    replacements = [ReplacementEntry("Alice", "first_name", "[A]"), ReplacementEntry("Bob", "", "[B]")]
    legacy = EntitiesSchema(entities=[EntitySchema.model_validate(span.as_dict()) for span in spans])
    actual = apply_entity_spans(text, spans, replacements, allow_value_fallback=fallback)
    assert actual == apply_replacements_to_spans(text, legacy, replacements, allow_value_fallback=fallback)
    assert actual[0] == ("[A] met [B]." if fallback else "[A] met Bob.")
    assert actual[1].targeted_span_count == 3
    assert actual[1].applied_span_count == (2 if fallback else 1)
    assert actual[1].skipped_span_count == (1 if fallback else 2)
