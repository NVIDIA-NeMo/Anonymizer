# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nested values leave the library as plain JSON keyed by names, with None for NaN and for an empty object."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from anonymizer.engine.cells import is_missing, plain
from anonymizer.engine.constants import COL_ENTITY_CLASSIFICATION, COL_FINAL_ENTITIES, COL_REPLACEMENT_APPLICATION
from anonymizer.engine.io.output_shape import from_output_shape, to_output_shape


def test_plain_gives_json_types_and_none_for_nan_and_empty_objects() -> None:
    value = {"a": np.int64(2), "b": np.array([1.5, float("nan")]), "c": {}, "d": ({"e": pd.NA},)}
    assert plain(value) == {"a": 2, "b": [1.5, None], "c": None, "d": [{"e": None}]}
    assert type(plain(value)["a"]) is int
    assert plain({}) is None
    assert plain(float("nan")) is None


def test_label_keyed_values_become_lists_of_records_and_come_back() -> None:
    application = {"applied_span_count": 1, "skipped_span_label_counts": json.dumps({"city": 2})}
    frame = pd.DataFrame(
        {
            COL_ENTITY_CLASSIFICATION: [{"first_name": "direct_identifier"}, {}],
            COL_REPLACEMENT_APPLICATION: [application, {"applied_span_count": 0, "skipped_span_label_counts": "{}"}],
            COL_FINAL_ENTITIES: [{"entities": []}, float("nan")],
            "passthrough": [{}, float("nan")],
            "_note": [{}, float("nan")],
            "_privacy_qa__iteration_notes": [[1.0, float("nan")], {}],
        }
    )
    shaped = to_output_shape(frame)
    assert shaped[COL_ENTITY_CLASSIFICATION].tolist() == [
        [{"label": "first_name", "classification": "direct_identifier"}],
        None,
    ]
    assert shaped[COL_REPLACEMENT_APPLICATION][0]["skipped_span_label_counts"] == [{"label": "city", "count": 2}]
    assert shaped[COL_REPLACEMENT_APPLICATION][1]["skipped_span_label_counts"] is None
    assert shaped[COL_FINAL_ENTITIES].tolist() == [{"entities": []}, None]
    # The caller's own columns are not reshaped, even one named like an internal column.
    assert shaped["passthrough"][0] == {}
    assert shaped["_note"][0] == {}
    assert shaped["_note"].isna()[1]
    kept = shaped["_privacy_qa__iteration_notes"]
    assert kept[0][1] != kept[0][1]  # NaN kept: a round-like name without a round number is the caller's column
    assert kept[1] == {}
    back = from_output_shape(shaped)
    assert back[COL_ENTITY_CLASSIFICATION][0] == {"first_name": "direct_identifier"}
    assert back[COL_REPLACEMENT_APPLICATION][0] == application
    assert back[COL_REPLACEMENT_APPLICATION][1]["skipped_span_label_counts"] == "{}"


def test_combined_rewrite_round_columns_are_shaped() -> None:
    shaped = to_output_shape(pd.DataFrame({"_privacy_qa__iteration_0": [{}], "_privacy_qa__iteration_12": [{}]}))
    assert shaped.iloc[0].tolist() == [None, None]


def test_is_missing_names_null_markers_and_not_empty_containers() -> None:
    assert [is_missing(v) for v in (None, float("nan"), pd.NA)] == [True, True, True]
    assert [is_missing(v) for v in ({}, [], "", 0)] == [False, False, False, False]
