# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from anonymizer.engine.constants import COL_ENTITIES_BY_VALUE, COL_REPLACED_TEXT, COL_TEXT, RECORD_ID_COLUMN
from anonymizer.engine.evaluation.replace.privacy_leakage import prepare_privacy_evaluation_inputs
from anonymizer.engine.schemas import EntitiesByValueSchema


@pytest.mark.parametrize("representation", ["dict", "json", "array", "schema"])
def test_prepare_preserves_original_targets_and_source(representation: str) -> None:
    entities = [{"value": "Alice", "labels": ["first_name", "full_name"]}]
    payload: object = {"entities_by_value": entities}
    if representation == "json":
        payload = json.dumps(payload)
    elif representation == "array":
        payload = {"entities_by_value": np.array(entities, dtype=object)}
    elif representation == "schema":
        payload = EntitiesByValueSchema.model_validate(payload)
    source = pd.DataFrame(
        {
            RECORD_ID_COLUMN: ["record-b", "record-a"],
            COL_TEXT: ["Alice arrived.", "Nothing sensitive."],
            COL_REPLACED_TEXT: ["Maya arrived.", "Nothing sensitive."],
            COL_ENTITIES_BY_VALUE: [payload, {"entities_by_value": []}],
        },
        index=pd.Index([7, 2]),
    )
    prepared = prepare_privacy_evaluation_inputs(source)
    pd.testing.assert_frame_equal(
        prepared.drop(columns=COL_ENTITIES_BY_VALUE), source.drop(columns=COL_ENTITIES_BY_VALUE)
    )
    assert prepared[COL_ENTITIES_BY_VALUE].tolist() == [
        {"entities_by_value": entities},
        {"entities_by_value": []},
    ]
    assert source.at[7, COL_ENTITIES_BY_VALUE] is payload
    prepared.at[7, COL_ENTITIES_BY_VALUE]["entities_by_value"][0]["labels"].append("changed")
    assert EntitiesByValueSchema.from_raw(payload).entities_by_value[0].labels == ["first_name", "full_name"]


@pytest.mark.parametrize("missing", [RECORD_ID_COLUMN, COL_TEXT, COL_REPLACED_TEXT, COL_ENTITIES_BY_VALUE])
def test_prepare_requires_input_columns(missing: str) -> None:
    source = pd.DataFrame(columns=pd.Index([RECORD_ID_COLUMN, COL_TEXT, COL_REPLACED_TEXT, COL_ENTITIES_BY_VALUE]))
    with pytest.raises(ValueError, match="missing:") as error:
        prepare_privacy_evaluation_inputs(source.drop(columns=missing))
    assert missing in str(error.value)


def test_prepare_rejects_malformed_entities() -> None:
    source = pd.DataFrame(
        {
            RECORD_ID_COLUMN: ["record-a"],
            COL_TEXT: ["Alice arrived."],
            COL_REPLACED_TEXT: ["Maya arrived."],
            COL_ENTITIES_BY_VALUE: ["not valid JSON"],
        }
    )
    with pytest.raises(ValueError, match="not valid JSON"):
        prepare_privacy_evaluation_inputs(source)
