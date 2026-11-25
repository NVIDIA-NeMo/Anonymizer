# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One geometry rule for range records: ``geometry_problem`` and the ``Positioned`` mixin keep each type's messages,
fields and key order; ``EntitySchema`` stays lenient."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from anonymizer.engine.detection.postprocess import EntitySpan
from anonymizer.engine.schemas import EntitySchema
from anonymizer.engine.schemas.detection import SourceTextRef, SourceView, ValidationInventoryEntry
from anonymizer.engine.spans import Span, geometry_problem


@pytest.mark.parametrize(
    ("args", "empty", "message"),
    [
        ((3, 3, None), False, "x must cover a non-empty span"),
        ((3, 2, None), True, "x end must not precede its start"),
        ((3, 3, None), True, ""),
        ((0, 3, "ab"), False, "x value length must match its source bounds"),
        ((0, 2, "ab"), False, ""),
    ],
)
def test_geometry_problem(args: tuple, empty: bool, message: str) -> None:
    assert geometry_problem("x", *args, empty=empty) == message


def test_records_keep_their_messages_and_key_order() -> None:
    ref = SourceTextRef(record_id="r", field_id="f")
    with pytest.raises(ValidationError, match="source view end must not precede its start"):
        SourceView(source_text_ref=ref, start_position=3, end_position=2)
    assert SourceView(source_text_ref=ref, start_position=2, end_position=2).span == Span(2, 2)
    with pytest.raises(ValidationError, match="inventory entry value length must match its source bounds"):
        ValidationInventoryEntry(
            alias="c1",
            observation_ref={"detection_result_id": "d", "observation_id": "o"},
            value="abc",
            label="x",
            start_position=0,
            end_position=2,
        )
    assert list(SourceView(source_text_ref=ref, start_position=0, end_position=1).model_dump()) == [
        "source_text_ref",
        "start_position",
        "end_position",
    ]
    assert EntitySpan("e", "ab", "x", 1, 3, 0.0, "s").span == Span(1, 3)


def test_entity_schema_stays_lenient() -> None:
    entity = EntitySchema(value="ab", start_position=5, end_position=3)
    assert entity.span_in("abcdef") is None
    assert EntitySchema(start_position=1, end_position=3).span_in("abcdef") == Span(1, 3)
