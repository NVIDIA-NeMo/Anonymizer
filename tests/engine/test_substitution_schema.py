# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Literal

import pytest
from data_designer.config.utils.storage_schema import compile_json_storage_schema
from data_designer.engine.models.parsers.errors import ParserException
from data_designer.engine.models.recipes.response_recipes import StructuredResponseRecipe
from pydantic import ValidationError

from anonymizer.engine.schemas.replace import EntityReplacementMapSchema

ValidationBackend = Literal["pydantic", "ndd"]


def _payload(fields: dict[str, str]) -> dict[str, list[dict[str, str]]]:
    return {
        "replacements": [
            {"original": "Alice", "label": "first_name", "synthetic": "Maya"},
            {"original": "Bob", "label": "first_name", **fields},
        ]
    }


def _validate(payload: dict[str, list[dict[str, str]]], backend: ValidationBackend) -> object:
    if backend == "pydantic":
        return EntityReplacementMapSchema.model_validate(payload).model_dump()
    schema, _ = compile_json_storage_schema(EntityReplacementMapSchema.model_json_schema())
    return StructuredResponseRecipe(json_schema=schema).parse(json.dumps(payload))


@pytest.mark.parametrize("backend", ["pydantic", "ndd"])
@pytest.mark.parametrize(
    "synthetic",
    [
        pytest.param("   ", id="spaces"),
        pytest.param("\t", id="tab"),
        pytest.param("\n", id="newline"),
        pytest.param("\r\n", id="carriage-return-newline"),
        pytest.param("\u00a0", id="no-break-space"),
        pytest.param("\u2009", id="thin-space"),
        pytest.param("\u202f", id="narrow-no-break-space"),
        pytest.param(" \t\r\n\u00a0\u2009\u202f", id="mixed-whitespace"),
    ],
)
def test_schema_rejects_whitespace_only_substitutes(synthetic: str, backend: ValidationBackend) -> None:
    with pytest.raises((ValidationError, ParserException)):
        _validate(_payload({"synthetic": synthetic}), backend)


@pytest.mark.parametrize("backend", ["pydantic", "ndd"])
@pytest.mark.parametrize(
    "fields", [pytest.param({}, id="missing-field"), pytest.param({"synthetic": ""}, id="empty-string")]
)
def test_schema_still_rejects_missing_and_empty_substitutes(fields: dict[str, str], backend: ValidationBackend) -> None:
    with pytest.raises((ValidationError, ParserException)):
        _validate(_payload(fields), backend)


@pytest.mark.parametrize("backend", ["pydantic", "ndd"])
@pytest.mark.parametrize("synthetic", ["N", "7", "林", " Maya ", "\tN\n", "\u202fM\u2009"])
def test_schema_preserves_nonblank_substitutes(synthetic: str, backend: ValidationBackend) -> None:
    payload = _payload({"synthetic": synthetic})

    assert _validate(payload, backend) == payload


def test_schema_identifies_the_bad_entry_in_a_mixed_map() -> None:
    with pytest.raises(ValidationError) as exc_info:
        EntityReplacementMapSchema.model_validate(_payload({"synthetic": " \t\u202f"}))

    assert [error["loc"] for error in exc_info.value.errors()] == [("replacements", 1, "synthetic")]


@pytest.mark.parametrize("backend", ["pydantic", "ndd"])
def test_schema_does_not_change_mapping_completeness_rules(backend: ValidationBackend) -> None:
    assert _validate({"replacements": []}, backend) == {"replacements": []}
