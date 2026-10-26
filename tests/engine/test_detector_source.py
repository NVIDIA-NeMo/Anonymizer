# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact source preservation across DataDesigner's JSON decoding and prompt rendering."""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from data_designer.config.column_configs import LLMTextColumnConfig
from data_designer.config.models import ModelConfig
from data_designer.config.run_config import RunConfig
from data_designer.engine.column_generators.generators.llm_completion import LLMTextCellGenerator
from data_designer.engine.processing.ginja.exceptions import EmptyTemplateRenderError

from anonymizer.config.models import DetectionModelSelection
from anonymizer.engine.constants import (
    COL_DETECTOR_SOURCE,
    COL_RAW_DETECTED,
    COL_SEED_DETECTION_RESULT,
    COL_TEXT,
    RECORD_ID_COLUMN,
)
from anonymizer.engine.detection.custom_columns import parse_detected_entities
from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
from anonymizer.engine.schemas import DetectionResult
from anonymizer.engine.workflow_columns.detection.config import DetectionTransformConfig
from anonymizer.engine.workflow_columns.detection.impl import DetectionTransformGenerator


@pytest.fixture
def detector_generators(
    stub_detector_model_configs: list[ModelConfig],
    stub_detection_model_selection: DetectionModelSelection,
) -> tuple[DetectionTransformGenerator, LLMTextCellGenerator, Mock]:
    workflow = EntityDetectionWorkflow(adapter=Mock())
    models, columns = workflow._build_detection_spec(
        model_configs=stub_detector_model_configs,
        selected_models=stub_detection_model_selection,
        gliner_detection_threshold=0.3,
    )
    source_config = cast(DetectionTransformConfig, next(c for c in columns if c.name == COL_DETECTOR_SOURCE))
    detector_config = cast(LLMTextColumnConfig, next(c for c in columns if c.name == COL_RAW_DETECTED))
    resources = Mock()
    resources.run_config = RunConfig()
    resources.model_registry.get_model_config.return_value = next(
        model for model in models if model.alias == detector_config.model_alias
    )
    model = resources.model_registry.get_model.return_value
    model.generate.return_value = ("[]", [])
    model.agenerate = AsyncMock(return_value=("[]", []))
    return (
        DetectionTransformGenerator(source_config, resource_provider=resources),
        LLMTextCellGenerator(detector_config, resource_provider=resources),
        model,
    )


@pytest.mark.parametrize(
    "text",
    [
        pytest.param('{\n  "name": "Alice",\n  "active": true\n}', id="formatted-object"),
        pytest.param('[ "Alice",  "Bob" ]', id="array"),
        pytest.param('"Alice"', id="json-string"),
        pytest.param("1.00", id="number"),
        pytest.param("true", id="boolean"),
        pytest.param("null", id="null"),
        pytest.param(' \n { "name" : "Alice" } \n\t', id="whitespace"),
        pytest.param('{"text":"line1\\nline2\\t\\"Alice\\""}', id="escaping"),
        pytest.param('{"姓名": "王小明", "city": "北京"}', id="unicode"),
        pytest.param('{"text": "{{ untouched }}"}', id="literal-jinja"),
        pytest.param("Alice lives in Boston.", id="prose"),
        pytest.param("", id="empty"),
        pytest.param("\r\nAlice\r\n", id="line-endings"),
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_detector_receives_exact_source(
    text: str,
    asynchronous: bool,
    detector_generators: tuple[DetectionTransformGenerator, LLMTextCellGenerator, Mock],
) -> None:
    prepare, detector, model = detector_generators
    row: dict[str, Any] = {COL_TEXT: text}
    prepare.generate(row)

    if not text:
        with pytest.raises(EmptyTemplateRenderError):
            if asynchronous:
                asyncio.run(detector.agenerate(row))
            else:
                detector.generate(row)
        assert row[COL_TEXT] == text
        model.generate.assert_not_called()
        model.agenerate.assert_not_called()
        return

    if asynchronous:
        result = asyncio.run(detector.agenerate(row))
        kwargs = model.agenerate.call_args.kwargs
    else:
        result = detector.generate(row)
        kwargs = model.generate.call_args.kwargs

    assert kwargs["prompt"] == text
    assert result[COL_TEXT] == text
    assert json.loads(result[COL_DETECTOR_SOURCE]) == {"text": text}
    assert result[COL_RAW_DETECTED] == "[]"


def test_json_source_detector_offsets_remain_canonical(
    detector_generators: tuple[DetectionTransformGenerator, LLMTextCellGenerator, Mock],
) -> None:
    prepare, detector, model = detector_generators
    text = json.dumps({"messages": [{"content": "王小明 met Alice in Boston."}]}, ensure_ascii=False, indent=2)
    start = text.index("Alice")
    model.generate.return_value = (
        json.dumps(
            {"entities": [{"text": "Alice", "label": "first_name", "start": start, "end": start + 5, "score": 0.9}]}
        ),
        [],
    )
    row: dict[str, Any] = {COL_TEXT: text, RECORD_ID_COLUMN: "synthetic"}
    prepare.generate(row)
    detector.generate(row)
    parse_detected_entities(row, detection_result_id="r1", source_instance="synthetic-detector")

    result = DetectionResult.model_validate(row[COL_SEED_DETECTION_RESULT])
    assert model.generate.call_args.kwargs["prompt"] == text
    assert len(result.observations) == 1
    observation = result.observations[0]
    assert (observation.start_position, observation.end_position) == (start, start + 5)
    assert text[observation.start_position : observation.end_position] == observation.value == "Alice"
