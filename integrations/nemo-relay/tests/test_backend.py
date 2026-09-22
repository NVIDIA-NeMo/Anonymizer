# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from nemo_anonymizer_relay.backend import (
    AnonymizerBackend,
    BackendConfig,
    BackendResultError,
    RedactionSpan,
    _pack_texts,
    _spans_from_result,
    require_api_keys,
)


def backend_config() -> BackendConfig:
    return BackendConfig(
        detector_endpoint="http://127.0.0.1:8001/v1",
        detector_model="gliner",
        detector_api_key_env="EMPTY",
        evaluator_endpoint="https://example.invalid/v1",
        evaluator_model="evaluator",
        evaluator_api_key_env="MISSING_TEST_EVALUATOR_KEY",
        threshold=0.3,
        timeout_seconds=30,
        data_summary=None,
    )


def spans_from_result(dataframe: Any, packed: Any, text_count: int) -> list[list[RedactionSpan]]:
    return _spans_from_result(
        dataframe,
        packed,
        text_count,
        id_column="relay_batch_id",
        text_column="text",
    )


def result_frame(text: str, *entities: tuple[str, str, int, int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "relay_batch_id": "0",
                "text": text,
                "final_entities": {
                    "entities": [
                        {"value": value, "label": label, "start_position": start, "end_position": end}
                        for value, label, start, end in entities
                    ]
                },
            }
        ]
    )


def test_result_mapping_uses_explicit_segment_ids() -> None:
    result = result_frame("Ana\n\n[RELAY OBSERVABILITY FIELD]\n\nsafe", ("Ana", "first_name", 0, 3))

    decisions = spans_from_result(result, _pack_texts(["Ana", "safe"]), 2)

    assert decisions[0][0].label == "first_name"
    assert decisions[1] == []


def test_result_mapping_rejects_missing_rows() -> None:
    result = pd.DataFrame(
        {
            "relay_batch_id": pd.Series(dtype="object"),
            "text": pd.Series(dtype="object"),
            "final_entities": pd.Series(dtype="object"),
        }
    )
    with pytest.raises(RuntimeError, match="missing_input_batch"):
        spans_from_result(result, _pack_texts(["one", "two"]), 2)


def test_result_mapping_clips_cross_field_span_fail_closed() -> None:
    packed = _pack_texts(["Ana", "Bob"])
    result = result_frame(
        packed[0].text,
        (packed[0].text, "combined_identity", 0, len(packed[0].text)),
    )

    assert spans_from_result(result, packed, 2) == [
        [RedactionSpan(0, 3, "combined_identity")],
        [RedactionSpan(0, 3, "combined_identity")],
    ]


def test_result_mapping_accepts_anonymizer_trimmed_display_value() -> None:
    packed = _pack_texts([" Ana "])
    result = result_frame(packed[0].text, ("Ana", "first_name", 0, 5))

    assert spans_from_result(result, packed, 1) == [[RedactionSpan(0, 5, "first_name")]]


def test_result_mapping_rejects_inconsistent_detector_offsets() -> None:
    packed = _pack_texts(["Ana greeted Ana"])
    result = result_frame(packed[0].text, ("Ana", "first_name", 4, 11))

    with pytest.raises(RuntimeError, match="entity_value_mismatch"):
        spans_from_result(result, packed, 1)


def test_packing_splits_at_budget() -> None:
    packed = _pack_texts(["Ana", "safe", "Bob"], max_chars=30)

    assert [len(record.placements) for record in packed] == [1, 1, 1]
    assert [record.text for record in packed] == ["Ana", "safe", "Bob"]


def test_model_config_wires_detection_pipeline() -> None:
    payload = json.loads(AnonymizerBackend(backend_config())._model_config_json())

    assert [model["alias"] for model in payload["model_configs"]] == [
        "relay-gliner",
        "relay-evaluator",
    ]
    assert all(model["skip_health_check"] is True for model in payload["model_configs"])
    assert payload["selected_models"]["detection"] == {
        "entity_detector": "relay-gliner",
        "entity_validator": ["relay-evaluator"],
        "entity_augmenter": "relay-evaluator",
        "latent_detector": "relay-evaluator",
    }


def test_full_pipeline_requires_evaluator_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_TEST_EVALUATOR_KEY", raising=False)

    with pytest.raises(ValueError, match="MISSING_TEST_EVALUATOR_KEY"):
        require_api_keys(backend_config())


def test_backend_normalizes_provider_errors_before_worker_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    class Value:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

    class FailingAnonymizer(Value):
        def run(self, **_kwargs: Any) -> Any:
            raise RuntimeError("provider echoed Marisol Vega")

    monkeypatch.setitem(
        sys.modules,
        "anonymizer",
        SimpleNamespace(
            Anonymizer=FailingAnonymizer,
            AnonymizerConfig=Value,
            Detect=Value,
            ModelProvider=Value,
            Redact=Value,
            TextRecord=Value,
            TextRecordsInput=Value,
        ),
    )

    with pytest.raises(BackendResultError) as error:
        AnonymizerBackend(backend_config()).detect(["Marisol Vega"])

    assert str(error.value) == "anonymizer_runtime_failure"
