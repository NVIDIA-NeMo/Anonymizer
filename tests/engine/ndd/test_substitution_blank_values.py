# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest
from data_designer.config.models import ModelConfig
from data_designer.config.run_config import RunConfig
from data_designer.interface.data_designer import DataDesigner

from anonymizer.config.models import ReplaceModelSelection
from anonymizer.config.replace_strategies import Substitute
from anonymizer.engine.constants import (
    COL_ENTITIES_BY_VALUE,
    COL_FINAL_ENTITIES,
    COL_REPLACED_TEXT,
    COL_REPLACEMENT_MAP,
    COL_TAGGED_TEXT,
    COL_TEXT,
)
from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN, NddAdapter
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceResult, LlmReplaceWorkflow
from anonymizer.engine.replace.replace_runner import ReplacementResult, ReplacementWorkflow
from tests.engine.ndd.stub_provider import Script, stub_provider


def _inputs() -> pd.DataFrame:
    values = ["Alice", "Bob"]
    return pd.DataFrame(
        {
            RECORD_ID_COLUMN: ["bad", "good"],
            COL_TEXT: values,
            COL_TAGGED_TEXT: ["BAD_RECORD Alice", "GOOD_RECORD Bob"],
            COL_FINAL_ENTITIES: [
                {
                    "entities": [
                        {
                            "value": value,
                            "label": "first_name",
                            "start_position": 0,
                            "end_position": len(value),
                        }
                    ]
                }
                for value in values
            ],
            COL_ENTITIES_BY_VALUE: [
                {"entities_by_value": [{"value": value, "labels": ["first_name"]}]} for value in values
            ],
        }
    )


def _record(prompt: str) -> str:
    return "bad" if "BAD_RECORD" in prompt else "good"


def _response(record: str, synthetic: str) -> tuple[int, str, str]:
    return (
        200,
        json.dumps(
            {
                "replacements": [
                    {
                        "original": "Alice" if record == "bad" else "Bob",
                        "label": "first_name",
                        "synthetic": synthetic,
                    }
                ]
            }
        ),
        "stop",
    )


def _run(
    tmp_path: Path,
    script: Script,
    preview_num_records: int | None,
    *,
    schema_restarts: int = 0,
    map_only: bool = False,
) -> LlmReplaceResult | ReplacementResult:
    with stub_provider(script) as provider:
        designer = DataDesigner(
            artifact_path=tmp_path / "artifacts", model_providers=[provider], auto_configure_logging=False
        )
        designer.set_run_config(
            RunConfig(
                max_conversation_restarts=schema_restarts,
                max_conversation_correction_steps=0,
                otel_metrics_port=None,
            )
        )
        workflow = LlmReplaceWorkflow(adapter=NddAdapter(designer))
        models = [ModelConfig(alias="m", model="stub-model", provider="stub", skip_health_check=True)]
        selection = ReplaceModelSelection(replacement_generator="m")
        if map_only:
            return workflow.generate_map_only(
                _inputs(),
                model_configs=models,
                selected_models=selection,
                preview_num_records=preview_num_records,
            )
        return ReplacementWorkflow(llm_workflow=workflow).run(
            _inputs(),
            replace_method=Substitute(),
            model_configs=models,
            selected_models=selection,
            preview_num_records=preview_num_records,
        )


@pytest.mark.parametrize("preview_num_records", [None, 2], ids=["run", "preview"])
@pytest.mark.parametrize("blank", ["   ", "\t", "\n", " \t\n\u00a0\u2009\u202f"])
def test_blank_substitute_is_schema_rejected_while_valid_record_survives(
    tmp_path: Path, preview_num_records: int | None, blank: str
) -> None:
    requests: Counter[str] = Counter()

    def reply(_model: str, prompt: str, _earlier: int) -> tuple[int, str, str]:
        record = _record(prompt)
        requests[record] += 1
        return _response(record, blank if record == "bad" else "N")

    result = _run(tmp_path, reply, preview_num_records)

    assert result.dataframe[RECORD_ID_COLUMN].tolist() == ["good"]
    assert result.dataframe[COL_REPLACED_TEXT].tolist() == ["N"]
    assert requests == {"bad": 1, "good": 1}
    assert len(result.failed_records) == 1
    failure = result.failed_records[0]
    assert (failure.record_id, failure.step) == ("bad", "replace-map-generation")
    assert "could not be parsed" in failure.reason
    for value in ["Alice", "Bob"]:
        assert value not in failure.reason


@pytest.mark.parametrize("preview_num_records", [None, 2], ids=["run", "preview"])
def test_blank_substitute_uses_existing_schema_restart_before_completeness_check(
    tmp_path: Path, preview_num_records: int | None
) -> None:
    requests: Counter[str] = Counter()
    bad_prompts: list[str] = []

    def reply(_model: str, prompt: str, _earlier: int) -> tuple[int, str, str]:
        record = _record(prompt)
        requests[record] += 1
        if record == "bad":
            bad_prompts.append(prompt)
            return _response(record, "\t" if requests[record] == 1 else "Maya")
        return _response(record, "N")

    result = _run(tmp_path, reply, preview_num_records, schema_restarts=1)

    assert result.failed_records == []
    assert result.dataframe[RECORD_ID_COLUMN].tolist() == ["bad", "good"]
    assert result.dataframe[COL_REPLACED_TEXT].tolist() == ["Maya", "N"]
    assert requests == {"bad": 2, "good": 1}
    assert len(bad_prompts) == 2 and bad_prompts[0] == bad_prompts[1]


@pytest.mark.parametrize("preview_num_records", [None, 2], ids=["run", "preview"])
@pytest.mark.parametrize("synthetic", ["N", " Maya "])
def test_nonblank_substitute_is_preserved_without_another_call(
    tmp_path: Path, preview_num_records: int | None, synthetic: str
) -> None:
    requests: Counter[str] = Counter()

    def reply(_model: str, prompt: str, _earlier: int) -> tuple[int, str, str]:
        record = _record(prompt)
        requests[record] += 1
        return _response(record, synthetic)

    result = _run(tmp_path, reply, preview_num_records)

    assert result.failed_records == []
    assert requests == {"bad": 1, "good": 1}
    assert result.dataframe[COL_REPLACED_TEXT].tolist() == [synthetic, synthetic]
    assert [entry["replacements"][0]["synthetic"] for entry in result.dataframe[COL_REPLACEMENT_MAP]] == [
        synthetic,
        synthetic,
    ]


@pytest.mark.parametrize("preview_num_records", [None, 2], ids=["run", "preview"])
def test_shared_rewrite_map_generation_rejects_blank_without_completeness_retry(
    tmp_path: Path, preview_num_records: int | None
) -> None:
    requests: Counter[str] = Counter()

    def reply(_model: str, prompt: str, _earlier: int) -> tuple[int, str, str]:
        record = _record(prompt)
        requests[record] += 1
        return _response(record, "\u202f\u2009" if record == "bad" else "N")

    result = _run(tmp_path, reply, preview_num_records, map_only=True)

    assert result.dataframe[RECORD_ID_COLUMN].tolist() == ["good"]
    assert requests == {"bad": 1, "good": 1}
    assert [failure.record_id for failure in result.failed_records] == ["bad"]
    assert result.dataframe[COL_REPLACEMENT_MAP].iloc[0]["replacements"][0]["synthetic"] == "N"
