# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
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
    COL_REPLACEMENT_APPLICATION,
    COL_REPLACEMENT_MAP,
    COL_TAGGED_TEXT,
    COL_TEXT,
)
from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN, NddAdapter
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow
from anonymizer.engine.replace.replace_runner import ReplacementResult, ReplacementWorkflow
from tests.engine.ndd.stub_provider import Script, stub_provider

_NARROW = "Acme\u202fCorp"
_THIN = "Acme\u2009Corp"
_PLAIN = "Acme Corp"


def _input(values: list[str]) -> pd.DataFrame:
    text = " | ".join(values)
    spans = []
    start = 0
    for value in values:
        spans.append(
            {"value": value, "label": "company_name", "start_position": start, "end_position": start + len(value)}
        )
        start += len(value) + len(" | ")
    return pd.DataFrame(
        {
            RECORD_ID_COLUMN: ["unicode-record"],
            COL_TEXT: [text],
            COL_TAGGED_TEXT: [text],
            COL_FINAL_ENTITIES: [{"entities": spans}],
            COL_ENTITIES_BY_VALUE: [
                {"entities_by_value": [{"value": value, "labels": ["company_name"]} for value in values]}
            ],
        }
    )


def _run(tmp_path: Path, dataframe: pd.DataFrame, script: Script, preview_num_records: int | None) -> ReplacementResult:
    with stub_provider(script) as provider:
        designer = DataDesigner(
            artifact_path=tmp_path / "artifacts", model_providers=[provider], auto_configure_logging=False
        )
        designer.set_run_config(
            RunConfig(max_conversation_restarts=0, max_conversation_correction_steps=0, otel_metrics_port=None)
        )
        return ReplacementWorkflow(llm_workflow=LlmReplaceWorkflow(adapter=NddAdapter(designer))).run(
            dataframe,
            replace_method=Substitute(),
            model_configs=[ModelConfig(alias="m", model="stub-model", provider="stub", skip_health_check=True)],
            selected_models=ReplaceModelSelection(replacement_generator="m"),
            preview_num_records=preview_num_records,
        )


def _requested_inventory(prompt: str) -> str:
    messages = json.loads(prompt)
    text = "\n".join(
        message["content"]
        if isinstance(message["content"], str)
        else "\n".join(part.get("text", "") for part in message["content"])
        for message in messages
    )
    return text.split("Entities to replace:", 1)[1].split("Examples:", 1)[0]


@pytest.mark.parametrize("preview_num_records", [None, 1], ids=["run", "preview"])
def test_real_adapter_recovers_unicode_original_without_retry_or_source_changes(
    tmp_path: Path, preview_num_records: int | None
) -> None:
    prompts: list[str] = []
    dataframe = _input([_NARROW])
    original_entities = json.dumps(dataframe[COL_FINAL_ENTITIES].iloc[0])

    def reply(_model: str, prompt: str, _earlier: int) -> tuple[int, str, str]:
        prompts.append(prompt)
        return (
            200,
            json.dumps({"replacements": [{"original": _THIN, "label": "company_name", "synthetic": "Nova Labs"}]}),
            "stop",
        )

    result = _run(tmp_path, dataframe, reply, preview_num_records)

    assert result.failed_records == []
    assert len(prompts) == 1
    assert result.dataframe[COL_REPLACED_TEXT].tolist() == ["Nova Labs"]
    assert result.dataframe[COL_REPLACEMENT_MAP].iloc[0] == {
        "replacements": [{"original": _NARROW, "label": "company_name", "synthetic": "Nova Labs"}]
    }
    assert result.dataframe[COL_REPLACEMENT_APPLICATION].iloc[0] == {
        "targeted_span_count": 1,
        "applied_span_count": 1,
        "skipped_span_count": 0,
        "skipped_span_label_counts": {},
    }
    assert dataframe[COL_TEXT].iloc[0] == _NARROW
    assert result.dataframe[COL_TEXT].iloc[0] == _NARROW
    assert json.dumps(dataframe[COL_FINAL_ENTITIES].iloc[0]) == original_entities
    assert result.dataframe[COL_FINAL_ENTITIES].iloc[0] == dataframe[COL_FINAL_ENTITIES].iloc[0]


@pytest.mark.parametrize("preview_num_records", [None, 1], ids=["run", "preview"])
@pytest.mark.parametrize("recover_on_retry", [True, False], ids=["exact-retry", "still-ambiguous"])
def test_real_adapter_does_not_guess_between_unicode_equivalent_requested_originals(
    tmp_path: Path, preview_num_records: int | None, recover_on_retry: bool
) -> None:
    prompts: list[str] = []
    dataframe = _input([_NARROW, _PLAIN])
    ambiguous = {"original": _THIN, "label": "company_name", "synthetic": "Unassigned Company"}
    exact = [
        {"original": _NARROW, "label": "company_name", "synthetic": "Nova Labs"},
        {"original": _PLAIN, "label": "company_name", "synthetic": "Summit Press"},
    ]

    def reply(_model: str, prompt: str, _earlier: int) -> tuple[int, str, str]:
        prompts.append(prompt)
        entries = exact if len(prompts) == 2 and recover_on_retry else [ambiguous]
        return 200, json.dumps({"replacements": entries}), "stop"

    result = _run(tmp_path, dataframe, reply, preview_num_records)

    assert len(prompts) == 2
    assert _NARROW in _requested_inventory(prompts[1])
    assert _PLAIN in _requested_inventory(prompts[1])
    assert dataframe[COL_TEXT].iloc[0] == f"{_NARROW} | {_PLAIN}"
    if recover_on_retry:
        assert result.failed_records == []
        assert result.dataframe[COL_REPLACED_TEXT].tolist() == ["Nova Labs | Summit Press"]
        assert result.dataframe[COL_REPLACEMENT_MAP].iloc[0] == {"replacements": exact}
    else:
        assert result.dataframe.empty
        assert len(result.failed_records) == 1
        failure = result.failed_records[0]
        assert failure.record_id == "unicode-record"
        assert failure.step == "replace-map-generation"
        for value in [_NARROW, _PLAIN, _THIN, "Unassigned Company"]:
            assert value not in failure.reason
