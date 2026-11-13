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
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow
from anonymizer.engine.replace.replace_runner import ReplacementResult, ReplacementWorkflow
from tests.engine.ndd.stub_provider import Script, stub_provider

_ALICE = {"original": "Alice", "label": "first_name", "synthetic": "Maya"}
_BOB = {"original": "Bob", "label": "first_name", "synthetic": "Noah"}
_CARA = {"original": "Cara", "label": "first_name", "synthetic": "Lena"}


def _inputs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            RECORD_ID_COLUMN: ["partial", "complete", "empty"],
            COL_TEXT: ["Alice sees Bob.", "Cara.", "No entities."],
            COL_TAGGED_TEXT: ["PARTIAL_RECORD Alice sees Bob.", "COMPLETE_RECORD Cara.", "No entities."],
            COL_FINAL_ENTITIES: [
                {
                    "entities": [
                        {"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5},
                        {"value": "Bob", "label": "first_name", "start_position": 11, "end_position": 14},
                    ]
                },
                {"entities": [{"value": "Cara", "label": "first_name", "start_position": 0, "end_position": 4}]},
                {"entities": []},
            ],
            COL_ENTITIES_BY_VALUE: [
                {"entities_by_value": [{"value": value, "labels": ["first_name"]} for value in values]}
                for values in [["Alice", "Bob"], ["Cara"], []]
            ],
        }
    )


def _prompt_text(prompt: str) -> str:
    messages = json.loads(prompt)
    return "\n".join(
        message["content"]
        if isinstance(message["content"], str)
        else "\n".join(part.get("text", "") for part in message["content"])
        for message in messages
    )


def _requested_inventory(prompt: str) -> str:
    return _prompt_text(prompt).split("Entities to replace:", 1)[1].split("Examples:", 1)[0]


def _run(tmp_path: Path, script: Script, preview_num_records: int | None) -> ReplacementResult:
    with stub_provider(script) as provider:
        designer = DataDesigner(
            artifact_path=tmp_path / "artifacts", model_providers=[provider], auto_configure_logging=False
        )
        designer.set_run_config(
            RunConfig(max_conversation_restarts=0, max_conversation_correction_steps=0, otel_metrics_port=None)
        )
        return ReplacementWorkflow(llm_workflow=LlmReplaceWorkflow(adapter=NddAdapter(designer))).run(
            _inputs(),
            replace_method=Substitute(),
            model_configs=[ModelConfig(alias="m", model="stub-model", provider="stub", skip_health_check=True)],
            selected_models=ReplaceModelSelection(replacement_generator="m"),
            preview_num_records=preview_num_records,
        )


@pytest.mark.parametrize("preview_num_records", [None, 3], ids=["run", "preview"])
@pytest.mark.parametrize(
    "extra_entry",
    [
        pytest.param(None, id="omitted"),
        pytest.param({**_BOB, "original": "bob"}, id="changed-original"),
        pytest.param({**_BOB, "label": "name"}, id="changed-label"),
    ],
)
def test_real_adapter_retries_only_missing_pairs_and_keeps_accepted_values(
    tmp_path: Path, preview_num_records: int | None, extra_entry: dict[str, str] | None
) -> None:
    partial_prompts: list[str] = []
    complete_requests = 0

    def reply(_model: str, prompt: str, _earlier: int) -> tuple[int, str, str]:
        nonlocal complete_requests
        text = _prompt_text(prompt)
        if "COMPLETE_RECORD" in text:
            complete_requests += 1
            entries = [_CARA]
        else:
            partial_prompts.append(prompt)
            entries = (
                [_BOB, {**_ALICE, "synthetic": "Changed"}]
                if len(partial_prompts) == 2
                else [_ALICE, *([extra_entry] if extra_entry is not None else [])]
            )
        return 200, json.dumps({"replacements": entries}), "stop"

    result = _run(tmp_path, reply, preview_num_records)

    assert result.failed_records == []
    assert result.dataframe[RECORD_ID_COLUMN].tolist() == ["partial", "complete", "empty"]
    assert result.dataframe[COL_REPLACED_TEXT].tolist() == ["Maya sees Noah.", "Lena.", "No entities."]
    assert len(partial_prompts) == 2
    assert complete_requests == 1
    assert '"Alice"' in _requested_inventory(partial_prompts[0])
    assert '"Bob"' in _requested_inventory(partial_prompts[0])
    assert '"Alice"' not in _requested_inventory(partial_prompts[1])
    assert '"Bob"' in _requested_inventory(partial_prompts[1])
    assert result.dataframe[COL_REPLACEMENT_MAP].iloc[0] == {"replacements": [_ALICE, _BOB]}


@pytest.mark.parametrize("preview_num_records", [None, 3], ids=["run", "preview"])
@pytest.mark.parametrize("invalid_retry", [False, True], ids=["still-missing", "schema-invalid"])
def test_real_adapter_fails_only_the_record_that_cannot_complete_its_retry(
    tmp_path: Path, preview_num_records: int | None, invalid_retry: bool
) -> None:
    requests: Counter[str] = Counter()

    def reply(_model: str, prompt: str, _earlier: int) -> tuple[int, str, str]:
        record = "complete" if "COMPLETE_RECORD" in _prompt_text(prompt) else "partial"
        requests[record] += 1
        if record == "complete":
            entries = [_CARA]
        elif requests[record] == 2 and invalid_retry:
            entries = [{"original": "Bob", "label": "first_name", "synthetic": ""}]
        else:
            entries = [_ALICE]
        return 200, json.dumps({"replacements": entries}), "stop"

    result = _run(tmp_path, reply, preview_num_records)

    assert result.dataframe[RECORD_ID_COLUMN].tolist() == ["complete", "empty"]
    assert result.dataframe[COL_REPLACED_TEXT].tolist() == ["Lena.", "No entities."]
    assert requests == {"partial": 2, "complete": 1}
    assert len(result.failed_records) == 1
    failure = result.failed_records[0]
    assert failure.record_id == "partial"
    assert failure.step == "replace-map-generation"
    for value in ["Alice", "Bob", "Maya", "Noah", "Cara", "Lena"]:
        assert value not in failure.reason
