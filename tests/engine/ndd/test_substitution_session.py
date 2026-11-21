# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The completeness retry under a stable Session, through the real adapter, prompt and parser."""

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
    COL_TAGGED_TEXT,
    COL_TEXT,
)
from anonymizer.engine.graph_core.session import Session, SessionChanges
from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN, NddAdapter
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow
from anonymizer.engine.replace.replace_runner import ReplacementResult, ReplacementWorkflow
from tests.engine.ndd.stub_provider import stub_provider

_NAMES = ("Alice", "Bob", "Cara", "Dave", "Eve")
_RETRY = "This is the one retry for missing mappings."


def _frame(texts: list[str]) -> pd.DataFrame:
    entities = [
        [
            {
                "value": name,
                "label": "first_name",
                "start_position": t.index(name),
                "end_position": t.index(name) + len(name),
            }
            for name in _NAMES
            if name in t
        ]
        for t in texts
    ]
    return pd.DataFrame(
        {
            RECORD_ID_COLUMN: [f"r{i}" for i in range(len(texts))],
            COL_TEXT: texts,
            COL_TAGGED_TEXT: texts,
            COL_FINAL_ENTITIES: [{"entities": e} for e in entities],
            COL_ENTITIES_BY_VALUE: [
                {"entities_by_value": [{"value": s["value"], "labels": ["first_name"]} for s in e]} for e in entities
            ],
        }
    )


def _prompt_text(prompt: str) -> str:
    return "\n".join(
        m["content"] if isinstance(m["content"], str) else "\n".join(p.get("text", "") for p in m["content"])
        for m in json.loads(prompt)
    )


def _inventory(text: str) -> str:
    return text.split("Entities to replace:", 1)[1].split("Examples:", 1)[0]


def _run(
    tmp_path: Path,
    texts: list[str],
    answers: list[dict[str, str]],
    session: Session,
    prompts: list[str],
    preview_num_records: int | None = None,
) -> ReplacementResult:
    """One Substitute run; model request ``i`` answers the requested values found in ``answers[i]``."""

    def reply(_model: str, prompt: str, _earlier: int) -> tuple[int, str, str]:
        text = _prompt_text(prompt)
        prompts.append(text)
        answer = answers[len(prompts) - 1]
        inventory = _inventory(text)
        entries = [
            {"original": value, "label": "first_name", "synthetic": synthetic}
            for value, synthetic in answer.items()
            if f'"{value}"' in inventory
        ]
        return 200, json.dumps({"replacements": entries}), "stop"

    with stub_provider(reply) as provider:
        designer = DataDesigner(
            artifact_path=tmp_path / f"artifacts-{len(list(tmp_path.iterdir()))}",
            model_providers=[provider],
            auto_configure_logging=False,
        )
        designer.set_run_config(
            RunConfig(max_conversation_restarts=0, max_conversation_correction_steps=0, otel_metrics_port=None)
        )
        result = ReplacementWorkflow(llm_workflow=LlmReplaceWorkflow(adapter=NddAdapter(designer))).run(
            _frame(texts),
            replace_method=Substitute(),
            model_configs=[ModelConfig(alias="m", model="stub-model", provider="stub", skip_health_check=True)],
            selected_models=ReplaceModelSelection(replacement_generator="m"),
            preview_num_records=preview_num_records,
            session=session,
        )
    assert len(prompts) == len(answers)
    return result


def _slots(session: Session) -> dict[str, str]:
    slots = json.loads(session.snapshot().to_bytes())["slots"]
    return {s["value"]: s["assignment"] for s in slots if s["subject"] == "by-value"}


def _held_session(tmp_path: Path) -> Session:
    session = Session(stable_replacements=True)
    _run(tmp_path, ["Alice sees Bob."], [{"Alice": "Maya", "Bob": "Noah"}], session, [])
    return session


@pytest.mark.parametrize("preview_num_records", [None, 1], ids=["run", "preview"])
def test_real_adapter_retries_only_the_pair_the_session_does_not_hold(
    tmp_path: Path, preview_num_records: int | None
) -> None:
    session, prompts = _held_session(tmp_path), []
    texts = ["Alice meets Cara and Dave."] + (["Eve."] if preview_num_records else [])
    # The first response omits Alice (held) and Dave (new); the retry answers Dave.
    answers = [{"Cara": "Lena", "Eve": "Ivy"}, {"Dave": "Omar"}]
    result = _run(tmp_path, texts, answers, session, prompts, preview_num_records)

    assert len(prompts) == 2
    assert _RETRY not in prompts[0] and _RETRY in prompts[1]
    assert '"Alice"' in _inventory(prompts[0])
    assert '"Alice"' not in _inventory(prompts[1]) and '"Dave"' in _inventory(prompts[1])
    retry_block = prompts[1].split(_RETRY, 1)[1]
    assert "{'original': 'Alice', 'label': 'first_name', 'synthetic': 'Maya'}" in retry_block
    assert all('"Eve"' not in _inventory(prompt) for prompt in prompts)
    assert result.failed_records == []
    assert result.dataframe[COL_REPLACED_TEXT].tolist()[0] == "Maya meets Lena and Omar."
    assert result.session_changes == SessionChanges(accepted=2, reused=1)
    assert _slots(session) == {"alice": "Maya", "bob": "Noah", "cara": "Lena", "dave": "Omar"}


def test_real_adapter_retry_cannot_take_a_substitute_the_session_holds(tmp_path: Path) -> None:
    session, prompts = _held_session(tmp_path), []
    # The retry proposes Maya, which the Session holds for Alice.
    result = _run(tmp_path, ["Cara meets Dave."], [{"Cara": "Lena"}, {"Dave": "Maya"}], session, prompts)

    assert len(prompts) == 2
    assert result.dataframe[COL_REPLACED_TEXT].tolist() == ["Lena meets [SUBSTITUTE_FIRST_NAME_1]."]
    assert result.placeholder_counts == {"session": 1}
    assert _slots(session)["alice"] == "Maya"
