# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A placeholder minted for a label stays as minted, including when the label's upper-case form adds combining marks
or changes length (``ǰ`` gives ``J`` and a mark): ``_is_placeholder`` accepts every token ``_collision_safe_synthetic``
can mint, so no later spelling of the value re-cases it."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast
from unittest.mock import patch

import pandas as pd
import pytest

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput
from anonymizer.config.replace_strategies import Substitute
from anonymizer.engine.constants import COL_ENTITIES_BY_VALUE, COL_FINAL_ENTITIES, COL_REPLACEMENT_MAP, COL_TEXT
from anonymizer.engine.detection.detection_workflow import (
    EntityDetectionResult,
    EntityDetectionWorkflow,
    _build_entities_by_value,
)
from anonymizer.engine.graph_core.session import Session
from anonymizer.engine.ndd.adapter import NddAdapter, WorkflowRunResult
from anonymizer.engine.replace.llm_replace_workflow import _collision_safe_synthetic, _is_placeholder
from anonymizer.interface.anonymizer import Anonymizer

TEXT = "Bob met Jon and jon."


def _entities(label: str) -> dict[str, Any]:
    """Bob as a first name, and Jon and jon (one value, two spellings) under ``label``."""
    values = {"Bob": "first_name", "Jon": label, "jon": label}
    return {
        "entities": [
            {
                "id": str(i),
                "value": v,
                "label": lab,
                "start_position": TEXT.index(v),
                "end_position": TEXT.index(v) + len(v),
            }
            for i, (v, lab) in enumerate(values.items())
        ]
    }


class _Detector:
    def __init__(self, label: str) -> None:
        self.label = label

    def run(self, dataframe: pd.DataFrame, **_: object) -> EntityDetectionResult:
        frame = dataframe.copy()
        found = [_entities(self.label) for _ in frame[COL_TEXT]]
        frame[COL_FINAL_ENTITIES] = found
        frame[COL_ENTITIES_BY_VALUE] = [_build_entities_by_value(e) for e in found]
        return EntityDetectionResult(frame, [], tuple(range(len(frame))))


def _substitute(tmp_path: Path, label: str, session: Session | None, consistency: Literal["record", "run"]) -> str:
    """``TEXT`` through ``Anonymizer.run``; the model proposes "Bob" for Jon and jon, a value of the record, so the
    value under ``label`` gets a minted placeholder, and Bob gets "Zed"."""
    proposals = {"Bob": "Zed", "Jon": "Bob", "jon": "Bob"}

    def run_workflow(self: NddAdapter, df: pd.DataFrame, **_: Any) -> WorkflowRunResult:
        out = df.copy().reset_index(drop=True)
        out[COL_REPLACEMENT_MAP] = [
            {
                "replacements": [
                    {"original": e["value"], "label": e["label"], "synthetic": proposals[e["value"]]}
                    for e in _entities(label)["entities"]
                ]
            }
            for _ in out[COL_TEXT]
        ]
        return WorkflowRunResult(dataframe=out, failed_records=[], source_positions=tuple(range(len(out))))

    source = tmp_path / "in.csv"
    pd.DataFrame({"text": [TEXT]}).to_csv(source, index=False)
    anonymizer = Anonymizer(detection_workflow=cast(EntityDetectionWorkflow, _Detector(label)))
    keyword: dict[str, Any] = {} if session is None else {"session": session}
    with patch.object(NddAdapter, "run_workflow", run_workflow):
        result = anonymizer.run(
            config=AnonymizerConfig(replace=Substitute(consistency=consistency)),
            data=AnonymizerInput(source=str(source)),
            **keyword,
        )
    return str(result.dataframe["text_replaced"][0])


# Labels whose upper-case form adds or changes marks or length, and controls where it does not.
MARKED = ["ǰob", "ΐob", "ŉob", "ßob"]
CONTROLS = ["job", "İstanbul", "ǆa"]
SESSIONS = {
    "no-session": lambda: None,
    "session": Session,
    "stable-session": lambda: Session(stable_replacements=True),
}


@pytest.mark.parametrize("consistency", ["record", "run"])
@pytest.mark.parametrize("session", list(SESSIONS))
@pytest.mark.parametrize("label", [*MARKED, *CONTROLS])
def test_a_minted_placeholder_is_not_re_cased_for_a_lower_case_spelling(
    tmp_path: Path, label: str, session: str, consistency: Literal["record", "run"]
) -> None:
    minted = _collision_safe_synthetic(label, index=1, protected_original_values=())
    output = _substitute(tmp_path, label, SESSIONS[session](), consistency)
    assert output == f"Zed met {minted} and {minted}."


@pytest.mark.parametrize("label", ["ǰ", "ΐ", "ŉ", "ß"])
def test_is_placeholder_accepts_tokens_whose_upper_case_adds_marks_or_changes_length(label: str) -> None:
    minted = _collision_safe_synthetic(label, index=1, protected_original_values=())
    assert _is_placeholder(minted), minted
    assert not _is_placeholder(minted.lower()), minted.lower()
    assert not _is_placeholder(minted.title()), minted.title()
    assert not _is_placeholder(minted.replace("SUBSTITUTE", "substitute"))


@pytest.mark.parametrize("label", ["first_name", "İstanbul", "ǆa", "ǰob", "a-ŉ b", "ǰ!!", "!!"])
def test_is_placeholder_accepts_every_minted_token(label: str) -> None:
    for index in (1, 2, 10):
        minted = _collision_safe_synthetic(label, index=index, protected_original_values=())
        assert _is_placeholder(minted), minted


@pytest.mark.parametrize(
    "value",
    [
        "[SUBSTITUTE_̌_1]",  # a mark with no letter to follow
        "[SUBSTITUTE__J̌_1]",  # a leading underscore
        "[SUBSTITUTE_J̌__1]",  # a trailing underscore
        "[SUBSTITUTE_ǰ_1]",  # the label's own character, not its upper-case form
        "[SUBSTITUTE_First_1]",
        "[SUBSTITUTE_J̌_01]",
        "[SUBSTITUTE__1]",
    ],
)
def test_is_placeholder_rejects_tokens_minting_cannot_produce(value: str) -> None:
    assert not _is_placeholder(value), value
