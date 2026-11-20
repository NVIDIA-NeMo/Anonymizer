# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public replay under a stable Session: the same records and accepted mentions give the same text and checkpoint
bytes across a checkpoint restore, whatever order the model returns its entries in and whatever order the detector
returns its rows and entities in."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
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
from anonymizer.engine.graph_core.session import Session, SessionSnapshot
from anonymizer.engine.ndd.adapter import NddAdapter, WorkflowRunResult
from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.interface.results import AnonymizerResult, SessionCallChanges

Mentions = dict[str, list[tuple[str, str]]]


class _Detector:
    """The given (value, label) mentions of each text, each at its next occurrence; ``reverse_entities`` returns each
    row's entities reversed and ``reverse_rows`` returns the rows reversed, with their source positions."""

    def __init__(self, mentions: Mentions, *, reverse_entities: bool = False, reverse_rows: bool = False) -> None:
        self.mentions, self.reverse_entities, self.reverse_rows = mentions, reverse_entities, reverse_rows

    def run(self, dataframe: pd.DataFrame, **_: object) -> EntityDetectionResult:
        frame = dataframe.copy()
        rows = []
        for text in frame[COL_TEXT]:
            text, spans, after = str(text), [], {}
            for i, (value, label) in enumerate(self.mentions[text]):
                start = text.index(value, after.get(value, 0))
                after[value] = start + 1
                spans.append(
                    {
                        "id": str(i),
                        "value": value,
                        "label": label,
                        "start_position": start,
                        "end_position": start + len(value),
                    }
                )
            rows.append({"entities": spans[::-1] if self.reverse_entities else spans})
        frame[COL_FINAL_ENTITIES] = rows
        frame[COL_ENTITIES_BY_VALUE] = [_build_entities_by_value(e) for e in rows]
        positions = tuple(range(len(frame)))
        if self.reverse_rows:
            frame, positions = frame.iloc[::-1].reset_index(drop=True), positions[::-1]
        return EntityDetectionResult(frame, [], positions)


def _run(
    tmp_path: Path, texts: list[str], detector: _Detector, entries: list[tuple[str, str, str]], session: Session
) -> AnonymizerResult:
    """One Substitute ``run``; the model returns ``entries`` (original, label, substitute) in the given order, each
    row's requested pairs only."""

    def run_workflow(self: NddAdapter, df: pd.DataFrame, **_: Any) -> WorkflowRunResult:
        out = df.copy().reset_index(drop=True)
        out[COL_REPLACEMENT_MAP] = [
            {
                "replacements": [
                    {"original": o, "label": label, "synthetic": s}
                    for o, label, s in entries
                    if (o, label) in set(detector.mentions[str(text)])
                ]
            }
            for text in out[COL_TEXT]
        ]
        return WorkflowRunResult(dataframe=out, failed_records=[], source_positions=tuple(range(len(out))))

    source = tmp_path / f"in-{len(list(tmp_path.iterdir()))}.csv"
    pd.DataFrame({"text": texts}).to_csv(source, index=False)
    anonymizer = Anonymizer(detection_workflow=cast(EntityDetectionWorkflow, detector))
    with patch.object(NddAdapter, "run_workflow", run_workflow):
        return anonymizer.run(
            config=AnonymizerConfig(replace=Substitute()), data=AnonymizerInput(source=str(source)), session=session
        )


def _restored(session: Session) -> Session:
    return Session.restore(SessionSnapshot.from_bytes(session.snapshot().to_bytes()))


_ONE_ROW = ["Alice met ALICE."]
_TWO_ROWS = ["Alice left.", "ALICE came back."]
_CASES = {
    # One value in two spellings, one label, the same proposal for both.
    "same-proposal": (
        _ONE_ROW,
        {_ONE_ROW[0]: [("Alice", "first_name"), ("ALICE", "first_name")]},
        [("Alice", "first_name", "Bob"), ("ALICE", "first_name", "Bob")],
        ["Bob met BOB."],
        1,
    ),
    # Upper-casing the substitute changes its normalized value (dotless i), so it is emitted as assigned.
    "dotless-i": (
        _ONE_ROW,
        {_ONE_ROW[0]: [("Alice", "first_name"), ("ALICE", "first_name")]},
        [("Alice", "first_name", "Işıl"), ("ALICE", "first_name", "Luna")],
        ["Işıl met Işıl."],
        1,
    ),
    # The same value under two labels, each holding Işıl: the first label's upper-case spelling must not reserve a
    # fallback against the Session's assignments.
    "dotless-i-two-labels": (
        ["Alice met ALICE, Alice."],
        {"Alice met ALICE, Alice.": [("Alice", "first_name"), ("ALICE", "first_name"), ("Alice", "last_name")]},
        [("Alice", "first_name", "Işıl"), ("ALICE", "first_name", "Işıl"), ("Alice", "last_name", "Işıl")],
        ["Işıl met Işıl, Işıl."],
        2,
    ),
    # The spellings in two rows: target order, not the detector's row order, picks the casing reference.
    "two-rows": (
        _TWO_ROWS,
        {_TWO_ROWS[0]: [("Alice", "first_name")], _TWO_ROWS[1]: [("ALICE", "first_name")]},
        [("Alice", "first_name", "Bob"), ("ALICE", "first_name", "Bob")],
        ["Bob left.", "BOB came back."],
        1,
    ),
}


@pytest.mark.parametrize("case", list(_CASES))
@pytest.mark.parametrize("detector_order", ["as-input", "reversed"])
@pytest.mark.parametrize("model_order", ["as-input", "reversed"])
def test_a_replay_after_restore_gives_the_same_text_and_checkpoint(
    tmp_path: Path, case: str, detector_order: str, model_order: str
) -> None:
    texts, mentions, entries, expected, assignments = _CASES[case]
    session = Session(stable_replacements=True)
    first = _run(tmp_path, texts, _Detector(mentions), entries, session)
    assert list(first.dataframe["text_replaced"]) == expected
    assert first.session_changes == SessionCallChanges(added=assignments)
    saved = session.snapshot().to_bytes()

    reversed_ = detector_order == "reversed"
    detector = _Detector(mentions, reverse_entities=reversed_, reverse_rows=reversed_)
    replay_entries = entries[::-1] if model_order == "reversed" else entries
    restored = Session.restore(SessionSnapshot.from_bytes(saved))
    for _ in range(2):
        replay = _run(tmp_path, texts, detector, replay_entries, restored)
        assert list(replay.dataframe["text_replaced"]) == expected
        assert replay.session_changes == SessionCallChanges(reused=assignments)
        assert restored.snapshot().to_bytes() == saved
        restored = _restored(restored)


def test_a_genuine_collision_still_reserves_a_fallback(tmp_path: Path) -> None:
    """Positive control: a held substitute that is a value of the record falls back, and the fallback is reserved."""
    session = Session(stable_replacements=True)
    text = "Alice left."
    _run(tmp_path, [text], _Detector({text: [("Alice", "first_name")]}), [("Alice", "first_name", "Bob")], session)
    text = "Alice met Bob."
    mentions = {text: [("Alice", "first_name"), ("Bob", "first_name")]}
    entries = [("Alice", "first_name", "Zoe"), ("Bob", "first_name", "Carl")]
    second = _run(tmp_path, [text], _Detector(mentions), entries, session)
    assert list(second.dataframe["text_replaced"]) == ["Zoe met Carl."]
    assert second.session_changes == SessionCallChanges(added=1, reused=1, fallbacks=1)
