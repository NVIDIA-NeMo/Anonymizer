# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Substitute under a stable Session: substitutes carry across calls, and no substitute the Session has seen, held or
used once in a record, is given to another value in a later call."""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pandas as pd
import pytest

import anonymizer.engine.replace.route as route
from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput
from anonymizer.config.replace_strategies import Substitute
from anonymizer.engine.constants import COL_ENTITIES_BY_VALUE, COL_FINAL_ENTITIES, COL_REPLACEMENT_MAP, COL_TEXT
from anonymizer.engine.detection.detection_workflow import (
    EntityDetectionResult,
    EntityDetectionWorkflow,
    _build_entities_by_value,
)
from anonymizer.engine.graph_core.execution import ProtectionResult
from anonymizer.engine.graph_core.session import Session, SessionChanges, SessionView
from anonymizer.engine.ndd.adapter import NddAdapter, WorkflowRunResult
from anonymizer.interface.anonymizer import Anonymizer
from tests.engine import test_replace_detect_failures as rejected

LABELS = {"Alice": "first_name", "ALICE": "first_name", "Bob": "first_name", "Carol": "first_name"}
LABELS |= {"Dave": "first_name", "Paris": "city"}


class _Detector:
    """Every ``LABELS`` value found in the text, at its first occurrence."""

    def run(self, dataframe: pd.DataFrame, **_: object) -> EntityDetectionResult:
        frame = dataframe.copy()
        entities = [_entities(str(text)) for text in frame[COL_TEXT]]
        frame[COL_FINAL_ENTITIES] = entities
        frame[COL_ENTITIES_BY_VALUE] = [_build_entities_by_value(e) for e in entities]
        return EntityDetectionResult(frame, [], tuple(range(len(frame))))


def _entities(text: str) -> dict[str, Any]:
    spans = [
        {
            "id": str(i),
            "value": v,
            "label": label,
            "start_position": text.index(v),
            "end_position": text.index(v) + len(v),
        }
        for i, (v, label) in enumerate(LABELS.items())
        if v in text
    ]
    return {"entities": spans}


@contextlib.contextmanager
def _bound(session: Session | None, results: list[ProtectionResult]) -> Iterator[None]:
    """Every Replace invocation runs with ``session``, and its result is kept."""
    run = route._run

    def bound(graph: Any, workflow: Any, limits: Any, *_: Any) -> ProtectionResult:
        results.append(result := run(graph, workflow, limits, session=session))
        return result

    with patch.object(route, "_run", bound):
        yield


def _protect(
    tmp_path: Path,
    texts: list[str],
    proposals: dict[str, str],
    session: Session | None,
    results: list[ProtectionResult],
    rejecting: bool = False,
    bind: bool = True,
    trace: bool = False,
) -> pd.DataFrame:
    """One ``Anonymizer.run`` with Substitute; the model proposes ``proposals[value]`` for each detected value."""
    source = tmp_path / f"in-{len(list(tmp_path.iterdir()))}.csv"
    pd.DataFrame({"text": texts}).to_csv(source, index=False)
    entities = rejected._entities if rejecting else _entities
    detector = rejected._OverlappingDetector() if rejecting else _Detector()

    def run_workflow(self: NddAdapter, df: pd.DataFrame, *, workflow_name: str, **_: Any) -> WorkflowRunResult:
        assert workflow_name == "replace-map-generation", workflow_name
        out = df.copy().reset_index(drop=True)
        out[COL_REPLACEMENT_MAP] = [
            {
                "replacements": [
                    {"original": e["value"], "label": e["label"], "synthetic": proposals[e["value"]]}
                    for e in entities(str(text))["entities"]
                ]
            }
            for text in out[COL_TEXT]
        ]
        return WorkflowRunResult(dataframe=out, failed_records=[], source_positions=tuple(range(len(out))))

    anonymizer = Anonymizer(detection_workflow=cast(EntityDetectionWorkflow, detector))
    bound = _bound(session, results) if bind else contextlib.nullcontext()
    with patch.object(NddAdapter, "run_workflow", run_workflow), bound:
        result = anonymizer.run(config=AnonymizerConfig(replace=Substitute()), data=AnonymizerInput(source=str(source)))
    return result.trace_dataframe if trace else result.dataframe


def _slots(session: Session, subject: str = "by-value") -> dict[tuple[str, str], str]:
    """The snapshot's slots under ``subject``, as (label, normalized value) to assignment."""
    slots = json.loads(session.snapshot().to_bytes())["slots"]
    return {(s["label"], s["value"]): s["assignment"] for s in slots if s["subject"] == subject}


def test_a_stable_session_reuses_substitutes_across_calls(tmp_path: Path) -> None:
    session, results = Session(stable_replacements=True), []
    first = _protect(tmp_path, ["Alice lives in Paris."], {"Alice": "Bob", "Paris": "Lyon"}, session, results)
    proposals = {"ALICE": "Zed", "Carol": "Dana", "Paris": "Nice"}
    second = _protect(tmp_path, ["ALICE met Carol in Paris."], proposals, session, results)
    assert list(first["text_replaced"]) == ["Bob lives in Lyon."]
    # The retained substitutes win over the new proposals, re-cased against the normalized original.
    assert list(second["text_replaced"]) == ["BOB met Dana in Lyon."]
    assert [r.session_changes for r in results] == [SessionChanges(accepted=2), SessionChanges(accepted=1, reused=2)]
    assert _slots(session) == {
        ("first_name", "alice"): "Bob",
        ("city", "paris"): "Lyon",
        ("first_name", "carol"): "Dana",
    }


@pytest.mark.parametrize("session", [None, Session()], ids=["no-session", "off-session"])
def test_without_a_stable_session_nothing_changes(tmp_path: Path, session: Session | None) -> None:
    calls = [(["Alice lives in Paris."], {"Alice": "Bob", "Paris": "Lyon"})]
    calls.append(
        (["Alice met Carol in Paris.", "Carol saw Alice."], {"Alice": "Zed", "Carol": "Dana", "Paris": "Nice"})
    )
    results: list[ProtectionResult] = []
    plain = [_protect(tmp_path, texts, p, None, [], bind=False, trace=True) for texts, p in calls]
    with_session = [_protect(tmp_path, texts, p, session, results, trace=True) for texts, p in calls]
    assert all(COL_REPLACEMENT_MAP in frame.columns for frame in plain)
    for a, b in zip(plain, with_session):
        pd.testing.assert_frame_equal(a, b)
    assert list(with_session[1]["text_replaced"]) == ["Zed met Dana in Nice.", "Dana saw Zed."]
    assert [r.session_changes for r in results] == [SessionChanges(), SessionChanges()]
    assert session is None or session.revision == 0


@pytest.mark.parametrize("cover", [True, False], ids=["cover", "no-cover"])
def test_a_substitute_assigned_in_an_earlier_call_is_not_given_to_another_value(tmp_path: Path, cover: bool) -> None:
    session, results = Session(stable_replacements=True), []
    _protect(tmp_path, ["Alice lives here."], {"Alice": "Bob"}, session, results)
    # The model proposes, for another value, the substitute Alice already holds.
    uncovered = patch.object(SessionView, "assigned_elsewhere", lambda *_: False)
    with contextlib.nullcontext() if cover else uncovered:
        second = _protect(tmp_path, ["Carol lives here."], {"Carol": "bob"}, session, results)
    carol = "[SUBSTITUTE_FIRST_NAME_1]" if cover else "bob"  # without the cover, the two values share one
    assert list(second["text_replaced"]) == [f"{carol} lives here."]
    assert _slots(session) == {("first_name", "alice"): "Bob", ("first_name", "carol"): carol}


def test_a_placeholder_from_an_earlier_call_is_not_reused(tmp_path: Path) -> None:
    session, results = Session(stable_replacements=True), []
    first = _protect(tmp_path, ["Alice and Carol."], {"Alice": "Bob", "Carol": "Bob"}, session, results)
    assert list(first["text_replaced"]) == ["Bob and [SUBSTITUTE_FIRST_NAME_1]."]
    # Dave's own proposal is his original, so he needs a placeholder; this call's first index is Carol's.
    second = _protect(tmp_path, ["Dave."], {"Dave": "Dave"}, session, results)
    assert list(second["text_replaced"]) == ["[SUBSTITUTE_FIRST_NAME_2]."]


def test_a_record_local_placeholder_is_not_reused_in_a_later_call(tmp_path: Path) -> None:
    session, results = Session(stable_replacements=True), []
    # Alice's run-wide Bob collides with the real Bob in the second record, so Alice takes a placeholder there only.
    first = _protect(tmp_path, ["Alice is here.", "Alice and Bob."], {"Alice": "Bob", "Bob": "Carl"}, session, results)
    assert list(first["text_replaced"]) == ["Bob is here.", "[SUBSTITUTE_FIRST_NAME_1] and Carl."]
    second = _protect(tmp_path, ["Dave is here."], {"Dave": "Dave"}, session, results)
    assert list(second["text_replaced"]) == ["[SUBSTITUTE_FIRST_NAME_2] is here."]
    assert _slots(session, "substitute-fallback") == {
        ("first_name", "[substitute_first_name_1]"): "[substitute_first_name_1]"
    }
    assert [r.session_changes for r in results] == [SessionChanges(accepted=3), SessionChanges(accepted=1)]


def test_a_record_local_fallback_is_given_to_no_value_in_a_later_call(tmp_path: Path) -> None:
    session, results = Session(stable_replacements=True), []
    _protect(tmp_path, ["Alice is here."], {"Alice": "Bob"}, session, results)
    # In a record with the real Bob, Alice falls back to the model's own proposal, Zed, for that record only.
    second = _protect(tmp_path, ["Alice and Bob."], {"Alice": "Zed", "Bob": "Carl"}, session, results)
    assert list(second["text_replaced"]) == ["Zed and Carl."]
    # Neither Alice again (in a record with Bob, so she falls back again) nor Carol may now take Zed.
    texts = ["Carol is here.", "Alice and Bob met."]
    third = _protect(tmp_path, texts, {"Alice": "Zed", "Bob": "Carl", "Carol": "Zed"}, session, results)
    assert list(third["text_replaced"]) == [
        "[SUBSTITUTE_FIRST_NAME_1] is here.",
        "[SUBSTITUTE_FIRST_NAME_2] and Carl met.",
    ]
    # The marker is bookkeeping: no value holds Zed, and Alice still holds Bob.
    assert _slots(session) == {
        ("first_name", "alice"): "Bob",
        ("first_name", "bob"): "Carl",
        ("first_name", "carol"): "[SUBSTITUTE_FIRST_NAME_1]",
    }
    assert set(_slots(session, "substitute-fallback")) == {
        ("first_name", "zed"),
        ("first_name", "[substitute_first_name_2]"),
    }
    # Call 2 accepts Bob's slot and the Zed marker; call 3 Carol's slot and the placeholder marker.
    assert [r.session_changes for r in results[1:]] == [
        SessionChanges(accepted=2, reused=1),
        SessionChanges(accepted=2, reused=2),
    ]


def test_one_value_on_several_rows_of_a_call_is_decided_once(tmp_path: Path) -> None:
    session, results = Session(stable_replacements=True), []
    texts = ["Alice one.", "Alice two.", "Alice three."]
    first = _protect(tmp_path, texts, {"Alice": "Bob"}, session, results)
    second = _protect(tmp_path, texts, {"Alice": "Zed"}, session, results)
    assert list(first["text_replaced"]) == list(second["text_replaced"]) == ["Bob one.", "Bob two.", "Bob three."]
    assert [r.session_changes for r in results] == [SessionChanges(accepted=1), SessionChanges(reused=1)]


def test_a_substitute_equal_to_an_earlier_original_is_rejected(tmp_path: Path) -> None:
    session, results = Session(stable_replacements=True), []
    _protect(tmp_path, ["Alice lives here."], {"Alice": "Bob"}, session, results)
    second = _protect(tmp_path, ["Carol lives here."], {"Carol": "ALICE"}, session, results)
    assert list(second["text_replaced"]) == ["[SUBSTITUTE_FIRST_NAME_1] lives here."]


def test_a_held_substitute_stays_when_it_later_appears_as_an_original(tmp_path: Path) -> None:
    session, results = Session(stable_replacements=True), []
    _protect(tmp_path, ["Alice lives here."], {"Alice": "Bob"}, session, results)
    _protect(tmp_path, ["Bob lives here."], {"Bob": "Erin"}, session, results)
    third = _protect(tmp_path, ["Alice is back."], {"Alice": "Fay"}, session, results)
    assert list(third["text_replaced"]) == ["Bob is back."]


def test_a_row_rejected_at_detect_contributes_nothing_to_the_session(tmp_path: Path) -> None:
    session, results = Session(stable_replacements=True), []
    proposals = {**rejected.SUBSTITUTES[rejected.REJECTED], **rejected.SUBSTITUTES[rejected.VALID]}
    proposals["Paris"] = "Nice"
    texts = [rejected.REJECTED, rejected.VALID]
    output = _protect(tmp_path, texts, proposals, session, results, rejecting=True)
    assert list(output["text_replaced"]) == ["Dave lives in Nice."]
    (result,) = results
    assert result.session_changes == SessionChanges(accepted=2)
    # Only the valid row's slots, and Paris holds the valid row's substitute, not the rejected row's.
    assert _slots(session) == {("first_name", "bob"): "Dave", ("city", "paris"): "Nice"}
    # A later call reuses the valid row's substitute.
    later = _protect(tmp_path, ["Off to Paris."], {"Paris": "Lyon"}, session, results)
    assert list(later["text_replaced"]) == ["Off to Nice."]


def test_no_value_reaches_the_logs_or_new_reprs(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    session, results = Session(stable_replacements=True), []
    with caplog.at_level(logging.DEBUG):
        _protect(tmp_path, ["Alice and Carol."], {"Alice": "Bobbington", "Carol": "Bobbington"}, session, results)
        _protect(tmp_path, ["Dave met Alice."], {"Dave": "Bobbington", "Alice": "Zebulon"}, session, results)
    rendered = caplog.text + "".join(repr(r.session_changes) for r in results) + repr(session) + repr(session._view())
    rendered += repr(session.snapshot())
    for value in ("alice", "carol", "bobbington", "zebulon", "substitute_"):
        assert value not in rendered.lower()
