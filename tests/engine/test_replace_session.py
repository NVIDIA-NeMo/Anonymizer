# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``Anonymizer.run``/``preview`` with ``session=``: under a stable Session Substitute carries substitutes across calls,
and no substitute the Session has seen, held or used once in a record, is given to another value in a later call;
Redact, Annotate and Hash leave the Session unchanged; Rewrite and overriding runners are rejected before detection."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import json
import logging
import random
import re
import types
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, Rewrite
from anonymizer.config.replace_strategies import Annotate, Hash, Redact, ReplaceMethod, Substitute
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_ENTITIES_BY_VALUE,
    COL_ENTITIES_FOR_REPLACE,
    COL_FINAL_ENTITIES,
    COL_LEAKAGE_MASS,
    COL_REPLACEMENT_MAP,
    COL_REPLACEMENT_RETRY_CONTEXT,
    COL_REWRITTEN_TEXT,
    COL_TEXT,
    COL_UTILITY_SCORE,
    COL_WEIGHTED_LEAKAGE_RATE,
)
from anonymizer.engine.detection.detection_workflow import (
    EntityDetectionResult,
    EntityDetectionWorkflow,
    _build_entities_by_value,
)
from anonymizer.engine.graph_core.session import FALLBACK, Session, SessionChanges, SessionView, normalize
from anonymizer.engine.ndd.adapter import NddAdapter, WorkflowRunResult
from anonymizer.engine.replace.llm_replace_workflow import (
    LlmReplaceWorkflow,
    _collision_safe_synthetic,
    _fold_replacement_maps,
    _is_placeholder,
)
from anonymizer.engine.replace.replace_runner import ReplacementResult, ReplacementWorkflow
from anonymizer.engine.rewrite.combined_rewrite_workflow import CombinedRewriteWorkflow
from anonymizer.engine.rewrite.rewrite_workflow import RewriteResult, RewriteWorkflow
from anonymizer.graph import SessionSnapshot
from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.interface.errors import InvalidConfigError
from anonymizer.interface.results import AnonymizerResult, Replacement, SessionCallChanges
from tests.engine import test_replace_detect_failures as rejected

LABELS = {"Alice": "first_name", "ALICE": "first_name", "Bob": "first_name", "Carol": "first_name"}
LABELS |= {"Dave": "first_name", "Zed": "first_name", "Paris": "city"}


class _Detector:
    """Every ``LABELS`` value found in the text, at its first occurrence; ``calls`` counts the runs."""

    calls = 0

    def run(self, dataframe: pd.DataFrame, **_: object) -> EntityDetectionResult:
        type(self).calls += 1
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


_UNSET: Any = object()


def _protect(
    tmp_path: Path,
    texts: list[str],
    proposals: dict[str, str],
    session: Session | None = _UNSET,
    results: list[Any] | None = None,
    rejecting: bool = False,
    trace: bool = False,
    preview: bool = False,
    strategy: ReplaceMethod | None = None,
    anonymizer: Anonymizer | None = None,
    detector: Any = None,
    kept: list[Any] | None = None,
) -> pd.DataFrame:
    """One ``Anonymizer.run`` (or ``preview``) with Substitute (or ``strategy``), passing ``session`` unless unset and
    keeping the result's session changes in ``results``; the model proposes ``proposals[value]`` for each value."""
    source = tmp_path / f"in-{len(list(tmp_path.iterdir()))}.csv"
    pd.DataFrame({"text": texts}).to_csv(source, index=False)
    entities = rejected._entities if rejecting else _entities
    detector = detector or (rejected._OverlappingDetector() if rejecting else _Detector())

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

    anonymizer = anonymizer or Anonymizer(detection_workflow=cast(EntityDetectionWorkflow, detector))
    call = anonymizer.preview if preview else anonymizer.run
    config = AnonymizerConfig(replace=strategy or Substitute())
    keyword: dict[str, Any] = {} if session is _UNSET else {"session": session}
    with patch.object(NddAdapter, "run_workflow", run_workflow):
        result = call(config=config, data=AnonymizerInput(source=str(source)), **keyword)
    if results is not None:
        results.append(result.session_changes)
    if kept is not None:
        kept.append(result)
    return result.trace_dataframe if trace else result.dataframe


def _slots(session: Session, subject: str = "by-value") -> dict[tuple[str, str], str]:
    """The snapshot's slots under ``subject``, as (label, normalized value) to assignment."""
    slots = json.loads(session.snapshot().to_bytes())["slots"]
    return {(s["label"], s["value"]): s["assignment"] for s in slots if s["subject"] == subject}


@pytest.mark.parametrize("preview", [False, True], ids=["run", "preview"])
def test_a_stable_session_reuses_substitutes_across_calls(tmp_path: Path, preview: bool) -> None:
    session, results = Session(stable_replacements=True), []
    first = _protect(
        tmp_path, ["Alice lives in Paris."], {"Alice": "Bob", "Paris": "Lyon"}, session, results, preview=preview
    )
    proposals = {"ALICE": "Zed", "Carol": "Dana", "Paris": "Nice"}
    second = _protect(tmp_path, ["ALICE met Carol in Paris."], proposals, session, results, preview=preview)
    assert list(first["text_replaced"]) == ["Bob lives in Lyon."]
    # The retained substitutes win over the new proposals, re-cased against the normalized original.
    assert list(second["text_replaced"]) == ["BOB met Dana in Lyon."]
    assert results == [SessionCallChanges(added=2), SessionCallChanges(added=1, reused=2)]
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
    results: list[SessionChanges | None] = []
    plain = [_protect(tmp_path, texts, p, trace=True) for texts, p in calls]
    with_session = [_protect(tmp_path, texts, p, session, results, trace=True) for texts, p in calls]
    assert all(COL_REPLACEMENT_MAP in frame.columns for frame in plain)
    for a, b in zip(plain, with_session):
        pd.testing.assert_frame_equal(a, b)
    assert list(with_session[1]["text_replaced"]) == ["Zed met Dana in Nice.", "Dana saw Zed."]
    assert results == [None, None]
    assert session is None or session.revision == 0


@pytest.mark.parametrize("cover", [True, False], ids=["cover", "no-cover"])
def test_a_substitute_assigned_in_an_earlier_call_is_not_given_to_another_value(tmp_path: Path, cover: bool) -> None:
    session, results = Session(stable_replacements=True), []
    _protect(tmp_path, ["Alice lives here."], {"Alice": "Bob"}, session, results)
    # The model proposes, for another value, the substitute Alice already holds.
    uncovered = patch.object(SessionView, "assigned_to_other_value", lambda *_: False)
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
    protected = SessionCallChanges(added=1, placeholders_protected=1)
    assert results == [SessionCallChanges(added=2, fallbacks=1, placeholders_protected=1), protected]


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
    # Call 2 adds Bob's slot and the Zed marker; call 3 Carol's slot and the placeholder marker. Only the Session
    # rules Zed out in call 3, so both placeholders count under that cause.
    assert results[1:] == [
        SessionCallChanges(added=1, reused=1, fallbacks=1),
        SessionCallChanges(added=1, reused=2, fallbacks=1, placeholders_session=2),
    ]
    # A real Zed, whose original is the marker's substitute, gets an assignment of its own, never the marker's.
    fourth = _protect(tmp_path, ["Zed is here."], {"Zed": "Yuri"}, session, results)
    fifth = _protect(tmp_path, ["Zed again."], {"Zed": "Xavi"}, session, results)
    assert [*fourth["text_replaced"], *fifth["text_replaced"]] == ["Yuri is here.", "Yuri again."]
    assert _slots(session)[("first_name", "zed")] == "Yuri"


def test_one_value_on_several_rows_of_a_call_is_decided_once(tmp_path: Path) -> None:
    session, results = Session(stable_replacements=True), []
    texts = ["Alice one.", "Alice two.", "Alice three."]
    first = _protect(tmp_path, texts, {"Alice": "Bob"}, session, results)
    second = _protect(tmp_path, texts, {"Alice": "Zed"}, session, results)
    assert list(first["text_replaced"]) == list(second["text_replaced"]) == ["Bob one.", "Bob two.", "Bob three."]
    assert results == [SessionCallChanges(added=1), SessionCallChanges(reused=1)]


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
    assert results == [SessionCallChanges(added=2)]
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
    rendered = caplog.text + "".join(repr(c) for c in results) + repr(session) + repr(session._view())
    rendered += repr(session.snapshot())
    for value in ("alice", "carol", "bobbington", "zebulon", "substitute_"):
        assert value not in rendered.lower()


def test_a_snapshot_restored_between_calls_gives_the_same_substitutes(tmp_path: Path) -> None:
    calls = [
        (["Alice lives in Paris."], {"Alice": "Bob", "Paris": "Lyon"}),
        (["Carol met Alice."], {"Carol": "Dana", "Alice": "Zed"}),
        (["Carol, Alice and Dave in Paris."], {"Carol": "Erin", "Alice": "Fay", "Dave": "Gus", "Paris": "Nice"}),
    ]
    continuous = Session(stable_replacements=True)
    expected = [list(_protect(tmp_path, t, p, continuous)["text_replaced"]) for t, p in calls]
    session = Session(stable_replacements=True)
    outputs = []
    for texts, proposals in calls:
        outputs.append(list(_protect(tmp_path, texts, proposals, session)["text_replaced"]))
        session = Session.restore(SessionSnapshot.from_bytes(session.snapshot().to_bytes()))
    assert outputs == expected
    assert expected[2] == ["Dana, Bob and Gus in Lyon."]
    assert session.snapshot().to_bytes() == continuous.snapshot().to_bytes()


@pytest.mark.parametrize("strategy", [Redact(), Annotate(), Hash(key="test-key")], ids=["redact", "annotate", "hash"])
def test_a_local_strategy_leaves_a_stable_session_unchanged(tmp_path: Path, strategy: ReplaceMethod) -> None:
    session, results = Session(stable_replacements=True), []
    _protect(tmp_path, ["Alice lives in Paris."], {"Alice": "Bob", "Paris": "Lyon"}, session, results)
    before = session.snapshot().to_bytes()
    texts = ["Alice met Carol in Paris.", "Carol saw Alice."]
    plain = _protect(tmp_path, texts, {}, strategy=strategy, trace=True)
    with_session = _protect(tmp_path, texts, {}, session, results, strategy=strategy, trace=True)
    pd.testing.assert_frame_equal(plain, with_session)
    assert results[1] == SessionCallChanges()
    assert session.snapshot().to_bytes() == before


class _OwnReplace(ReplacementWorkflow):
    def run(self, dataframe: pd.DataFrame, **_: Any) -> ReplacementResult:
        frame = dataframe.copy()
        frame["text_replaced"] = "custom"
        return ReplacementResult(dataframe=frame, failed_records=[])


def _rewritten(dataframe: pd.DataFrame) -> RewriteResult:
    frame = dataframe.copy()
    frame[COL_REWRITTEN_TEXT] = "custom"
    frame[[COL_UTILITY_SCORE, COL_LEAKAGE_MASS, COL_WEIGHTED_LEAKAGE_RATE]] = [1.0, 0.0, 0.0]
    frame[COL_ANY_HIGH_LEAKED] = False
    return RewriteResult(frame, [])


class _OwnRewrite(RewriteWorkflow):
    def run(self, dataframe: pd.DataFrame, **_: Any) -> RewriteResult:
        return _rewritten(dataframe)


class _OwnCombined(CombinedRewriteWorkflow):
    def run(self, dataframe: pd.DataFrame, **_: Any) -> RewriteResult:
        return _rewritten(dataframe)


@pytest.mark.parametrize(
    ("config", "anonymizer_kwargs", "session", "message"),
    [
        (AnonymizerConfig(rewrite=Rewrite()), {}, "stable", "Rewrite cannot take a stable Session"),
        (AnonymizerConfig(rewrite=Rewrite(use_combined_graph=True)), {}, "stable", "Rewrite cannot take"),
        (
            AnonymizerConfig(replace=Substitute()),
            {"replace_runner": _OwnReplace()},
            "stable",
            "A custom replace runner that overrides run() cannot take part in a stable Session",
        ),
        (
            AnonymizerConfig(rewrite=Rewrite()),
            {"rewrite_runner": _OwnRewrite(adapter=cast(NddAdapter, None))},
            "stable",
            "Rewrite cannot take a stable Session",
        ),
        (AnonymizerConfig(replace=Substitute()), {}, "not-a-session", "session must be a Session"),
    ],
    ids=["rewrite", "combined-rewrite", "own-replace-runner", "own-rewrite-runner", "not-a-session"],
)
@pytest.mark.parametrize("preview", [False, True], ids=["run", "preview"])
def test_a_session_the_run_cannot_take_is_rejected_before_detection(
    tmp_path: Path,
    config: AnonymizerConfig,
    anonymizer_kwargs: dict[str, Any],
    session: str,
    message: str,
    preview: bool,
) -> None:
    source = tmp_path / "in.csv"
    pd.DataFrame({"text": ["Alice lives in Paris."]}).to_csv(source, index=False)
    detector = _Detector()
    anonymizer = Anonymizer(detection_workflow=cast(EntityDetectionWorkflow, detector), **anonymizer_kwargs)
    value: Any = Session(stable_replacements=True) if session == "stable" else Session(stable_replacements=True).mode
    call = anonymizer.preview if preview else anonymizer.run
    before = _Detector.calls
    with patch.object(NddAdapter, "run_workflow") as model, pytest.raises(InvalidConfigError, match=re.escape(message)):
        call(config=config, data=AnonymizerInput(source=str(source)), session=value)
    assert _Detector.calls == before
    assert model.call_count == 0


@pytest.mark.parametrize("session", [None, Session()], ids=["no-session", "off-session"])
def test_an_overriding_runner_runs_as_before_without_a_stable_session(tmp_path: Path, session: Session | None) -> None:
    anonymizer = Anonymizer(detection_workflow=cast(EntityDetectionWorkflow, _Detector()), replace_runner=_OwnReplace())
    output = _protect(tmp_path, ["Alice is here."], {}, session, anonymizer=anonymizer, trace=True)
    assert list(output["text_replaced"]) == ["custom"]


def test_a_stock_runner_with_its_own_llm_workflow_takes_a_stable_session(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    class OwnLlm(LlmReplaceWorkflow):
        pass

    detector = cast(EntityDetectionWorkflow, _Detector())
    anonymizer = Anonymizer(detection_workflow=detector)
    runner = ReplacementWorkflow(llm_workflow=OwnLlm(adapter=anonymizer._adapter))
    anonymizer = Anonymizer(detection_workflow=detector, replace_runner=runner)
    session, results = Session(stable_replacements=True), []
    _protect(tmp_path, ["Alice is here."], {"Alice": "Bob"}, session, results, anonymizer=anonymizer)
    with caplog.at_level(logging.INFO, logger="anonymizer"):
        second = _protect(tmp_path, ["Alice again."], {"Alice": "Zed"}, session, results, anonymizer=anonymizer)
    assert list(second["text_replaced"]) == ["Bob again."]
    assert results == [SessionCallChanges(added=1), SessionCallChanges(reused=1)]
    assert (
        "session changes: 0 added, 1 reused, 0 revised, 0 ignored, 0 fallbacks, 0 placeholders_protected" in caplog.text
    )


@pytest.mark.parametrize("combined", [False, True], ids=["rewrite", "combined"])
def test_an_overriding_rewrite_runner_runs_as_before_with_an_off_session(tmp_path: Path, combined: bool) -> None:
    source = tmp_path / "in.csv"
    pd.DataFrame({"text": ["Alice lives in Paris."]}).to_csv(source, index=False)
    runner: dict[str, Any] = (
        {"combined_rewrite_runner": _OwnCombined(Mock())} if combined else {"rewrite_runner": _OwnRewrite(Mock())}
    )
    anonymizer = Anonymizer(detection_workflow=cast(EntityDetectionWorkflow, _Detector()), **runner)
    config = AnonymizerConfig(rewrite=Rewrite(use_combined_graph=combined))
    data = AnonymizerInput(source=str(source))
    plain = anonymizer.run(config=config, data=data)
    with_session = anonymizer.run(config=config, data=data, session=Session())
    pd.testing.assert_frame_equal(plain.trace_dataframe, with_session.trace_dataframe)
    assert list(with_session.dataframe["text_rewritten"]) == ["custom"]
    assert with_session.session_changes is None


class _Unresolved(_Detector):
    """No positions and no record IDs, so the route reroutes over the frame as returned: reversed, first row dropped
    (as ``tests/interface/test_facade_detection_pins.py``'s unresolved detector)."""

    def run(self, dataframe: pd.DataFrame, **_: object) -> EntityDetectionResult:
        returned = dataframe.drop(columns=[c for c in dataframe.columns if c != COL_TEXT]).iloc[:0:-1]
        return EntityDetectionResult(super().run(returned.reset_index(drop=True)).dataframe, [], ())


def test_the_reroute_takes_a_stable_session(tmp_path: Path) -> None:
    session, results = Session(stable_replacements=True), []
    texts = ["Carol is here.", "Alice lives in Paris."]
    proposals = {"Alice": "Bob", "Paris": "Lyon", "Carol": "Dana"}
    first = _protect(tmp_path, texts, proposals, session, results, detector=_Unresolved())
    second = _protect(tmp_path, texts, {"Alice": "Zed", "Paris": "Nice"}, session, results, detector=_Unresolved())
    assert list(first["text_replaced"]) == list(second["text_replaced"]) == ["Bob lives in Lyon."]
    assert results == [SessionCallChanges(added=2), SessionCallChanges(reused=2)]


class _OldSignature(ReplacementWorkflow):
    """Overrides ``run_with_detection`` with the signature it had before ``session`` (``unspanned_check`` included)."""

    def run_with_detection(  # ty: ignore[invalid-method-override]
        self,
        dataframe: pd.DataFrame,
        *,
        detect: Any,
        replace_method: ReplaceMethod,
        model_configs: Any,
        selected_models: Any,
        preview_num_records: int | None = None,
        unspanned_check: bool = False,
    ) -> ReplacementResult:
        return super().run_with_detection(
            dataframe,
            detect=detect,
            replace_method=replace_method,
            model_configs=model_configs,
            selected_models=selected_models,
            preview_num_records=preview_num_records,
            unspanned_check=unspanned_check,
        )


class _DroppingRunner(ReplacementWorkflow):
    """Overrides ``run_with_detection`` with ``**kwargs`` and delegates without ``session``."""

    def run_with_detection(self, dataframe: pd.DataFrame, **kwargs: Any) -> ReplacementResult:
        kwargs.pop("session", None)
        return super().run_with_detection(dataframe, **kwargs)


class _OldValidatedMaps(LlmReplaceWorkflow):
    """Overrides ``generate_validated_maps`` with the signature it had before ``session``."""

    def generate_validated_maps(  # ty: ignore[invalid-method-override]
        self,
        dataframe: pd.DataFrame,
        *,
        model_configs: Any,
        selected_models: Any,
        instructions: str | None = None,
        entities_column: str = COL_ENTITIES_BY_VALUE,
        preview_num_records: int | None = None,
        consistency: Any = "record",
    ) -> Any:
        return super().generate_validated_maps(
            dataframe,
            model_configs=model_configs,
            selected_models=selected_models,
            instructions=instructions,
            entities_column=entities_column,
            preview_num_records=preview_num_records,
            consistency=consistency,
        )


class _DroppingMaps(LlmReplaceWorkflow):
    """Overrides ``generate_validated_maps`` with ``**kwargs`` and delegates without ``session``."""

    def generate_validated_maps(self, dataframe: pd.DataFrame, **kwargs: Any) -> Any:
        kwargs.pop("session", None)
        return super().generate_validated_maps(dataframe, **kwargs)


class _OldMapOnly(LlmReplaceWorkflow):
    """Overrides ``generate_map_only`` with its signature, which never took ``session``."""

    def generate_map_only(self, dataframe: pd.DataFrame, **kwargs: Any) -> Any:
        return super().generate_map_only(dataframe, **kwargs)


_HOOKS: dict[str, tuple[type[ReplacementWorkflow], type[LlmReplaceWorkflow]]] = {
    "old-validated-maps": (ReplacementWorkflow, _OldValidatedMaps),
    "dropping-validated-maps": (ReplacementWorkflow, _DroppingMaps),
    "old-run-with-detection": (_OldSignature, LlmReplaceWorkflow),
    "dropping-run-with-detection": (_DroppingRunner, LlmReplaceWorkflow),
    "old-map-only": (ReplacementWorkflow, _OldMapOnly),
}


def _hooked(hook: str) -> Anonymizer:
    """An Anonymizer whose replace runner carries ``_HOOKS[hook]``'s overrides."""
    runner_type, llm_type = _HOOKS[hook]
    detector = cast(EntityDetectionWorkflow, _Detector())
    runner = runner_type(llm_workflow=llm_type(adapter=Anonymizer(detection_workflow=detector)._adapter))
    return Anonymizer(detection_workflow=detector, replace_runner=runner)


@pytest.mark.parametrize("hook", list(_HOOKS))
@pytest.mark.parametrize("session", [_UNSET, None, Session()], ids=["absent", "none", "off-session"])
@pytest.mark.parametrize("preview", [False, True], ids=["run", "preview"])
def test_an_old_hook_override_works_without_a_stable_session(
    tmp_path: Path, hook: str, session: Session | None, preview: bool
) -> None:
    texts, proposals = ["Alice lives in Paris."], {"Alice": "Bob", "Paris": "Lyon"}
    expected = _protect(tmp_path, texts, proposals, trace=True, preview=preview)
    output = _protect(tmp_path, texts, proposals, session, anonymizer=_hooked(hook), trace=True, preview=preview)
    pd.testing.assert_frame_equal(output, expected)
    assert list(output["text_replaced"]) == ["Bob lives in Lyon."]


_NAMED = {
    "old-validated-maps": "generate_validated_maps() on its replacement map workflow",
    "dropping-validated-maps": "generate_validated_maps() on its replacement map workflow",
    "old-run-with-detection": "run_with_detection()",
    "dropping-run-with-detection": "run_with_detection()",
    "instance-run-with-detection": "run_with_detection()",
}


@pytest.mark.parametrize("hook", list(_NAMED))
@pytest.mark.parametrize("preview", [False, True], ids=["run", "preview"])
def test_an_overriding_hook_is_rejected_under_a_stable_session_before_detection(
    tmp_path: Path, hook: str, preview: bool
) -> None:
    if hook == "instance-run-with-detection":
        anonymizer = _hooked("old-map-only")
        runner = anonymizer._replace_runner
        stock = runner.run_with_detection
        # Set on the instance, forwarding everything: still not the stock method.
        object.__setattr__(runner, "run_with_detection", lambda *a, **kw: stock(*a, **kw))
    else:
        anonymizer = _hooked(hook)
    source = tmp_path / "in.csv"
    pd.DataFrame({"text": ["Alice lives in Paris."]}).to_csv(source, index=False)
    session = Session(stable_replacements=True)
    call = anonymizer.preview if preview else anonymizer.run
    before = _Detector.calls
    message = f"A custom replace runner that overrides {_NAMED[hook]} cannot take part in a stable Session"
    with (
        patch.object(NddAdapter, "run_workflow") as model,
        pytest.raises(InvalidConfigError, match=re.escape(message)) as raised,
    ):
        call(config=AnonymizerConfig(replace=Substitute()), data=AnonymizerInput(source=str(source)), session=session)
    assert "forwards **kwargs" in str(raised.value)
    assert _Detector.calls == before
    assert model.call_count == 0
    assert session.size == 0


@pytest.mark.parametrize("preview", [False, True], ids=["run", "preview"])
def test_a_generate_map_only_override_keeps_a_stable_session(tmp_path: Path, preview: bool) -> None:
    """The Substitute route never calls ``generate_map_only``, so overriding it leaves the stock map step in place."""
    anonymizer, session, results = _hooked("old-map-only"), Session(stable_replacements=True), []
    _protect(tmp_path, ["Alice is here."], {"Alice": "Bob"}, session, results, anonymizer=anonymizer, preview=preview)
    second = _protect(
        tmp_path, ["Alice again."], {"Alice": "Zed"}, session, results, anonymizer=anonymizer, preview=preview
    )
    assert list(second["text_replaced"]) == ["Bob again."]
    assert results == [SessionCallChanges(added=1), SessionCallChanges(reused=1)]


def _dropping(method: Any) -> Any:
    """``method`` wrapped with ``functools.wraps`` so that it drops ``session``."""

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs.pop("session", None)
        return method(self, *args, **kwargs)

    return wrapper


def _runner(adapter: NddAdapter, case: str) -> ReplacementWorkflow:
    """A replace runner built as ``case`` names: stock forms first, then forms that must be rejected."""

    class Sub(ReplacementWorkflow):
        pass

    class SubSub(Sub):
        pass

    class Llm(LlmReplaceWorkflow):
        pass

    class LlmSub(Llm):
        pass

    class Rebound(ReplacementWorkflow):
        run_with_detection = ReplacementWorkflow.run_with_detection

    class Init(ReplacementWorkflow):
        def __init__(self) -> None:
            super().__init__(llm_workflow=LlmReplaceWorkflow(adapter=adapter))

    class InheritsDropping(_DroppingRunner):
        pass

    class InheritsDroppingMaps(_DroppingMaps):
        pass

    class Wrapped(ReplacementWorkflow):
        run_with_detection = _dropping(ReplacementWorkflow.run_with_detection)

    class WrappedMaps(LlmReplaceWorkflow):
        generate_validated_maps = _dropping(LlmReplaceWorkflow.generate_validated_maps)

    class ClassBound(ReplacementWorkflow):
        _llm_workflow = _DroppingMaps(adapter=adapter)
        run_with_detection = classmethod(ReplacementWorkflow.run_with_detection)

    def stock(runner_type: Any = ReplacementWorkflow, llm_type: Any = LlmReplaceWorkflow) -> ReplacementWorkflow:
        return runner_type(llm_workflow=llm_type(adapter=adapter))

    simple = {
        "stock": lambda: stock(),
        "subclass-of-subclass": lambda: stock(SubSub),
        "llm-subclass-of-subclass": lambda: stock(llm_type=LlmSub),
        "class-rebinding-the-stock-function": lambda: stock(Rebound),
        "map-only-override": lambda: stock(llm_type=_OldMapOnly),
        "init-only-override": lambda: Init(),
        "inherited-dropping-run-with-detection": lambda: stock(InheritsDropping),
        "inherited-dropping-validated-maps": lambda: stock(llm_type=InheritsDroppingMaps),
        "wrapped-run-with-detection": lambda: stock(Wrapped),
        "wrapped-validated-maps": lambda: stock(llm_type=WrappedMaps),
        "classmethod-run-with-detection": lambda: stock(ClassBound),
    }
    if case in simple:
        return simple[case]()
    runner = stock()
    llm = cast(LlmReplaceWorkflow, runner._llm_workflow)
    if case == "staticmethod-run-with-detection":
        runner = stock(type("Static", (ReplacementWorkflow,), {}))
        dropping = _dropping(ReplacementWorkflow.run_with_detection)
        setattr(type(runner), "run_with_detection", staticmethod(lambda *a, **kw: dropping(runner, *a, **kw)))
    elif case == "instance-run":
        object.__setattr__(runner, "run", lambda df, **kw: ReplacementWorkflow.run(runner, df, **kw))
    elif case == "instance-validated-maps":
        stock_maps = llm.generate_validated_maps
        object.__setattr__(llm, "generate_validated_maps", lambda df, **kw: stock_maps(df, **kw))
    elif case == "stock-run-with-detection-bound-to-another-runner":
        other = stock(llm_type=_DroppingMaps)
        object.__setattr__(
            runner, "run_with_detection", types.MethodType(ReplacementWorkflow.run_with_detection, other)
        )
    elif case == "stock-validated-maps-bound-to-another-workflow":
        bound = types.MethodType(LlmReplaceWorkflow.generate_validated_maps, _DroppingMaps(adapter=adapter))
        object.__setattr__(llm, "generate_validated_maps", bound)
    else:
        raise AssertionError(case)
    return runner


_STOCK_FORMS = [
    "stock",
    "subclass-of-subclass",
    "llm-subclass-of-subclass",
    "class-rebinding-the-stock-function",
    "map-only-override",
    "init-only-override",
]
_REJECTED_FORMS = {
    "inherited-dropping-run-with-detection": "run_with_detection()",
    "inherited-dropping-validated-maps": "generate_validated_maps() on its replacement map workflow",
    "wrapped-run-with-detection": "run_with_detection()",
    "wrapped-validated-maps": "generate_validated_maps() on its replacement map workflow",
    "staticmethod-run-with-detection": "run_with_detection()",
    "instance-run": "run()",
    "instance-validated-maps": "generate_validated_maps() on its replacement map workflow",
    "stock-run-with-detection-bound-to-another-runner": "run_with_detection()",
    "stock-validated-maps-bound-to-another-workflow": "generate_validated_maps() on its replacement map workflow",
    "classmethod-run-with-detection": "run_with_detection()",
}


@pytest.mark.parametrize("case", _STOCK_FORMS)
def test_a_stock_form_of_runner_takes_a_stable_session(tmp_path: Path, case: str) -> None:
    detector = cast(EntityDetectionWorkflow, _Detector())
    runner = _runner(Anonymizer(detection_workflow=detector)._adapter, case)
    anonymizer = Anonymizer(detection_workflow=detector, replace_runner=runner)
    session, results = Session(stable_replacements=True), []
    _protect(tmp_path, ["Alice is here."], {"Alice": "Bob"}, session, results, anonymizer=anonymizer)
    second = _protect(tmp_path, ["Alice again."], {"Alice": "Zed"}, session, results, anonymizer=anonymizer)
    assert list(second["text_replaced"]) == ["Bob again."]
    assert results == [SessionCallChanges(added=1), SessionCallChanges(reused=1)]


@pytest.mark.parametrize("strategy", [Redact(), Annotate(), Hash(key="test-key")], ids=["redact", "annotate", "hash"])
def test_a_stock_runner_takes_a_stable_session_with_a_local_strategy(tmp_path: Path, strategy: ReplaceMethod) -> None:
    detector = cast(EntityDetectionWorkflow, _Detector())
    runner = _runner(Anonymizer(detection_workflow=detector)._adapter, "stock")
    anonymizer = Anonymizer(detection_workflow=detector, replace_runner=runner)
    session, results = Session(stable_replacements=True), []
    _protect(tmp_path, ["Alice is here."], {}, session, results, strategy=strategy, anonymizer=anonymizer)
    assert results == [SessionCallChanges()]


@pytest.mark.parametrize("case", list(_REJECTED_FORMS))
def test_a_rebound_or_wrapped_hook_is_rejected_under_a_stable_session(tmp_path: Path, case: str) -> None:
    detector = _Detector()
    runner = _runner(Anonymizer(detection_workflow=cast(EntityDetectionWorkflow, detector))._adapter, case)
    anonymizer = Anonymizer(detection_workflow=cast(EntityDetectionWorkflow, detector), replace_runner=runner)
    source = tmp_path / "in.csv"
    pd.DataFrame({"text": ["Alice lives in Paris."]}).to_csv(source, index=False)
    session, before = Session(stable_replacements=True), _Detector.calls
    message = f"A custom replace runner that overrides {_REJECTED_FORMS[case]} cannot take part in a stable Session"
    with patch.object(NddAdapter, "run_workflow") as model, pytest.raises(InvalidConfigError, match=re.escape(message)):
        anonymizer.run(
            config=AnonymizerConfig(replace=Substitute()), data=AnonymizerInput(source=str(source)), session=session
        )
    assert _Detector.calls == before
    assert model.call_count == 0
    assert session.size == 0


def test_a_placeholder_is_reused_verbatim_in_a_later_call(tmp_path: Path) -> None:
    session = Session(stable_replacements=True)
    first = _protect(tmp_path, ["Alice and Carol."], {"Alice": "Bob", "Carol": "Bob"}, session)
    assert list(first["text_replaced"]) == ["Bob and [SUBSTITUTE_FIRST_NAME_1]."]
    # The Session holds Carol's placeholder against the normalized "carol"; it is not re-cased to "Carol".
    second = _protect(tmp_path, ["Carol is here."], {"Carol": "Zed"}, session)
    assert list(second["text_replaced"]) == ["[SUBSTITUTE_FIRST_NAME_1] is here."]


@pytest.mark.parametrize(
    ("label", "names", "placeholder"),
    [
        ("first_name", ("Alice", "alice", "Carol"), "[SUBSTITUTE_FIRST_NAME_1]"),
        ("prénom", ("Zoë", "zoë", "Élodie"), "[SUBSTITUTE_PRÉNOM_1]"),
        ("名前", ("Zoë", "zoë", "Élodie"), "[SUBSTITUTE_名前_1]"),
    ],
    ids=["ascii", "accented", "cjk"],
)
@pytest.mark.parametrize("consistency", ["record", "run"])
def test_a_placeholder_is_not_re_cased_for_a_case_variant_in_one_call(
    consistency: Any, label: str, names: tuple[str, str, str], placeholder: str
) -> None:
    value, variant, other = names

    def entry(original: str, synthetic: str) -> dict[str, str]:
        return {"original": original, "label": label, "synthetic": synthetic}

    # The value's proposal is a real value of the record, so it takes a placeholder; the variant shares it as written.
    maps = [{"replacements": [entry(value, other), entry(variant, other), entry(other, "Dan")]}]
    (folded,) = _fold_replacement_maps(maps, consistency)
    assert [e["synthetic"] for e in folded["replacements"]] == [placeholder, placeholder, "Dan"]


def test_is_placeholder_accepts_exactly_the_minted_tokens() -> None:
    for label in ("first_name", "email", "prénom", "名前", "-x y-", "!!"):
        for index in (1, 2, 10):
            token = _collision_safe_synthetic(label, index=index, protected_original_values=())
            assert _is_placeholder(token), token
    for value in ("[SUBSTITUTE_X_01]", "[SUBSTITUTE_A_0]", "[SUBSTITUTE_First_1]", "[REDACTED_FIRST_NAME]"):
        assert not _is_placeholder(value), value


@pytest.mark.parametrize(
    ("synthetic", "expected"),
    [
        ("[SUBSTITUTE_CITY_7]", True),
        ("[substitute_city_7]", True),
        ("[SUBSTITUTE_FIRST_NAME_7]", False),
        ("Lyon", False),
    ],
    ids=["own-label", "lower-case", "another-label", "plain"],
)
def test_replacement_placeholder_is_tied_to_the_label_and_case_insensitive(synthetic: str, expected: bool) -> None:
    frame = pd.DataFrame(
        {COL_REPLACEMENT_MAP: [{"replacements": [{"original": "Paris", "label": "CITY", "synthetic": synthetic}]}]}
    )
    result = AnonymizerResult(dataframe=frame, trace_dataframe=frame, resolved_text_column="text", failed_records=[])
    assert result.replacement_map(0) == (Replacement("Paris", "CITY", synthetic, expected),)


@pytest.mark.parametrize("preview", [False, True], ids=["run", "preview"])
def test_the_result_reads_each_rows_replacement_map(tmp_path: Path, preview: bool) -> None:
    """Each entry says whether its substitute is a placeholder, also after the trace goes through parquet."""
    session, kept = Session(stable_replacements=True), []
    texts = ["Alice and Carol.", "Paris."]
    _protect(tmp_path, texts, {"Alice": "Bob", "Carol": "Bob", "Paris": "Lyon"}, session, preview=preview, kept=kept)
    result = kept[0]
    assert result.session_changes == SessionCallChanges(added=3, placeholders_taken=1)
    expected = (
        Replacement("Alice", "first_name", "Bob", False),
        Replacement("Carol", "first_name", "[SUBSTITUTE_FIRST_NAME_1]", True),
    )
    assert result.replacement_map(0) == expected
    assert result.replacement_map(1) == (Replacement("Paris", "city", "Lyon", False),)
    with pytest.raises(IndexError):
        result.replacement_map(2)
    path = tmp_path / "trace.parquet"
    result.trace_dataframe[[COL_REPLACEMENT_MAP]].to_parquet(path)
    reloaded = dataclasses.replace(result, trace_dataframe=pd.read_parquet(path))
    assert reloaded.replacement_map(0) == expected
    no_map = dataclasses.replace(result, trace_dataframe=result.trace_dataframe.drop(columns=[COL_REPLACEMENT_MAP]))
    assert no_map.replacement_map(0) == ()


@pytest.mark.parametrize("stable", [False, True], ids=["no-session", "stable"])
def test_placeholder_is_a_token_shape_check_not_a_minted_count(tmp_path: Path, stable: bool) -> None:
    """A placeholder-shaped substitute the model proposes reads ``placeholder=True``; none is counted as minted."""
    session, kept = (Session(stable_replacements=True) if stable else _UNSET), []
    output = _protect(tmp_path, ["Alice is here."], {"Alice": "[SUBSTITUTE_FIRST_NAME_1]"}, session, kept=kept)
    assert list(output["text_replaced"]) == ["[SUBSTITUTE_FIRST_NAME_1] is here."]
    assert kept[0].replacement_map(0) == (Replacement("Alice", "first_name", "[SUBSTITUTE_FIRST_NAME_1]", True),)
    assert kept[0].session_changes == (SessionCallChanges(added=1) if stable else None)


# One string detected under two labels: a user name that is also a customer id.
_USER = "lei_wilson_4541"


def _user_entities(text: str, one_label: Any = _entities) -> dict[str, Any]:
    """``_entities`` plus ``_USER``'s first two occurrences, as a user name and then as a customer id."""
    spans = []
    starts = [m.start() for m in re.finditer(_USER, text)][:2]
    for start, label in zip(starts, ("user_name", "customer_id")):
        span = {"value": _USER, "label": label, "start_position": start, "end_position": start + len(_USER)}
        spans.append({"id": f"{label}-{start}", **span})
    return {"entities": [*spans, *one_label(text)["entities"]]}


@pytest.fixture
def two_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(f"{__name__}._entities", _user_entities)


@pytest.mark.usefixtures("two_labels")
def test_one_value_under_two_labels_keeps_one_substitute_across_calls(tmp_path: Path) -> None:
    session, results = Session(stable_replacements=True), []
    proposals = {_USER: "kim_park_1180", "Alice": "Bob"}
    first = _protect(tmp_path, [f"Ticket from {_USER}, account {_USER}."], proposals, session, results)
    # Within the call, the second label is not blocked by the first label of the same value.
    assert list(first["text_replaced"]) == ["Ticket from kim_park_1180, account kim_park_1180."]
    second = _protect(
        tmp_path, [f"{_USER} wrote again as {_USER}."], {**proposals, _USER: "max_roe_2207"}, session, results
    )
    assert list(second["text_replaced"]) == ["kim_park_1180 wrote again as kim_park_1180."]
    assert _slots(session) == {("user_name", _USER): "kim_park_1180", ("customer_id", _USER): "kim_park_1180"}
    assert _slots(session, "substitute-fallback") == {}
    assert results == [SessionCallChanges(added=2), SessionCallChanges(reused=2)]


@pytest.mark.usefixtures("two_labels")
def test_different_values_still_never_share_a_substitute_with_a_two_label_value(tmp_path: Path) -> None:
    session, results = Session(stable_replacements=True), []
    # In a call, Alice proposes the user's substitute; in the next, Carol proposes it again, re-cased.
    first = _protect(
        tmp_path, [f"{_USER} ({_USER}) and Alice."], {_USER: "kim_park_1180", "Alice": "kim_park_1180"}, session
    )
    assert list(first["text_replaced"]) == ["kim_park_1180 (kim_park_1180) and [SUBSTITUTE_FIRST_NAME_1]."]
    second = _protect(
        tmp_path, [f"{_USER} ({_USER}) and Carol."], {_USER: "x_1", "Carol": "KIM_PARK_1180"}, session, results
    )
    assert list(second["text_replaced"]) == ["kim_park_1180 (kim_park_1180) and [SUBSTITUTE_FIRST_NAME_2]."]
    assert results == [SessionCallChanges(added=1, reused=2, placeholders_taken=1)]


@pytest.mark.usefixtures("two_labels")
@pytest.mark.parametrize("label", ["user_name", "customer_id"])
def test_a_later_call_with_one_label_of_the_value_keeps_its_substitute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, label: str
) -> None:
    session, results = Session(stable_replacements=True), []
    # The substitute is cased, so the Session's assignment matches only once folded.
    first = _protect(tmp_path, [f"{_USER} is {_USER}."], {_USER: "Kim_Park_1180"}, session, results)
    assert list(first["text_replaced"]) == ["Kim_Park_1180 is Kim_Park_1180."]

    def one_label(text: str) -> dict[str, Any]:
        start = text.index(_USER)
        span = {"value": _USER, "label": label, "start_position": start, "end_position": start + len(_USER)}
        return {"entities": [{"id": "0", **span}]}

    monkeypatch.setattr(f"{__name__}._entities", one_label)
    for proposal in ("Kim_Park_1180", "max_roe_2207"):
        later = _protect(tmp_path, [f"{_USER} wrote again."], {_USER: proposal}, session, results)
        assert list(later["text_replaced"]) == ["Kim_Park_1180 wrote again."]
    assert _slots(session, "substitute-fallback") == {}
    assert results == [SessionCallChanges(added=2), SessionCallChanges(reused=1), SessionCallChanges(reused=1)]


def test_a_placeholder_after_the_same_value_emitted_its_substitute_is_tallied_as_session() -> None:
    """Jordan's last name holds Zed; his first name proposes Zed too, which the Session has as another original. The
    placeholder's cause is the Session, not a rival key: only Jordan himself emitted Zed."""
    from collections import Counter

    from anonymizer.engine.graph_core.session import BY_VALUE, SessionMode

    maps = [
        {
            "replacements": [
                {"original": "Jordan", "label": "last_name", "synthetic": "Lee"},
                {"original": "Jordan", "label": "first_name", "synthetic": "Zed"},
            ]
        }
    ]
    assignments = {(BY_VALUE, "last_name", "jordan"): "Zed", (BY_VALUE, "first_name", "zed"): "Amy"}
    tally: Counter[tuple[str, ...]] = Counter()
    view = SessionView(SessionMode(True, False), 1, assignments)
    _fold_replacement_maps(maps, "run", None, tally, view, [])
    assert {key[-1] for key in tally if key[-1].startswith("placeholder_")} == {"placeholder_session"}


@pytest.mark.usefixtures("two_labels")
@pytest.mark.parametrize("alone", [False, True], ids=["both-labels", "one-label"])
def test_a_restored_snapshot_with_one_substitute_for_two_values_is_not_hidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alone: bool
) -> None:
    session, results = Session(stable_replacements=True), []
    _protect(tmp_path, [f"{_USER} is {_USER}."], {_USER: "kim_park_1180"}, session)
    # An edited snapshot gives Bob the user's substitute too: a state the fold never writes.
    data = json.loads(session.snapshot().to_bytes())
    [slot] = [s for s in data["slots"] if s["label"] == "customer_id"]
    data["slots"].append({**slot, "label": "first_name", "value": "bob"})
    restored = Session.restore(SessionSnapshot.from_bytes(json.dumps(data).encode()))

    def one_label(text: str) -> dict[str, Any]:
        start = text.index(_USER)
        span = {"value": _USER, "label": "user_name", "start_position": start, "end_position": start + len(_USER)}
        return {"entities": [{"id": "0", **span}]}

    if alone:
        monkeypatch.setattr(f"{__name__}._entities", one_label)
    text = f"{_USER} wrote again." if alone else f"{_USER} wrote again as {_USER}."
    later = _protect(tmp_path, [text], {_USER: "kim_park_1180"}, restored, results)
    assert "kim_park_1180" not in later["text_replaced"][0]
    assert results[0].placeholders_session == (1 if alone else 2)


def test_assigned_to_other_value_ignores_the_same_value_under_another_label() -> None:
    from anonymizer.engine.graph_core.session import BY_VALUE, FALLBACK, SessionMode, SubjectKey

    def view(assignments: dict[Any, str]) -> SessionView:
        return SessionView(SessionMode(True, False), 1, assignments)

    slot = (BY_VALUE, "user_name", "lei")
    assert not view({(BY_VALUE, "customer_id", "lei"): "Kim"}).assigned_to_other_value("KIM", slot)
    assert not view({slot: "Kim"}).assigned_to_other_value("kim", slot)
    assert view({(BY_VALUE, "first_name", "bob"): "Kim"}).assigned_to_other_value("kim", slot)
    # Another subject is another holder, even for the same original.
    assert view({(SubjectKey("account"), "user_name", "lei"): "Kim"}).assigned_to_other_value("kim", slot)
    # Originals and the value are compared normalized.
    assert not view({(BY_VALUE, "customer_id", " LEI "): "Kim"}).assigned_to_other_value("kim", slot)
    assert view({(BY_VALUE, "first_name", "bob"): "Kim"}).assigned_to_other_value("  KIM ", slot)
    # A fallback entry names no value, so it always counts as another value's.
    assert view({(FALLBACK, "user_name", "kim"): "kim"}).assigned_to_other_value("kim", slot)


def _self_consistent(session: Session) -> list[tuple[str, str, str]]:
    """By-value slots whose assignment the Session's own view rules out for that slot; empty when consistent."""
    view = session._view()
    return [
        (slot[1], slot[2], entry.assignment)
        for slot, entry in session._slots.items()
        if slot[0] is not FALLBACK and view.assigned_to_other_value(entry.assignment, slot)
    ]


def _octavia_entities(text: str) -> dict[str, Any]:
    """Each Lumen and Octavia in ``text``: a first name, or an account id right after "account "."""
    spans = []
    for m in re.finditer("Lumen|Octavia", text):
        label = "account_id" if text[: m.start()].lower().endswith("account ") else "first_name"
        spans.append({"id": str(len(spans)), "value": m[0], "label": label, "start_position": m.start()})
        spans[-1]["end_position"] = m.end()
    return {"entities": spans}


_OCTAVIA = {"Octavia": "Mira", "Lumen": "Nora"}


@pytest.mark.parametrize(
    ("calls", "expected"),
    [
        ([["Lumen and Octavia.", "Account Octavia."]], ["Nora and Mira.", "Account [SUBSTITUTE_ACCOUNT_ID_1]."]),
        ([["Account Octavia.", "Lumen and Octavia."]], ["Account Mira.", "Nora and [SUBSTITUTE_FIRST_NAME_1]."]),
        ([["Lumen and Octavia, account Octavia."]], ["Nora and Mira, account [SUBSTITUTE_ACCOUNT_ID_1]."]),
        ([["Account Octavia."], ["Lumen and Octavia."]], ["Nora and [SUBSTITUTE_FIRST_NAME_1]."]),
    ],
    ids=["fallback-row-first", "account-row-first", "one-record", "across-calls"],
)
def test_a_value_under_two_labels_is_never_both_a_fallback_and_an_assignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: list[list[str]], expected: list[str]
) -> None:
    """Octavia holds Lumen as a first name; Lumen in the record sends her first name to the fallback Mira, while her
    account id proposes Mira too. Mira is one of the two, never both, so replays are a fixed point."""
    monkeypatch.setattr(f"{__name__}._entities", _octavia_entities)
    session = Session(stable_replacements=True)
    _protect(tmp_path, ["Octavia."], {"Octavia": "Lumen"}, session)
    for texts in calls:
        output = _protect(tmp_path, texts, _OCTAVIA, session)
        assert _self_consistent(session) == []
    assert list(output["text_replaced"]) == expected
    restored = Session.restore(SessionSnapshot.from_bytes(session.snapshot().to_bytes()))
    replayed, results = [], []
    for _ in range(2):
        before = restored.snapshot().to_bytes()
        replayed += list(_protect(tmp_path, ["Account Octavia."], _OCTAVIA, restored, results)["text_replaced"])
        assert restored.snapshot().to_bytes() == before
    assert replayed[0] == replayed[1]
    assert [r.added for r in results] == [0, 0]


def test_a_row_that_needs_a_fallback_gets_a_new_placeholder_on_each_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record-local fallback is never reused in a later call, by design, so such a row is not a fixed point."""
    monkeypatch.setattr(f"{__name__}._entities", _octavia_entities)
    session = Session(stable_replacements=True)
    _protect(tmp_path, ["Octavia."], {"Octavia": "Lumen"}, session)
    _protect(tmp_path, ["Lumen and Octavia.", "Account Octavia."], _OCTAVIA, session)
    restored = Session.restore(SessionSnapshot.from_bytes(session.snapshot().to_bytes()))
    replayed, results = [], []
    for _ in range(2):
        before = restored.snapshot().to_bytes()
        replayed += list(_protect(tmp_path, ["Lumen and Octavia."], _OCTAVIA, restored, results)["text_replaced"])
        assert restored.snapshot().to_bytes() != before
    assert replayed == ["Nora and [SUBSTITUTE_FIRST_NAME_1].", "Nora and [SUBSTITUTE_FIRST_NAME_2]."]
    assert [r.fallbacks for r in results] == [1, 1]
    assert _self_consistent(restored) == []


def _echo(value: str, answer: str | tuple[str, str]) -> tuple[str, str]:
    return answer if isinstance(answer, tuple) else (value, answer)


def _scripted(
    tmp_path: Path,
    texts: list[str],
    answers: list[dict[Any, Any]],
    session: Session | None,
    requests: list[list[tuple[list[str], str | None]]] | None = None,
    kept: list[Any] | None = None,
    preview: bool = False,
) -> pd.DataFrame:
    """One Substitute ``run`` (or ``preview``) whose model call ``i`` answers each requested value found in
    ``answers[i]``, by value or by (value, label), with a substitute or an echoed original and a substitute, and
    omits the rest; ``requests`` gets each call's rows as (requested values, retry context)."""
    script = list(answers)

    def run_workflow(self: NddAdapter, df: pd.DataFrame, **_: Any) -> WorkflowRunResult:
        out = df.copy().reset_index(drop=True)
        answer = script.pop(0)
        if requests is not None:
            requests.append(
                [
                    ([e["value"] for e in row[COL_ENTITIES_FOR_REPLACE]], row.get(COL_REPLACEMENT_RETRY_CONTEXT))
                    for _, row in out.iterrows()
                ]
            )
        out[COL_REPLACEMENT_MAP] = [
            {
                "replacements": [
                    {"original": echoed, "label": label, "synthetic": synthetic}
                    for e in row
                    for label in e["labels"]
                    for given in [answer.get((e["value"], label), answer.get(e["value"]))]
                    if given is not None
                    for echoed, synthetic in [_echo(e["value"], given)]
                ]
            }
            for row in out[COL_ENTITIES_FOR_REPLACE]
        ]
        return WorkflowRunResult(dataframe=out, failed_records=[], source_positions=tuple(range(len(out))))

    source = tmp_path / f"scripted-{len(list(tmp_path.iterdir()))}.csv"
    pd.DataFrame({"text": texts}).to_csv(source, index=False)
    anonymizer = Anonymizer(detection_workflow=cast(EntityDetectionWorkflow, _Detector()))
    call = anonymizer.preview if preview else anonymizer.run
    with patch.object(NddAdapter, "run_workflow", run_workflow):
        result = call(
            config=AnonymizerConfig(replace=Substitute()), data=AnonymizerInput(source=str(source)), session=session
        )
    assert script == [], "the model was called fewer times than scripted"
    if kept is not None:
        kept.append(result)
    return result.dataframe


def test_a_pair_the_session_holds_is_not_retried_when_the_model_omits_it(tmp_path: Path) -> None:
    session, kept = Session(stable_replacements=True), []
    _scripted(tmp_path, ["Alice lives here."], [{"Alice": "Bob"}], session)
    requests: list[Any] = []
    # The model omits Alice, whom the Session holds; one model call, no retry and no failed record.
    second = _scripted(tmp_path, ["Alice met Carol."], [{"Carol": "Dana"}], session, requests, kept)
    assert list(second["text_replaced"]) == ["Bob met Dana."]
    assert [[values for values, _ in call] for call in requests] == [[["Alice", "Carol"]]]
    assert kept[0].failed_records == []
    assert kept[0].session_changes == SessionCallChanges(added=1, reused=1)


def test_a_retried_pair_cannot_take_a_substitute_the_session_holds(tmp_path: Path) -> None:
    """The completeness retry's proposal goes through the Session's fold like a first-attempt one."""
    session = Session(stable_replacements=True)
    _protect(tmp_path, ["Alice lives in Paris."], {"Alice": "Bob", "Paris": "Lyon"}, session)
    answers = [[("Carol", "Dana")], [("Dave", "Bob")]]  # the first attempt omits Dave; the retry proposes Bob

    def run_workflow(self: NddAdapter, df: pd.DataFrame, **_: Any) -> WorkflowRunResult:
        out = df.copy().reset_index(drop=True)
        pairs = answers.pop(0)
        out[COL_REPLACEMENT_MAP] = [
            {"replacements": [{"original": o, "label": "first_name", "synthetic": s} for o, s in pairs]}
        ] * len(out)
        return WorkflowRunResult(dataframe=out, failed_records=[], source_positions=tuple(range(len(out))))

    source = tmp_path / "retry.csv"
    pd.DataFrame({"text": ["Carol met Dave."]}).to_csv(source, index=False)
    anonymizer = Anonymizer(detection_workflow=cast(EntityDetectionWorkflow, _Detector()))
    with patch.object(NddAdapter, "run_workflow", run_workflow):
        result = anonymizer.run(
            config=AnonymizerConfig(replace=Substitute()), data=AnonymizerInput(source=str(source)), session=session
        )
    assert answers == []
    assert list(result.dataframe["text_replaced"]) == ["Dana met [SUBSTITUTE_FIRST_NAME_1]."]
    assert _slots(session)[("first_name", "alice")] == "Bob"


def test_the_retry_requests_only_pairs_the_session_does_not_hold(tmp_path: Path) -> None:
    session, kept, requests = Session(stable_replacements=True), [], []
    _scripted(tmp_path, ["Alice lives here."], [{"Alice": "Bob"}], session)
    # The first attempt omits Alice (held) and Dave (new); only Dave is asked for again.
    second = _scripted(
        tmp_path, ["Alice met Carol and Dave."], [{"Carol": "Dana"}, {"Dave": "Eve"}], session, requests, kept
    )
    assert [[values for values, _ in call] for call in requests] == [[["Alice", "Carol", "Dave"]], [["Dave"]]]
    context = json.loads(requests[1][0][1])
    assert {"original": "Alice", "label": "first_name", "synthetic": "Bob"} in context["replacements"]
    assert list(second["text_replaced"]) == ["Bob met Dana and Eve."]
    assert kept[0].session_changes == SessionCallChanges(added=2, reused=1)


def test_a_record_that_fails_its_retry_adds_nothing_to_the_session(tmp_path: Path) -> None:
    session, kept = Session(stable_replacements=True), []
    # Row 0 never gets Carol, so it fails after the retry; row 1 is complete on the first attempt.
    output = _scripted(
        tmp_path, ["Alice met Carol.", "Dave."], [{"Alice": "Bob", "Dave": "Eve"}, {}], session, kept=kept
    )
    assert list(output["text_replaced"]) == ["Eve."]
    assert len(kept[0].failed_records) == 1
    assert _slots(session) == {("first_name", "dave"): "Eve"}
    assert kept[0].session_changes == SessionCallChanges(added=1)


def test_decisions_follow_kept_rows_when_a_middle_row_fails(tmp_path: Path) -> None:
    session, kept = Session(stable_replacements=True), []
    texts = ["Alice.", "Carol met Dave.", "Bob."]
    output = _scripted(tmp_path, texts, [{"Alice": "Eve", "Carol": "Dana", "Bob": "Finn"}, {}], session, kept=kept)
    assert list(output["text_replaced"]) == ["Eve.", "Finn."]
    assert len(kept[0].failed_records) == 1
    assert _slots(session) == {("first_name", "alice"): "Eve", ("first_name", "bob"): "Finn"}


def test_a_placeholder_shown_only_in_the_retry_context_is_not_counted(tmp_path: Path) -> None:
    session, kept, requests = Session(stable_replacements=True), [], []
    _scripted(tmp_path, ["Alice lives here."], [{"Alice": "Bob"}], session)
    # Alice's proposal is Carol, a value in the record, so the record-level retry context shows a placeholder; the
    # final fold gives Alice the Session's Bob instead.
    answers = [{"Alice": "Carol", "Carol": "Dana"}, {"Dave": "Eve"}]
    second = _scripted(tmp_path, ["Alice met Carol and Dave."], answers, session, requests, kept)
    assert "[SUBSTITUTE_FIRST_NAME_1]" in requests[1][0][1]
    assert list(second["text_replaced"]) == ["Bob met Dana and Eve."]
    assert kept[0].session_changes == SessionCallChanges(added=2, reused=1)


@pytest.mark.parametrize("session", [None, Session()], ids=["no-session", "off-session"])
def test_without_a_stable_session_the_retry_is_unchanged(tmp_path: Path, session: Session | None) -> None:
    requests: list[Any] = []
    output = _scripted(tmp_path, ["Alice met Carol."], [{"Carol": "Dana"}, {"Alice": "Zoe"}], session, requests)
    assert [[values for values, _ in call] for call in requests] == [[["Alice", "Carol"]], [["Alice"]]]
    assert list(output["text_replaced"]) == ["Zoe met Dana."]


def test_a_unicode_space_variant_reuses_the_session_substitute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(LABELS, "Acme Corp", "organization")
    monkeypatch.setitem(LABELS, "Acme\u00a0Corp", "organization")
    session, kept, requests = Session(stable_replacements=True), [], []
    _scripted(tmp_path, ["Acme Corp hired."], [{"Acme Corp": "Nova Labs"}], session)
    # The model echoes the no-break-space value with a plain space; it is reconciled to the requested original.
    answers = [{"Acme\u00a0Corp": ("Acme Corp", "Other Inc")}]
    second = _scripted(tmp_path, ["Acme\u00a0Corp hired."], answers, session, requests, kept)
    assert len(requests) == 1
    assert list(second["text_replaced"]) == ["Nova Labs hired."]
    assert kept[0].session_changes == SessionCallChanges(reused=1)
    assert _slots(session) == {("organization", "acme corp"): "Nova Labs"}


_POOL = ["Alice", "ALICE", "Bob", "Carol", "Dave", "Zed", "Paris"]


def _random_frame(rng: random.Random) -> pd.DataFrame:
    rows: list[dict[str, list[dict[str, Any]]]] = []
    texts: list[str] = []
    for _ in range(rng.randint(1, 4)):
        values = rng.sample(_POOL + ["Alice  ", "Jordan"], rng.randint(1, 3))
        rows.append(
            {
                "entities_by_value": [
                    {
                        "value": v,
                        "labels": ["first_name", "last_name"]
                        if v == "Jordan"
                        else [LABELS.get(v.strip(), "first_name")],
                    }
                    for v in values
                ]
            }
        )
        texts.append(" ".join(values))
    return pd.DataFrame({COL_TEXT: texts, COL_ENTITIES_BY_VALUE: rows, "tagged_text": texts})


@pytest.mark.parametrize("seed", range(300))
def test_stable_session_substitutes_stay_unique_and_fixed_across_retried_calls(
    seed: int, stub_model_configs: list[Any], stub_replace_model_selection: Any
) -> None:
    rng = random.Random(seed)
    proposals = ["Kim", "kim", "Lee", "Max", "Bob", "Carol", "Ned", "Ola", "Kim  "]

    def run_workflow(df: pd.DataFrame, **_: Any) -> WorkflowRunResult:
        out = df.copy().reset_index(drop=True)
        out[COL_REPLACEMENT_MAP] = [
            {
                "replacements": [
                    {"original": e["value"], "label": label, "synthetic": rng.choice(proposals)}
                    for e in row
                    for label in e["labels"]
                    if rng.random() < 0.7
                ]
            }
            for row in out[COL_ENTITIES_FOR_REPLACE]
        ]
        return WorkflowRunResult(dataframe=out, failed_records=[], source_positions=tuple(range(len(out))))

    adapter = Mock()
    adapter.run_workflow.side_effect = run_workflow
    workflow = LlmReplaceWorkflow(adapter=adapter)
    session = Session(stable_replacements=True)
    values_of: dict[str, set[str]] = {}  # normalized substitute -> normalized values it was emitted for
    substitute_of: dict[tuple[str, str], str] = {}  # key -> its normalized substitute
    for _ in range(rng.randint(2, 4)):
        frame = _random_frame(rng)
        result = workflow.generate_validated_maps(
            frame,
            model_configs=stub_model_configs,
            selected_models=stub_replace_model_selection,
            session=session._view(),
        )
        fallbacks = {d.value.casefold() for row in result.decisions for d in row if d.slot[0] is FALLBACK}
        assert result.source_positions is not None
        for position, rmap in zip(result.source_positions, result.dataframe[COL_REPLACEMENT_MAP]):
            entries = rmap["replacements"]
            requested = {
                (e["value"], label)
                for e in frame[COL_ENTITIES_BY_VALUE].iloc[position]["entities_by_value"]
                for label in e["labels"]
            }
            assert requested <= {(e["original"], e["label"]) for e in entries}
            for e in entries:
                value, synthetic = normalize(e["original"]), normalize(e["synthetic"])
                values_of.setdefault(synthetic, set()).add(value)
                key = (e["label"], value)
                if synthetic not in fallbacks:
                    assert substitute_of.setdefault(key, synthetic) == synthetic, (seed, key)
        session._accept([(d, "test") for row in result.decisions for d in row])
        assert _self_consistent(session) == [], seed
    assert all(len(values) == 1 for values in values_of.values()), (seed, values_of)


@pytest.mark.usefixtures("two_labels")
def test_the_cover_fills_only_from_the_session_never_from_the_response(tmp_path: Path) -> None:
    session, kept, requests = Session(stable_replacements=True), [], []
    # The model answers the value under one label only; the Session holds nothing, so the other label is retried
    # and, omitted again, fails the record rather than taking the response's substitute for the same value.
    answers = [{(_USER, "user_name"): "Maya"}, {}]
    output = _scripted(tmp_path, [f"{_USER} and {_USER}."], answers, session, requests, kept)
    assert [[values for values, _ in call] for call in requests] == [[[_USER]], [[_USER]]]
    assert list(output["text_replaced"]) == []
    assert len(kept[0].failed_records) == 1
    assert _slots(session) == {}


def test_placeholders_are_counted_once_per_call_in_run_mode(tmp_path: Path) -> None:
    session, kept, requests = Session(stable_replacements=True), [], []
    _scripted(tmp_path, ["Alice lives here."], [{"Alice": "Bob"}], session)
    # Carol's first-attempt proposal is the Session's Bob, so the run-mode fold gives her a placeholder; the row is
    # retried for Dave, and the placeholder is still counted once for the call.
    answers = [{"Carol": "Bob"}, {"Dave": "Eve"}]
    second = _scripted(tmp_path, ["Carol met Dave."], answers, session, requests, kept)
    assert len(requests) == 2
    assert list(second["text_replaced"]) == ["[SUBSTITUTE_FIRST_NAME_1] met Eve."]
    assert kept[0].session_changes == SessionCallChanges(added=2, placeholders_session=1)
