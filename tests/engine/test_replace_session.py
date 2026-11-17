# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``Anonymizer.run``/``preview`` with ``session=``: under a stable Session Substitute carries substitutes across calls,
and no substitute the Session has seen, held or used once in a record, is given to another value in a later call;
Redact, Annotate and Hash leave the Session unchanged; Rewrite and overriding runners are rejected before detection."""

from __future__ import annotations

import contextlib
import json
import logging
import re
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
    COL_FINAL_ENTITIES,
    COL_LEAKAGE_MASS,
    COL_REPLACEMENT_MAP,
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
from anonymizer.engine.graph_core.session import Session, SessionChanges, SessionView
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
    results: list[SessionChanges | None] | None = None,
    rejecting: bool = False,
    trace: bool = False,
    preview: bool = False,
    strategy: ReplaceMethod | None = None,
    anonymizer: Anonymizer | None = None,
    detector: Any = None,
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
        results.append(result._session_changes)
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
    assert results == [SessionChanges(accepted=2), SessionChanges(accepted=1, reused=2)]
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
    assert results == [SessionChanges(), SessionChanges()]
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
    assert results == [SessionChanges(accepted=3), SessionChanges(accepted=1)]


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
    assert results[1:] == [
        SessionChanges(accepted=2, reused=1),
        SessionChanges(accepted=2, reused=2),
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
    assert results == [SessionChanges(accepted=1), SessionChanges(reused=1)]


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
    assert results == [SessionChanges(accepted=2)]
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


@pytest.mark.parametrize("strategy", [Redact(), Annotate(), Hash()], ids=["redact", "annotate", "hash"])
def test_a_local_strategy_leaves_a_stable_session_unchanged(tmp_path: Path, strategy: ReplaceMethod) -> None:
    session, results = Session(stable_replacements=True), []
    _protect(tmp_path, ["Alice lives in Paris."], {"Alice": "Bob", "Paris": "Lyon"}, session, results)
    before = session.snapshot().to_bytes()
    texts = ["Alice met Carol in Paris.", "Carol saw Alice."]
    plain = _protect(tmp_path, texts, {}, strategy=strategy, trace=True)
    with_session = _protect(tmp_path, texts, {}, session, results, strategy=strategy, trace=True)
    pd.testing.assert_frame_equal(plain, with_session)
    assert results[1] == SessionChanges()
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
    assert results == [SessionChanges(accepted=1), SessionChanges(reused=1)]
    assert "session changes: 0 accepted, 1 reused, 0 revised, 0 ignored" in caplog.text


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
    assert with_session._session_changes is None


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
    assert results == [SessionChanges(accepted=2), SessionChanges(reused=2)]


class _OldSignature(ReplacementWorkflow):
    """Overrides ``run_with_detection`` with the signature it had before ``session``."""

    def run_with_detection(  # ty: ignore[invalid-method-override]
        self,
        dataframe: pd.DataFrame,
        *,
        detect: Any,
        replace_method: ReplaceMethod,
        model_configs: Any,
        selected_models: Any,
        preview_num_records: int | None = None,
    ) -> ReplacementResult:
        return super().run_with_detection(
            dataframe,
            detect=detect,
            replace_method=replace_method,
            model_configs=model_configs,
            selected_models=selected_models,
            preview_num_records=preview_num_records,
        )


def test_an_old_run_with_detection_override_works_without_a_session(tmp_path: Path) -> None:
    detector = cast(EntityDetectionWorkflow, _Detector())
    runner = _OldSignature(llm_workflow=LlmReplaceWorkflow(adapter=Anonymizer(detection_workflow=detector)._adapter))
    anonymizer = Anonymizer(detection_workflow=detector, replace_runner=runner)
    texts, proposals = ["Alice lives in Paris."], {"Alice": "Bob", "Paris": "Lyon"}
    expected = _protect(tmp_path, texts, proposals, trace=True)
    pd.testing.assert_frame_equal(_protect(tmp_path, texts, proposals, anonymizer=anonymizer, trace=True), expected)


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
