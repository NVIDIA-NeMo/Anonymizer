# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``DetectorResult`` contract at the ``Anonymizer`` seam: which detectors return it, how it is validated, and the
columns the engine renders from it for the later steps."""

from __future__ import annotations

import logging
from typing import Any, cast
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from anonymizer import DetectedSpan, DetectorResult, DetectorRow, InvalidDetectorResultError
from anonymizer.config.anonymizer_config import AnonymizerConfig, Rewrite, TextRecord, TextRecordsInput
from anonymizer.config.replace_strategies import Redact, Substitute
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_ENTITIES_BY_VALUE,
    COL_FINAL_ENTITIES,
    COL_LATENT_ENTITIES,
    COL_LEAKAGE_MASS,
    COL_REPLACEMENT_MAP,
    COL_REWRITTEN_TEXT,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
    COL_UTILITY_SCORE,
    COL_WEIGHTED_LEAKAGE_RATE,
    RECORD_ID_COLUMN,
)
from anonymizer.engine.detection.detection_workflow import (
    EntityDetectionResult,
    EntityDetectionWorkflow,
    _build_entities_by_value,
)
from anonymizer.engine.detection.result import render_detector_result, validate_detector_result
from anonymizer.engine.ndd.adapter import FailedRecord, NddAdapter, WorkflowRunResult
from anonymizer.engine.ndd.model_loader import load_default_model_selection
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow
from anonymizer.engine.rewrite.combined_rewrite_workflow import CombinedRewriteWorkflow
from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow
from anonymizer.engine.schemas import LatentEntitySchema
from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.interface.errors import AnonymizerWorkflowError
from tests.detector_results import builtin_stub

TEXTS = {"a": "Alice lives in Paris.", "b": "Bob met Alice."}
DATA = TextRecordsInput(records=[TextRecord(id=i, text=t) for i, t in TEXTS.items()])


def _span(text: str, value: str, label: str) -> DetectedSpan:
    start = text.index(value)
    return DetectedSpan(start, start + len(value), label, value)


def _result(frame: pd.DataFrame, **_: object) -> DetectorResult:
    rows = []
    for record_id, text in zip(frame[RECORD_ID_COLUMN], frame[COL_TEXT]):
        spans = tuple(_span(text, v, "first_name") for v in ("Alice", "Bob") if v in text)
        rows.append(DetectorRow(record_id=record_id, spans=spans))
    return DetectorResult(rows=tuple(rows))


class _SpansOnly:
    def __init__(self) -> None:
        self.frames: list[pd.DataFrame] = []

    def run(self, dataframe: pd.DataFrame, **_: object) -> DetectorResult:
        self.frames.append(dataframe)
        return _result(dataframe)


def _frame_of(frame: pd.DataFrame, **_: object) -> EntityDetectionResult:
    out = frame.copy()
    out[COL_FINAL_ENTITIES] = [{"entities": []} for _ in range(len(out))]
    out[COL_ENTITIES_BY_VALUE] = [{"entities_by_value": []} for _ in range(len(out))]
    return EntityDetectionResult(out, [], tuple(range(len(out))))


class _Subclass(EntityDetectionWorkflow):
    def run(self, dataframe: pd.DataFrame, **_: object) -> Any:
        return _frame_of(dataframe)


def _spec_mock() -> Mock:
    detector = Mock(spec=EntityDetectionWorkflow)
    detector.run.side_effect = _frame_of
    return detector


# --- which detectors return a frame -----------------------------------------------------------------------------------


def test_an_explicitly_passed_built_in_workflow_returns_its_frame() -> None:
    """Built-in means the class itself, passed or defaulted: its frame is accepted and its records released."""
    result = Anonymizer(detection_workflow=builtin_stub(_frame_of)).run(
        config=AnonymizerConfig(replace=Redact()), data=DATA
    )
    assert result.failed_records == []
    assert list(result.dataframe["text_replaced"]) == list(TEXTS.values())


def test_a_built_in_workflow_returning_something_else_is_a_workflow_error_not_a_detector_error() -> None:
    """Only a wiring fault can make the built-in class return anything but its frame: no user detector is at fault."""
    with patch.object(NddAdapter, "run_workflow", side_effect=AssertionError("no model call")):
        with pytest.raises(AnonymizerWorkflowError, match="EntityDetectionWorkflow.run returned DataFrame") as raised:
            Anonymizer(detection_workflow=builtin_stub(lambda frame, **_: frame)).run(
                config=AnonymizerConfig(replace=Redact()), data=DATA
            )
    assert not isinstance(raised.value, InvalidDetectorResultError)


@pytest.mark.parametrize("make", [_Subclass, _spec_mock], ids=["subclass", "spec-mock"])
def test_a_subclass_or_spec_mock_returning_a_frame_raises(make: Any) -> None:
    detector = make() if make is _spec_mock else make(adapter=Mock())
    with pytest.raises(InvalidDetectorResultError, match="return a DetectorResult"):
        Anonymizer(detection_workflow=detector).run(config=AnonymizerConfig(replace=Redact()), data=DATA)


# --- validation -------------------------------------------------------------------------------------------------------


def _span_with(**fields: Any) -> DetectedSpan:
    """A span with ``fields`` replaced, typed or not: the validator, not the type checker, must reject a bad one."""
    values: dict[str, Any] = {"start": 0, "end": 5, "label": "first_name", "value": "Alice", **fields}
    return DetectedSpan(**values)


def _source() -> pd.DataFrame:
    return pd.DataFrame({COL_TEXT: list(TEXTS.values()), RECORD_ID_COLUMN: list(TEXTS)})


@pytest.mark.parametrize(
    "result",
    [
        DetectorResult(rows=(DetectorRow("zz"),)),
        DetectorResult(rows=(DetectorRow("a"), DetectorRow("a"))),
        DetectorResult(rows=(DetectorRow("a"),), failed=(FailedRecord("a", "detect", "x"),)),
        DetectorResult(failed=(FailedRecord("0", "detect", "x"),)),
        DetectorResult(rows=(DetectorRow("a", extra_values=(("Alice", ""),)),)),
        DetectorResult(rows=(DetectorRow("a", spans=(_span_with(label=None),)),)),
        DetectorResult(rows=(DetectorRow("a", spans=(_span_with(value=5),)),)),
        DetectorResult(rows=(DetectorRow("a", spans=(_span_with(start=0.0),)),)),
        DetectorResult(rows=(DetectorRow("a", spans=(_span_with(score="high"),)),)),
        DetectorResult(rows=(DetectorRow("a", spans=(_span_with(source=None),)),)),
        DetectorResult(rows=(DetectorRow("a", latent=cast(Any, ({"value": "Paris"},))),)),
        DetectorResult(failed=(FailedRecord("a", "detect", cast(Any, None)),)),
    ],
    ids=[
        "unknown-id",
        "repeated-id",
        "answered-and-failed",
        "failure-without-an-input-id",
        "unlabelled-extra",
        "null-label",
        "non-text-value",
        "float-offset",
        "text-score",
        "null-source",
        "latent-dict",
        "null-reason",
    ],
)
def test_a_result_that_cannot_be_placed_is_a_batch_error(result: DetectorResult) -> None:
    with pytest.raises(InvalidDetectorResultError):
        validate_detector_result(_source(), result)


def test_a_record_neither_answered_nor_failed_fails_with_dropped_output() -> None:
    rendered = render_detector_result(_source(), DetectorResult(rows=(DetectorRow("b"),)))
    assert rendered.failed_records == [FailedRecord("a", "detect", "dropped_output")]
    assert type(rendered.failed_records[0].reason) is str  # a plain string, not an enum member
    assert list(rendered.dataframe[RECORD_ID_COLUMN]) == ["b"]
    assert rendered.source_positions == (1,)


def test_the_rendered_frame_has_every_column_later_steps_read() -> None:
    """``tagged_text``, ``_tag_notation`` and ``_latent_entities`` are read by the rewrite generation and disposition
    stages; without them Substitute and Rewrite stop with a DataDesigner config error."""
    latent = LatentEntitySchema.model_validate(
        {
            "category": "latent_identifier",
            "label": "home_location",
            "value": "Paris",
            "confidence": "high",
            "evidence": ["lives in Paris"],
            "rationale": "The text states where the person lives, in plain words.",
        }
    )
    text = TEXTS["a"]
    row = DetectorRow(
        "a", spans=(_span(text, "Alice", "first_name"),), extra_values=(("Paris", "city"),), latent=(latent,)
    )
    rendered = render_detector_result(
        _source(), DetectorResult(rows=(row,), failed=(FailedRecord("b", "detect", "x"),))
    )
    cells = rendered.dataframe.iloc[0]
    assert cells[COL_TAG_NOTATION]
    assert "Alice" in cells[COL_TAGGED_TEXT] and cells[COL_TAGGED_TEXT] != text
    assert cells[COL_LATENT_ENTITIES]["latent_entities"][0]["value"] == "Paris"
    assert cells[COL_ENTITIES_BY_VALUE]["entities_by_value"] == [
        {"value": "Alice", "labels": ["first_name"]},
        {"value": "Paris", "labels": ["city"]},
    ]
    assert cells[COL_FINAL_ENTITIES]["entities"][0]["value"] == "Alice"


# --- a spans-only custom detector through Substitute and both Rewrite routes ------------------------------------------


def test_a_spans_only_detector_runs_through_substitute() -> None:
    def run_workflow(self: NddAdapter, df: pd.DataFrame, **_: Any) -> WorkflowRunResult:
        out = df.copy().reset_index(drop=True)
        pairs = (("Alice", "Carol"), ("Bob", "Dan"))
        out[COL_REPLACEMENT_MAP] = [
            {"replacements": [{"original": v, "label": "first_name", "synthetic": s} for v, s in pairs if v in t]}
            for t in out[COL_TEXT]
        ]
        return WorkflowRunResult(dataframe=out, failed_records=[], source_positions=tuple(range(len(out))))

    anonymizer = Anonymizer(detection_workflow=_SpansOnly())  # ty: ignore[invalid-argument-type]
    with patch.object(NddAdapter, "run_workflow", run_workflow):
        result = anonymizer.run(config=AnonymizerConfig(replace=Substitute()), data=DATA)
    assert result.failed_records == []
    assert list(result.dataframe["text_replaced"]) == ["Carol lives in Paris.", "Dan met Carol."]


@pytest.mark.parametrize("combined", [False, True], ids=["staged", "combined"])
def test_a_spans_only_detector_gives_rewrite_the_columns_it_reads(combined: bool) -> None:
    seen: list[pd.DataFrame] = []

    def capture(runner: Any) -> Any:
        def _produce(rows: Any, **options: Any) -> Any:
            def produce() -> None:
                seen.append(rows.frame.copy())
                rows.frame[COL_REWRITTEN_TEXT] = ["rewritten"] * len(rows.frame)
                for col in (COL_UTILITY_SCORE, COL_LEAKAGE_MASS, COL_WEIGHTED_LEAKAGE_RATE):
                    rows.frame[col] = 1.0 if col == COL_UTILITY_SCORE else 0.0
                rows.frame[COL_ANY_HIGH_LEAKED] = False

            return produce

        runner._produce = _produce
        return runner

    anonymizer = Anonymizer(
        detection_workflow=_SpansOnly(),  # ty: ignore[invalid-argument-type]
        rewrite_runner=capture(RewriteWorkflow(Mock())),
        combined_rewrite_runner=capture(CombinedRewriteWorkflow(Mock())),
    )
    goal = PrivacyGoal(protect="where each person lives", preserve="the events between people")
    result = anonymizer.run(
        config=AnonymizerConfig(rewrite=Rewrite(privacy_goal=goal, use_combined_graph=combined)), data=DATA
    )
    assert result.failed_records == []
    frame = seen[0]
    for column in (COL_TAGGED_TEXT, COL_TAG_NOTATION, COL_LATENT_ENTITIES, COL_ENTITIES_BY_VALUE):
        assert column in frame.columns and frame[column].notna().all(), column


# --- the detection log line counts the final entities -----------------------------------------------------------------


def test_the_detection_log_line_counts_a_custom_detectors_spans(caplog: pytest.LogCaptureFixture) -> None:
    """It counted ``_detected_entities`` only, so a custom detector always logged 0 entities."""
    anonymizer = Anonymizer(detection_workflow=_SpansOnly())  # ty: ignore[invalid-argument-type]
    with caplog.at_level(logging.INFO, logger="anonymizer"):
        anonymizer.run(config=AnonymizerConfig(replace=Redact()), data=DATA)
    assert "Detection complete — 3 entities found across 2 records" in caplog.text


@pytest.mark.parametrize("mode", ["replace", "rewrite"])
def test_a_record_left_out_of_a_reordered_partial_result_fails_once_as_dropped_output(mode: str) -> None:
    """Rows for C then A, for inputs A, B and C: B is reported exactly once, as ``dropped_output``, by its id."""
    data = TextRecordsInput(records=[TextRecord(id=i, text=f"{i} text") for i in ("A", "B", "C")])

    class Partial:
        def run(self, dataframe: pd.DataFrame, **_: object) -> DetectorResult:
            return DetectorResult(rows=(DetectorRow("C"), DetectorRow("A")))

    class Stub(RewriteWorkflow):
        def _produce(self, rows: Any, **options: Any) -> Any:
            def produce() -> None:
                rows.frame[COL_REWRITTEN_TEXT] = ["rewritten"] * len(rows.frame)

            return produce

    anonymizer = Anonymizer(detection_workflow=Partial(), rewrite_runner=Stub(Mock()))  # ty: ignore[invalid-argument-type]
    config = AnonymizerConfig(replace=Redact()) if mode == "replace" else AnonymizerConfig(rewrite=Rewrite())
    result = anonymizer.run(config=config, data=data)
    assert result.failed_records == [FailedRecord("B", "detect", "dropped_output")]
    assert sorted(result.dataframe["text"]) == ["A text", "C text"]


# --- identity: every row and failure carries its input record id ---------------------------------------------------


def _frame_without_ids(frame: pd.DataFrame, **_: object) -> EntityDetectionResult:
    return _frame_of(frame.drop(columns=[RECORD_ID_COLUMN]))


@pytest.mark.parametrize("mode", ["replace", "rewrite"])
def test_a_built_in_frame_without_ids_gets_the_input_id_at_each_placed_row(mode: str) -> None:
    """The released row carries the id of the input row it answers, not a generated or missing one."""

    class Stub(RewriteWorkflow):
        def _produce(self, rows: Any, **options: Any) -> Any:
            def produce() -> None:
                rows.frame[COL_REWRITTEN_TEXT] = ["rewritten"] * len(rows.frame)

            return produce

    anonymizer = Anonymizer(detection_workflow=builtin_stub(_frame_without_ids), rewrite_runner=Stub(Mock()))
    config = AnonymizerConfig(replace=Redact()) if mode == "replace" else AnonymizerConfig(rewrite=Rewrite())
    result = anonymizer.run(config=config, data=DATA)
    assert list(result.trace_dataframe[RECORD_ID_COLUMN]) == list(TEXTS)


def test_a_built_in_frame_whose_ids_disagree_with_its_positions_raises() -> None:
    def swapped(frame: pd.DataFrame, **_: object) -> EntityDetectionResult:
        result = _frame_of(frame)
        result.dataframe[RECORD_ID_COLUMN] = list(reversed(list(frame[RECORD_ID_COLUMN])))
        return result

    with pytest.raises(AnonymizerWorkflowError, match="disagree"):
        Anonymizer(detection_workflow=builtin_stub(swapped)).run(config=AnonymizerConfig(replace=Redact()), data=DATA)


def test_the_substitute_map_step_needs_the_record_id_column() -> None:
    """The second uuid5 generator is gone: a frame without ids is a wiring fault, not a reason to invent ids."""
    by_value = {"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]}
    frame = pd.DataFrame({COL_TEXT: ["Alice"], COL_ENTITIES_BY_VALUE: [by_value]})
    with pytest.raises(AnonymizerWorkflowError, match=RECORD_ID_COLUMN):
        LlmReplaceWorkflow(adapter=Mock()).generate_validated_maps(
            frame, model_configs=[], selected_models=load_default_model_selection().replace, instructions=None
        )


# --- values: a built-in detector's missed repeat in Replace -------------------------------------------------------


def _builtin_first_only(frame: pd.DataFrame, **_: object) -> EntityDetectionResult:
    """The built-in detector spanning only the first occurrence of "Bob" and "Alice"."""
    out = frame.copy()
    entities = []
    for text in out[COL_TEXT]:
        spans = []
        for value in ("Bob", "Alice"):
            if value in text:
                start = text.index(value)
                spans.append(
                    {"value": value, "label": "first_name", "start_position": start, "end_position": start + len(value)}
                )
        entities.append({"entities": spans})
    out[COL_FINAL_ENTITIES] = entities
    out[COL_ENTITIES_BY_VALUE] = [_build_entities_by_value(e) for e in entities]
    return EntityDetectionResult(out, [], tuple(range(len(out))))


REPEATS = TextRecordsInput(records=[TextRecord(id="r", text="Bob met Alice, then Bob and Alice left.")])


def test_a_built_in_substitute_run_replaces_every_repeat_with_one_map_entry() -> None:
    """Added repeat spans take the first span's label, so the map keeps one entry per value and every occurrence
    gets the same substitute."""

    def run_workflow(self: NddAdapter, df: pd.DataFrame, **_: Any) -> WorkflowRunResult:
        out = df.copy().reset_index(drop=True)
        out[COL_REPLACEMENT_MAP] = [
            {
                "replacements": [
                    {"original": v, "label": "first_name", "synthetic": s}
                    for v, s in (("Bob", "Dan"), ("Alice", "Carol"))
                ]
            }
            for _ in range(len(out))
        ]
        return WorkflowRunResult(dataframe=out, failed_records=[], source_positions=tuple(range(len(out))))

    anonymizer = Anonymizer(detection_workflow=builtin_stub(_builtin_first_only))
    with patch.object(NddAdapter, "run_workflow", run_workflow):
        result = anonymizer.run(config=AnonymizerConfig(replace=Substitute()), data=REPEATS)
    assert result.failed_records == []
    released = result.dataframe["text_replaced"].iloc[0]
    assert released == "Dan met Carol, then Dan and Carol left."
    spans = result.dataframe["final_entities"].iloc[0]["entities"]
    assert sorted(e["source"] for e in spans) == ["detector", "detector", "repeat", "repeat"]
    assert {e["label"] for e in spans} == {"first_name"}


def test_the_built_in_rewrite_view_is_the_view_of_its_final_entities() -> None:
    """Rewrite releases a missed repeat to the rewriter as before, and the view it reads is exactly the view
    derived from the row's final entities."""
    seen: list[pd.DataFrame] = []

    class Stub(RewriteWorkflow):
        def _produce(self, rows: Any, **options: Any) -> Any:
            def produce() -> None:
                seen.append(rows.frame.copy())
                rows.frame[COL_REWRITTEN_TEXT] = ["rewritten"] * len(rows.frame)

            return produce

    anonymizer = Anonymizer(detection_workflow=builtin_stub(_builtin_first_only), rewrite_runner=Stub(Mock()))
    anonymizer.run(config=AnonymizerConfig(rewrite=Rewrite()), data=REPEATS)
    row = seen[0].iloc[0]
    assert row[COL_ENTITIES_BY_VALUE] == _build_entities_by_value(row[COL_FINAL_ENTITIES])
    assert all(e.get("source") != "repeat" for e in row[COL_FINAL_ENTITIES]["entities"])


def test_a_malformed_span_raises_before_any_model_call() -> None:
    """A span with a null label used to reach the graph (and, in Rewrite, a model) before failing: it now raises."""

    class Malformed:
        def run(self, dataframe: pd.DataFrame, **_: object) -> DetectorResult:
            rows = [DetectorRow(i, spans=(_span_with(label=None),)) for i in dataframe[RECORD_ID_COLUMN]]
            return DetectorResult(rows=tuple(rows))

    adapter_calls: list[str] = []
    with patch.object(NddAdapter, "run_workflow", lambda *a, **k: adapter_calls.append("x")):
        with pytest.raises(InvalidDetectorResultError, match="malformed span"):
            Anonymizer(detection_workflow=Malformed()).run(  # ty: ignore[invalid-argument-type]
                config=AnonymizerConfig(rewrite=Rewrite()), data=DATA
            )
    assert adapter_calls == []


def _builtin_spans(*spans: tuple[str, str, int]) -> Any:
    """A built-in detector returning ``(value, label, nth occurrence)`` spans for every row."""

    def run(frame: pd.DataFrame, **_: object) -> EntityDetectionResult:
        out = frame.copy()
        entities = []
        for text in out[COL_TEXT]:
            found: list[dict[str, object]] = []
            for value, label, nth in spans:
                at = -1
                for _occurrence in range(nth + 1):
                    at = text.index(value, at + 1)
                start = at
                found.append(
                    {"value": value, "label": label, "start_position": start, "end_position": start + len(value)}
                )
            entities.append({"entities": found})
        out[COL_FINAL_ENTITIES] = entities
        out[COL_ENTITIES_BY_VALUE] = [_build_entities_by_value(e) for e in entities]
        return EntityDetectionResult(out, [], tuple(range(len(out))))

    return builtin_stub(run)


def test_a_built_in_repeat_that_crosses_a_kept_span_fails_the_row() -> None:
    """The first "New York" crosses the kept span "York University": no span is added there, and Replace fails the
    row with ``unspanned_value`` rather than release it."""
    data = TextRecordsInput(records=[TextRecord(id="r", text="New York University is in New York.")])
    detector = _builtin_spans(("York University", "organization", 0), ("New York", "city", 0))
    detector.run.side_effect = _shift_new_york(detector.run.side_effect)
    result = Anonymizer(detection_workflow=detector).run(config=AnonymizerConfig(replace=Redact()), data=data)
    assert result.failed_records == [FailedRecord("r", "detect", "unspanned_value")]
    assert type(result.failed_records[0].reason) is str  # a plain string, not an enum member
    assert result.dataframe.empty


def _shift_new_york(run: Any) -> Any:
    """Move the "New York" span to the second occurrence (the first is inside "New York University")."""

    def shifted(frame: pd.DataFrame, **kwargs: object) -> EntityDetectionResult:
        result = run(frame, **kwargs)
        for cell in result.dataframe[COL_FINAL_ENTITIES]:
            for entity in cell["entities"]:
                if entity["value"] == "New York":
                    entity["start_position"], entity["end_position"] = 26, 34
        result.dataframe[COL_ENTITIES_BY_VALUE] = [
            _build_entities_by_value(c) for c in result.dataframe[COL_FINAL_ENTITIES]
        ]
        return result

    return shifted


def test_a_built_in_repeat_takes_the_first_returned_label_when_labels_conflict() -> None:
    data = TextRecordsInput(records=[TextRecord(id="r", text="Paris met Paris in Paris.")])
    detector = _builtin_spans(("Paris", "first_name", 0), ("Paris", "city", 2))
    result = Anonymizer(detection_workflow=detector).run(config=AnonymizerConfig(replace=Redact()), data=data)
    assert result.failed_records == []
    added = [e for e in result.dataframe["final_entities"].iloc[0]["entities"] if e["source"] == "repeat"]
    assert [(e["start_position"], e["label"]) for e in added] == [(10, "first_name")]


def test_a_built_in_case_variant_gains_no_repeat_span() -> None:
    """A repeat span needs the detected spelling: "alice" after a detected "Alice" keeps its text, though the map
    offers it a substitute."""
    pairs: list[object] = []

    def run_workflow(self: NddAdapter, df: pd.DataFrame, **_: Any) -> WorkflowRunResult:
        out = df.copy().reset_index(drop=True)
        pairs.extend(out[COL_ENTITIES_BY_VALUE])
        out[COL_REPLACEMENT_MAP] = [
            {
                "replacements": [
                    {"original": "Alice", "label": "first_name", "synthetic": "Carol"},
                    {"original": "alice", "label": "first_name", "synthetic": "dana"},
                ]
            }
        ] * len(out)
        return WorkflowRunResult(dataframe=out, failed_records=[], source_positions=tuple(range(len(out))))

    data = TextRecordsInput(records=[TextRecord(id="r", text="Alice met alice.")])
    with patch.object(NddAdapter, "run_workflow", run_workflow):
        result = Anonymizer(detection_workflow=_builtin_spans(("Alice", "first_name", 0))).run(
            config=AnonymizerConfig(replace=Substitute()), data=data
        )
    assert result.failed_records == []
    assert result.dataframe["text_replaced"].iloc[0] == "Carol met alice."


@pytest.mark.parametrize(
    "result",
    [
        DetectorResult(rows=cast(Any, None)),
        DetectorResult(failed=cast(Any, 3)),
        DetectorResult(rows=cast(Any, (DetectorRow("a") for _ in range(1)))),
        DetectorResult(rows=(DetectorRow("a", spans=cast(Any, None)),)),
        DetectorResult(rows=(DetectorRow("a", extra_values=cast(Any, 1)),)),
        DetectorResult(rows=(DetectorRow("a", latent=cast(Any, iter(()))),)),
    ],
    ids=["rows-none", "failed-int", "rows-generator", "spans-none", "extra-int", "latent-iterator"],
)
def test_a_result_whose_collections_are_not_sequences_is_a_batch_error(result: DetectorResult) -> None:
    """Only tuples and lists are accepted: ``None`` or a number raised a plain ``TypeError``, and a generator would
    be consumed by validation and render nothing."""
    with pytest.raises(InvalidDetectorResultError, match="tuple or a list"):
        validate_detector_result(_source(), result)
