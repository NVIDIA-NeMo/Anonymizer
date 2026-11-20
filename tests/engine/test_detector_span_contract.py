# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The span contract for a custom ``detection_workflow``, shape by shape, through the public API.

Identical spans collapse into one and a span contained in another gives way to the outer one; every other bad span
fails its record at ``detect`` with a named reason. Either way no protected value reaches the output.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, Rewrite
from anonymizer.config.replace_strategies import Annotate, Hash, Redact, Substitute
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_ENTITIES_BY_VALUE,
    COL_FINAL_ENTITIES,
    COL_LEAKAGE_MASS,
    COL_REPLACEMENT_MAP,
    COL_REWRITE_BASELINE_TEXT,
    COL_REWRITE_DISPOSITION_BLOCK,
    COL_REWRITE_REPLACEMENT_READY,
    COL_REWRITTEN_TEXT,
    COL_TAG_NOTATION,
    COL_TEXT,
    COL_UTILITY_SCORE,
    COL_WEIGHTED_LEAKAGE_RATE,
)
from anonymizer.engine.detection.detect_step import _with_entities, accepted_mentions, covered_spans
from anonymizer.engine.detection.detection_workflow import (
    EntityDetectionResult,
    EntityDetectionWorkflow,
    _build_entities_by_value,
)
from anonymizer.engine.detection.result import DetectorResult
from anonymizer.engine.graph_core.authoring import GraphScope
from anonymizer.engine.ndd.adapter import FailedRecord, NddAdapter, WorkflowRunResult
from anonymizer.engine.ndd.workflow_utils import select_seed_cols
from anonymizer.engine.replace.replace_runner import ReplacementWorkflow
from anonymizer.engine.rewrite.combined_rewrite_workflow import CombinedRewriteWorkflow
from anonymizer.engine.rewrite.rewrite_generation import _prepare_rewrite_tagged_text
from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow
from anonymizer.engine.schemas import EntitiesSchema
from anonymizer.engine.value_matching import count_values, entity_spans
from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.interface.errors import InvalidConfigError
from tests.detector_results import as_detector_result, builtin_stub, input_record_ids

TEXT = "Alice Smith lives in Paris."  # Alice 0-5, Smith 6-11, Alice Smith 0-11, Paris 21-26
CONTROL = "Bob Jones lives in Rome."
PROTECTED = ("Alice", "Smith", "Paris", "Bob", "Jones", "Rome")


def _e(value: str, label: str, start: int, end: int) -> dict[str, Any]:
    return {"id": f"{label}-{start}", "value": value, "label": label, "start_position": start, "end_position": end}


NAME = _e("Alice Smith", "full_name", 0, 11)
CITY = _e("Paris", "city", 21, 26)
CONTROL_ENTITIES = [_e("Bob Jones", "full_name", 0, 9), _e("Rome", "city", 19, 23)]
FULL = "[REDACTED_FULL_NAME] lives in [REDACTED_CITY]."

# shape -> (entities, redacted text when the record is kept, or the detect reason when it fails)
SHAPES: dict[str, tuple[list[dict[str, Any]], str]] = {
    "nested_outer_first": ([NAME, _e("Alice", "first_name", 0, 5), CITY], FULL),
    "nested_inner_first": ([_e("Alice", "first_name", 0, 5), NAME, CITY], FULL),
    "nested_inner_later": ([NAME, _e("Smith", "last_name", 6, 11), CITY], FULL),
    "identical_same_label": ([NAME, NAME, CITY], FULL),
    "identical_other_label": ([NAME, _e("Alice Smith", "person", 0, 11), CITY], FULL),
    "touching": (
        [_e("Alice", "first_name", 0, 5), _e(" Smith", "last_name", 5, 11), CITY],
        "[REDACTED_FIRST_NAME][REDACTED_LAST_NAME] lives in [REDACTED_CITY].",
    ),
    "partial_overlap": ([NAME, _e("Smith lives", "other", 6, 17), CITY], "overlapping_output_ownership"),
    "offset_shifted": ([NAME, _e("Paris", "city", 22, 27)], "mention_slice_mismatch"),
    "case": ([NAME, _e("paris", "city", 21, 26)], "mention_slice_mismatch"),
    "whitespace": ([NAME, _e("Paris ", "city", 21, 26)], "mention_slice_mismatch"),
    "empty_value": ([NAME, _e("", "city", 21, 26)], "mention_slice_mismatch"),
    "out_of_range": ([NAME, _e("Paris.", "city", 21, 28)], "mention_out_of_range"),
    "negative": ([NAME, _e("Paris", "city", -1, 4)], "mention_out_of_range"),
    "zero_width_empty": ([NAME, _e("", "city", 21, 21), CITY], "mention_out_of_range"),
    "zero_width_valued": ([NAME, _e("Paris", "city", 21, 21)], "mention_out_of_range"),
    # A bad span inside a good one is not dropped as contained: it fails with its own reason.
    "nested_mismatch": ([NAME, _e("alice", "first_name", 0, 5), CITY], "mention_slice_mismatch"),
    # A span without a label gets no replacement: it never stands in for a labelled one.
    "identical_unlabelled_first": ([_e("Alice Smith", "", 0, 11), NAME, CITY], FULL),
    "nested_unlabelled_inner": ([NAME, _e("Alice", "", 0, 5), CITY], FULL),
    "identical_both_unlabelled": (
        [_e("Alice Smith", "", 0, 11), _e("Alice Smith", "", 0, 11), CITY],
        "mention_unlabelled",
    ),
    "nested_unlabelled_outer": (
        [_e("Alice Smith", "", 0, 11), _e("Alice", "first_name", 0, 5), CITY],
        "mention_unlabelled",
    ),
    # A span without a label would get no replacement: left on its own, it fails its record.
    "unlabelled": ([NAME, _e("Paris", "", 21, 26)], "mention_unlabelled"),
    # Containment does not excuse a partial overlap elsewhere in the record.
    "nested_and_partial": (
        [NAME, _e("Alice", "first_name", 0, 5), _e("Smith lives", "other", 6, 17), CITY],
        "overlapping_output_ownership",
    ),
}


def _rid(source: Path, row: int) -> str:
    """The input record id of ``source``'s row ``row``: the seam's uuid5, as every failure names it."""
    return input_record_ids(AnonymizerInput(source=str(source)))[row]


def _as_builtin(detector: _FrameDetector) -> EntityDetectionWorkflow:
    """``detector``'s frame returned by the built-in workflow: the frame-encoding and payload-parse rules below apply
    to the built-in detector's frame only, since a ``DetectorResult`` has no encodings or payloads to get wrong."""
    return builtin_stub(lambda dataframe, **_: detector.frame(dataframe))


class _FrameDetector:
    """A custom detector described by the frame it would have returned: ``run`` returns that frame's spans and
    by-value pairs as a ``DetectorResult``."""

    def frame(self, dataframe: pd.DataFrame) -> EntityDetectionResult:
        raise NotImplementedError

    def run(self, dataframe: pd.DataFrame, **_: object) -> DetectorResult:
        return as_detector_result(dataframe, self.frame(dataframe))


class _Detector(_FrameDetector):
    def __init__(self, entities: list[dict[str, Any]]) -> None:
        self.by_text = {TEXT: entities, CONTROL: CONTROL_ENTITIES}

    def frame(self, dataframe: pd.DataFrame) -> EntityDetectionResult:
        frame = dataframe.copy()
        entities = [{"entities": self.by_text[str(text)]} for text in frame[COL_TEXT]]
        frame[COL_FINAL_ENTITIES] = entities
        frame[COL_ENTITIES_BY_VALUE] = [_build_entities_by_value(e) for e in entities]
        return EntityDetectionResult(frame, [], tuple(range(len(frame))))


@pytest.mark.parametrize("shape", list(SHAPES))
def test_custom_detector_span_shape(shape: str, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    entities, expected = SHAPES[shape]
    source = tmp_path / "in.csv"
    pd.DataFrame({"text": [TEXT, CONTROL]}).to_csv(source, index=False)
    anonymizer = Anonymizer(detection_workflow=cast(EntityDetectionWorkflow, _Detector(entities)))
    with caplog.at_level(logging.DEBUG, logger="anonymizer"):
        result = anonymizer.run(config=AnonymizerConfig(replace=Redact()), data=AnonymizerInput(source=str(source)))

    control = "[REDACTED_FULL_NAME] lives in [REDACTED_CITY]."
    if expected.startswith("["):
        assert list(result.dataframe["text_replaced"]) == [expected, control]
        assert result.failed_records == []
    else:
        assert list(result.dataframe["text_replaced"]) == [control]
        assert result.failed_records == [FailedRecord(_rid(source, 0), "detect", expected)]
        assert f"1 record(s) failed during pipeline processing: detect/{expected}=1." in caplog.text
        # A record failed at detect is counted under detection, not replacement.
        assert "across 1 records (1 failed)" in caplog.text and "Replacement complete (0 failed)" in caplog.text
    shown = [*result.dataframe["text_replaced"], *(f.reason for f in result.failed_records), caplog.text]
    assert not [value for value in PROTECTED for text in shown if value in text]


def _kept(entities: list[dict[str, Any]]) -> list[tuple[int, int, str]]:
    datum = GraphScope().datum(TEXT)
    return [
        (m.start, m.end, m.label) for m in accepted_mentions(datum, EntitiesSchema.from_raw({"entities": entities}))
    ]


def test_identical_spans_keep_the_first_label() -> None:
    other = _e("Alice Smith", "person", 0, 11)
    assert _kept([NAME, other, CITY]) == [(0, 11, "full_name"), (21, 26, "city")]
    assert _kept([other, NAME, CITY]) == [(0, 11, "person"), (21, 26, "city")]


def test_contained_spans_give_way_to_the_outer_span() -> None:
    inner = [_e("Alice", "first_name", 0, 5), _e("Smith", "last_name", 6, 11)]
    assert _kept([*inner, CITY, NAME]) == [(0, 11, "full_name"), (21, 26, "city")]


def test_partial_and_bad_spans_are_left_for_acceptance_to_reject() -> None:
    partial = _e("Smith lives", "other", 6, 17)
    assert _kept([NAME, partial]) == [(0, 11, "full_name"), (6, 17, "other")]
    zero = _e("", "city", 21, 21)
    assert _kept([NAME, CITY, zero]) == [(0, 11, "full_name"), (21, 26, "city"), (21, 21, "city")]


ACCEPTED = [shape for shape, (_, expected) in SHAPES.items() if expected.startswith("[")]
# The spans each accepted shape keeps, as (start, end, label): what the output entities and later stages see.
KEPT = {shape: [(0, 11, "full_name"), (21, 26, "city")] for shape in ACCEPTED}
KEPT["touching"] = [(0, 5, "first_name"), (5, 11, "last_name"), (21, 26, "city")]


def _spans(raw: object) -> list[tuple[int, int, str]]:
    return [(e.start_position, e.end_position, e.label) for e in EntitiesSchema.from_raw(raw).entities]


def _requested_pairs(raw: object) -> set[tuple[str, str]]:
    parsed = raw if isinstance(raw, dict) else json.loads(str(raw))
    return {(item["value"], label) for item in parsed["entities_by_value"] for label in item["labels"]}


def _substitute_map(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {"replacements": [{"original": v, "label": lab, "synthetic": f"S{len(v)}{lab[:2]}"} for v, lab in pairs]}
        for pairs in map(_requested_pairs, df[COL_ENTITIES_BY_VALUE])
    ]


@pytest.mark.parametrize("strategy", ["redact", "hash", "substitute"])
@pytest.mark.parametrize("shape", ACCEPTED)
def test_accepted_shapes_are_protected_and_collapsed_in_the_output(shape: str, strategy: str, tmp_path: Path) -> None:
    source = tmp_path / "in.csv"
    pd.DataFrame({"text": [TEXT, CONTROL]}).to_csv(source, index=False)
    requested: list[set[tuple[str, str]]] = []

    def run_workflow(self: NddAdapter, df: pd.DataFrame, *, workflow_name: str, **_: Any) -> WorkflowRunResult:
        assert workflow_name == "replace-map-generation", workflow_name
        out = df.copy().reset_index(drop=True)
        requested.extend(map(_requested_pairs, out[COL_ENTITIES_BY_VALUE]))
        out[COL_REPLACEMENT_MAP] = _substitute_map(out)
        return WorkflowRunResult(dataframe=out, failed_records=[], source_positions=tuple(range(len(out))))

    method = {"redact": Redact(), "hash": Hash(key="s3cr3t"), "substitute": Substitute()}[strategy]
    anonymizer = Anonymizer(detection_workflow=cast(EntityDetectionWorkflow, _Detector(SHAPES[shape][0])))
    with patch.object(NddAdapter, "run_workflow", run_workflow):
        result = anonymizer.run(config=AnonymizerConfig(replace=method), data=AnonymizerInput(source=str(source)))

    assert result.failed_records == []
    assert not [v for v in PROTECTED for text in result.dataframe["text_replaced"] if v in text]
    assert _spans(result.dataframe["final_entities"].iloc[0]) == KEPT[shape]
    if strategy == "substitute":  # the map is requested for the kept spans only
        assert requested[0] == {(TEXT[a:b], label) for a, b, label in KEPT[shape]}


@pytest.mark.parametrize("shape", ACCEPTED)
def test_rewrite_sees_the_collapsed_spans_and_applies_its_replacements(shape: str, tmp_path: Path) -> None:
    source = tmp_path / "in.csv"
    pd.DataFrame({"text": [TEXT, CONTROL]}).to_csv(source, index=False)
    seen: list[dict[str, Any]] = []

    class StubRewrite(RewriteWorkflow):
        def _produce(self, rows: Any, **options: Any) -> Any:
            def produce() -> None:
                seen.extend(rows.frame.to_dict("records"))
                rows.frame[COL_REWRITTEN_TEXT] = ["rewritten"] * len(rows.frame)
                for col in (COL_UTILITY_SCORE, COL_LEAKAGE_MASS, COL_WEIGHTED_LEAKAGE_RATE):
                    rows.frame[col] = 1.0 if col == COL_UTILITY_SCORE else 0.0
                rows.frame[COL_ANY_HIGH_LEAKED] = False

            return produce

    anonymizer = Anonymizer(
        detection_workflow=cast(EntityDetectionWorkflow, _Detector(SHAPES[shape][0])),
        rewrite_runner=StubRewrite(Mock()),
    )
    anonymizer.run(config=AnonymizerConfig(rewrite=Rewrite()), data=AnonymizerInput(source=str(source)))

    row = next(r for r in seen if r[COL_TEXT] == TEXT)
    assert _spans(row[COL_FINAL_ENTITIES]) == KEPT[shape]
    # The replacement step of the rewrite generation, fed what a model would ask for: every pair it was shown.
    pairs = _requested_pairs(row[COL_ENTITIES_BY_VALUE])
    prepared = _prepare_rewrite_tagged_text(
        {
            **row,
            COL_REPLACEMENT_MAP: _substitute_map(pd.DataFrame({COL_ENTITIES_BY_VALUE: [row[COL_ENTITIES_BY_VALUE]]}))[
                0
            ],
            COL_REWRITE_DISPOSITION_BLOCK: [
                {"entity_value": v, "entity_label": lab, "protection_method_suggestion": "replace"} for v, lab in pairs
            ],
            COL_TAG_NOTATION: "bracket",
        }
    )
    assert prepared[COL_REWRITE_REPLACEMENT_READY] is True
    assert not [v for v in PROTECTED if v in prepared[COL_REWRITE_BASELINE_TEXT]]


class _StringDetector(_Detector):
    """A detector that returns its entity payloads as JSON strings, with a key of its own on every entity."""

    def frame(self, dataframe: pd.DataFrame) -> EntityDetectionResult:
        result = super().frame(dataframe)
        frame = result.dataframe
        tagged = [{"entities": [{**e, "note": "own"} for e in self.by_text[str(t)]]} for t in frame[COL_TEXT]]
        frame[COL_FINAL_ENTITIES] = [json.dumps(e) for e in tagged]
        frame[COL_ENTITIES_BY_VALUE] = [json.dumps(_build_entities_by_value(e)) for e in tagged]
        return result


def _string_payload_anonymizer(tmp_path: Path, **kwargs: Any) -> tuple[Anonymizer, AnonymizerInput]:
    source = tmp_path / "in.csv"
    pd.DataFrame({"text": [TEXT, CONTROL]}).to_csv(source, index=False)  # the collapsed row comes first
    detector = _StringDetector(SHAPES["nested_inner_first"][0])
    return Anonymizer(detection_workflow=_as_builtin(detector), **kwargs), AnonymizerInput(source=str(source))


def _assert_own_encoding(cells: list[object], kept: list[list[tuple[int, int, str]]]) -> None:
    assert all(isinstance(cell, str) for cell in cells)
    assert [_spans(cell) for cell in cells] == kept
    assert all(e["note"] == "own" and "score" not in e for cell in cells for e in json.loads(str(cell))["entities"])


def test_collapsed_rows_keep_the_detectors_encoding_in_replace(tmp_path: Path) -> None:
    anonymizer, data = _string_payload_anonymizer(tmp_path)
    result = anonymizer.run(config=AnonymizerConfig(replace=Redact()), data=data)

    assert list(result.dataframe["text_replaced"]) == [FULL, "[REDACTED_FULL_NAME] lives in [REDACTED_CITY]."]
    control = [(0, 9, "full_name"), (19, 23, "city")]
    _assert_own_encoding(list(result.dataframe["final_entities"]), [KEPT["nested_inner_first"], control])


def test_collapsed_rows_keep_the_detectors_encoding_in_rewrite(tmp_path: Path) -> None:
    seen: list[pd.DataFrame] = []

    class StubRewrite(RewriteWorkflow):
        def _produce(self, rows: Any, **options: Any) -> Any:
            def produce() -> None:
                seen.append(rows.frame.copy())
                rows.frame[COL_REWRITTEN_TEXT] = ["rewritten"] * len(rows.frame)
                for col in (COL_UTILITY_SCORE, COL_LEAKAGE_MASS, COL_WEIGHTED_LEAKAGE_RATE):
                    rows.frame[col] = 1.0 if col == COL_UTILITY_SCORE else 0.0
                rows.frame[COL_ANY_HIGH_LEAKED] = False

            return produce

    anonymizer, data = _string_payload_anonymizer(tmp_path, rewrite_runner=StubRewrite(Mock()))
    anonymizer.run(config=AnonymizerConfig(rewrite=Rewrite()), data=data)

    frame = seen[0]
    control = [(0, 9, "full_name"), (19, 23, "city")]
    _assert_own_encoding(list(frame[COL_FINAL_ENTITIES]), [KEPT["nested_inner_first"], control])
    # The rewrite stages' seed keeps every row's entities: no cell is encoded twice.
    seed = select_seed_cols(frame, [COL_FINAL_ENTITIES, COL_ENTITIES_BY_VALUE])
    assert [_spans(cell) for cell in seed[COL_FINAL_ENTITIES]] == [KEPT["nested_inner_first"], control]
    assert [_requested_pairs(cell) for cell in seed[COL_ENTITIES_BY_VALUE]] == [
        {("Alice Smith", "full_name"), ("Paris", "city")},
        {("Bob Jones", "full_name"), ("Rome", "city")},
    ]


def test_a_missing_by_value_cell_is_rebuilt_in_the_rows_own_encoding() -> None:
    nested = {"entities": [_e("Alice", "first_name", 0, 5), NAME, CITY]}
    plain = {"entities": CONTROL_ENTITIES}
    frame = pd.DataFrame(
        {
            COL_FINAL_ENTITIES: [json.dumps(nested), json.dumps(plain)],
            COL_ENTITIES_BY_VALUE: [None, json.dumps(_build_entities_by_value(plain))],
        }
    )
    out = _with_entities(frame, {0: {0}})

    assert all(isinstance(cell, str) for cell in out[COL_ENTITIES_BY_VALUE])
    assert _requested_pairs(out[COL_ENTITIES_BY_VALUE].iloc[0]) == {("Alice Smith", "full_name"), ("Paris", "city")}


def test_a_wrapped_entities_model_is_collapsed_not_emptied() -> None:
    cell = {"entities": EntitiesSchema.from_raw({"entities": [_e("Alice", "first_name", 0, 5), NAME, CITY]})}
    frame = pd.DataFrame({COL_FINAL_ENTITIES: [cell], COL_ENTITIES_BY_VALUE: [_build_entities_by_value(cell)]})
    out = _with_entities(frame, {0: {0}})

    assert _spans(out[COL_FINAL_ENTITIES].iloc[0]) == KEPT["nested_inner_first"]
    assert _requested_pairs(out[COL_ENTITIES_BY_VALUE].iloc[0]) == {("Alice Smith", "full_name"), ("Paris", "city")}


def test_an_unlabelled_span_fails_its_record_under_substitute(tmp_path: Path) -> None:
    source = tmp_path / "in.csv"
    pd.DataFrame({"text": [TEXT, CONTROL]}).to_csv(source, index=False)

    def run_workflow(self: NddAdapter, df: pd.DataFrame, **_: Any) -> WorkflowRunResult:
        out = df.copy().reset_index(drop=True)
        out[COL_REPLACEMENT_MAP] = _substitute_map(out)
        return WorkflowRunResult(dataframe=out, failed_records=[], source_positions=tuple(range(len(out))))

    anonymizer = Anonymizer(detection_workflow=cast(EntityDetectionWorkflow, _Detector(SHAPES["unlabelled"][0])))
    with patch.object(NddAdapter, "run_workflow", run_workflow):
        result = anonymizer.run(config=AnonymizerConfig(replace=Substitute()), data=AnonymizerInput(source=str(source)))

    assert result.failed_records == [FailedRecord(_rid(source, 0), "detect", "mention_unlabelled")]
    assert not [v for v in PROTECTED for text in result.dataframe["text_replaced"] if v in text]


class _NoByValueDetector(_Detector):
    """A detector that returns ``final_entities`` only: no by-value cell for the first row, or no column at all."""

    def __init__(self, entities: list[dict[str, Any]], *, drop_column: bool) -> None:
        super().__init__(entities)
        self.drop_column = drop_column

    def frame(self, dataframe: pd.DataFrame) -> EntityDetectionResult:
        result = super().frame(dataframe)
        if self.drop_column:
            result.dataframe.drop(columns=[COL_ENTITIES_BY_VALUE], inplace=True)
        else:
            result.dataframe[COL_ENTITIES_BY_VALUE] = [None, *result.dataframe[COL_ENTITIES_BY_VALUE].iloc[1:]]
        return result


@pytest.mark.parametrize("drop_column", [False, True], ids=["cell", "column"])
def test_rows_without_a_by_value_view_are_rewritten_not_passed_through(drop_column: bool, tmp_path: Path) -> None:
    source = tmp_path / "in.csv"
    pd.DataFrame({"text": [TEXT, CONTROL]}).to_csv(source, index=False)
    seen: list[dict[str, Any]] = []

    class StubRewrite(RewriteWorkflow):
        def _produce(self, rows: Any, **options: Any) -> Any:
            def produce() -> None:
                seen.extend(rows.frame.to_dict("records"))
                rows.frame[COL_REWRITTEN_TEXT] = ["rewritten"] * len(rows.frame)
                for col in (COL_UTILITY_SCORE, COL_LEAKAGE_MASS, COL_WEIGHTED_LEAKAGE_RATE):
                    rows.frame[col] = 1.0 if col == COL_UTILITY_SCORE else 0.0
                rows.frame[COL_ANY_HIGH_LEAKED] = False

            return produce

    detector = _NoByValueDetector([NAME, CITY], drop_column=drop_column)
    anonymizer = Anonymizer(
        detection_workflow=cast(EntityDetectionWorkflow, detector), rewrite_runner=StubRewrite(Mock())
    )
    result = anonymizer.run(config=AnonymizerConfig(rewrite=Rewrite()), data=AnonymizerInput(source=str(source)))

    assert result.failed_records == []
    assert list(result.dataframe["text_rewritten"]) == ["rewritten", "rewritten"]
    row = next(r for r in seen if r[COL_TEXT] == TEXT)
    assert _requested_pairs(row[COL_ENTITIES_BY_VALUE]) == {("Alice Smith", "full_name"), ("Paris", "city")}


def test_a_detector_without_a_by_value_column_gets_its_substitute_map(tmp_path: Path) -> None:
    source = tmp_path / "in.csv"
    pd.DataFrame({"text": [TEXT, CONTROL]}).to_csv(source, index=False)

    def run_workflow(self: NddAdapter, df: pd.DataFrame, **_: Any) -> WorkflowRunResult:
        out = df.copy().reset_index(drop=True)
        out[COL_REPLACEMENT_MAP] = _substitute_map(out)
        return WorkflowRunResult(dataframe=out, failed_records=[], source_positions=tuple(range(len(out))))

    detector = _NoByValueDetector([NAME, CITY], drop_column=True)
    anonymizer = Anonymizer(detection_workflow=cast(EntityDetectionWorkflow, detector))
    with patch.object(NddAdapter, "run_workflow", run_workflow):
        result = anonymizer.run(config=AnonymizerConfig(replace=Substitute()), data=AnonymizerInput(source=str(source)))

    assert result.failed_records == []
    assert not [v for v in PROTECTED for text in result.dataframe["text_replaced"] if v in text]


# A first-row by-value cell that does not list every kept pair: rebuilt from the row's entities.
STALE_CELLS = {
    "empty_string": "",
    "empty_dict": {},
    "empty_list": {"entities_by_value": []},
    "empty_list_str": json.dumps({"entities_by_value": []}),
    "stale": {"entities_by_value": [{"value": "Paris", "labels": ["city"]}]},
    "stale_str": json.dumps({"entities_by_value": [{"value": "Paris", "labels": ["city"]}]}),
}
KEPT_PAIRS = {("Alice Smith", "full_name"), ("Paris", "city")}


class _StaleByValueDetector(_Detector):
    def __init__(self, cell: object) -> None:
        super().__init__([NAME, CITY])
        self.cell = cell

    def frame(self, dataframe: pd.DataFrame) -> EntityDetectionResult:
        result = super().frame(dataframe)
        result.dataframe[COL_ENTITIES_BY_VALUE] = [self.cell, *result.dataframe[COL_ENTITIES_BY_VALUE].iloc[1:]]
        return result


def _capturing(runner: RewriteWorkflow, frames: list[pd.DataFrame]) -> RewriteWorkflow:
    """``runner`` with its producing step replaced by one that records the rows it is given and fakes a rewrite."""

    def _produce(rows: Any, **options: Any) -> Any:
        def produce() -> None:
            frames.append(rows.frame.copy())
            rows.frame[COL_REWRITTEN_TEXT] = ["rewritten"] * len(rows.frame)
            for col in (COL_UTILITY_SCORE, COL_LEAKAGE_MASS, COL_WEIGHTED_LEAKAGE_RATE):
                rows.frame[col] = 1.0 if col == COL_UTILITY_SCORE else 0.0
            rows.frame[COL_ANY_HIGH_LEAKED] = False

        return produce

    setattr(runner, "_produce", _produce)
    return runner


@pytest.mark.parametrize("mode", ["rewrite", "combined", "substitute"])
@pytest.mark.parametrize("cell", list(STALE_CELLS))
def test_an_empty_or_stale_by_value_cell_is_rebuilt(cell: str, mode: str, tmp_path: Path) -> None:
    source = tmp_path / "in.csv"
    pd.DataFrame({"text": [TEXT, CONTROL]}).to_csv(source, index=False)
    frames: list[pd.DataFrame] = []

    def run_workflow(self: NddAdapter, df: pd.DataFrame, **_: Any) -> WorkflowRunResult:
        out = df.copy().reset_index(drop=True)
        frames.append(out.copy())
        out[COL_REPLACEMENT_MAP] = _substitute_map(out)
        return WorkflowRunResult(dataframe=out, failed_records=[], source_positions=tuple(range(len(out))))

    anonymizer = Anonymizer(
        detection_workflow=cast(EntityDetectionWorkflow, _StaleByValueDetector(STALE_CELLS[cell])),
        rewrite_runner=_capturing(RewriteWorkflow(Mock()), frames),
        combined_rewrite_runner=cast(CombinedRewriteWorkflow, _capturing(CombinedRewriteWorkflow(Mock()), frames)),
    )
    config = {
        "rewrite": AnonymizerConfig(rewrite=Rewrite()),
        "combined": AnonymizerConfig(rewrite=Rewrite(use_combined_graph=True)),
        "substitute": AnonymizerConfig(replace=Substitute()),
    }[mode]
    with patch.object(NddAdapter, "run_workflow", run_workflow):
        result = anonymizer.run(config=config, data=AnonymizerInput(source=str(source)))

    assert result.failed_records == []
    assert KEPT_PAIRS in [_requested_pairs(c) for c in frames[0][COL_ENTITIES_BY_VALUE]]
    if mode == "substitute":
        assert not [v for v in PROTECTED for text in result.dataframe["text_replaced"] if v in text]
    else:
        assert list(result.dataframe["text_rewritten"]) == ["rewritten", "rewritten"]


@pytest.mark.parametrize("bad", ["label_null", "offset_text"])
def test_a_malformed_span_payload_fails_its_record_at_detect_in_rewrite(bad: str, tmp_path: Path) -> None:
    source = tmp_path / "in.csv"
    pd.DataFrame({"text": [TEXT, CONTROL]}).to_csv(source, index=False)
    broken = {**CITY, "label": None} if bad == "label_null" else {**CITY, "start_position": "x"}
    seen: list[dict[str, Any]] = []

    class StubRewrite(RewriteWorkflow):
        def _produce(self, rows: Any, **options: Any) -> Any:
            def produce() -> None:
                seen.extend(rows.frame.to_dict("records"))
                rows.frame[COL_REWRITTEN_TEXT] = ["rewritten"] * len(rows.frame)
                for col in (COL_UTILITY_SCORE, COL_LEAKAGE_MASS, COL_WEIGHTED_LEAKAGE_RATE):
                    rows.frame[col] = 1.0 if col == COL_UTILITY_SCORE else 0.0
                rows.frame[COL_ANY_HIGH_LEAKED] = False

            return produce

    anonymizer = Anonymizer(
        detection_workflow=_as_builtin(_Detector([NAME, broken])), rewrite_runner=StubRewrite(Mock())
    )
    result = anonymizer.run(config=AnonymizerConfig(rewrite=Rewrite()), data=AnonymizerInput(source=str(source)))

    assert result.failed_records == [FailedRecord(_rid(source, 0), "detect", "invalid_output")]
    assert list(result.dataframe["text_rewritten"]) == ["rewritten"]


class _MixedByValueDetector(_Detector):
    """Dict entity cells, JSON-string by-value cells, and one row's by-value cell left out."""

    def __init__(self, missing: int) -> None:
        super().__init__([NAME, CITY])
        self.missing = missing

    def frame(self, dataframe: pd.DataFrame) -> EntityDetectionResult:
        result = super().frame(dataframe)
        cells: list[object] = [json.dumps(cell) for cell in result.dataframe[COL_ENTITIES_BY_VALUE]]
        cells[self.missing] = None
        result.dataframe[COL_ENTITIES_BY_VALUE] = pd.Series(cells, index=result.dataframe.index, dtype=object)
        return result


@pytest.mark.parametrize("mode", ["rewrite", "combined"])
@pytest.mark.parametrize("missing", [0, 1], ids=["missing_first", "missing_second"])
def test_a_rebuilt_by_value_cell_reaches_the_seed_in_either_order(missing: int, mode: str, tmp_path: Path) -> None:
    source = tmp_path / "in.csv"
    pd.DataFrame({"text": [TEXT, CONTROL]}).to_csv(source, index=False)
    frames: list[pd.DataFrame] = []
    anonymizer = Anonymizer(
        detection_workflow=cast(EntityDetectionWorkflow, _MixedByValueDetector(missing)),
        rewrite_runner=_capturing(RewriteWorkflow(Mock()), frames),
        combined_rewrite_runner=cast(CombinedRewriteWorkflow, _capturing(CombinedRewriteWorkflow(Mock()), frames)),
    )
    result = anonymizer.run(
        config=AnonymizerConfig(rewrite=Rewrite(use_combined_graph=mode == "combined")),
        data=AnonymizerInput(source=str(source)),
    )

    assert result.failed_records == []
    # The rewrite stages' seed keeps every row's by-value view: no cell is encoded twice.
    seed = select_seed_cols(frames[0], [COL_ENTITIES_BY_VALUE])
    assert [_requested_pairs(cell) for cell in seed[COL_ENTITIES_BY_VALUE]] == [
        KEPT_PAIRS,
        {("Bob Jones", "full_name"), ("Rome", "city")},
    ]


THIRD = "Carol White moved to Oslo."
THIRD_ENTITIES = [_e("Carol White", "full_name", 0, 11), _e("Oslo", "city", 21, 25)]
PAIRS_BY_TEXT = {
    TEXT: KEPT_PAIRS,
    CONTROL: {("Bob Jones", "full_name"), ("Rome", "city")},
    THIRD: {("Carol White", "full_name"), ("Oslo", "city")},
}
CROSSING = SHAPES["partial_overlap"][0]
NULL_LABEL = [NAME, {**CITY, "label": None}]

# case -> (first row's entities, entities cell encodings, by-value cell encodings), rows [TEXT, CONTROL, THIRD]
MIXED: dict[str, tuple[list[dict[str, Any]], list[str], list[str]]] = {
    "rejected_first": (CROSSING, ["dict"] * 3, ["dict", "none", "str"]),
    "malformed_first": (NULL_LABEL, ["dict"] * 3, ["dict", "none", "str"]),
    "detector_mixed": ([NAME, CITY], ["dict"] * 3, ["dict", "str", "str"]),
    "no_by_value_mixed_entities": ([NAME, CITY], ["dict", "str", "str"], ["none"] * 3),
    "mixed_entities_only": ([NAME, CITY], ["dict", "str", "str"], ["str"] * 3),
}


def _encoded(payload: dict[str, Any], kind: str) -> object:
    return {"dict": payload, "str": json.dumps(payload), "none": None}[kind]


class _MixedEncodingDetector(_Detector):
    def __init__(self, case: str) -> None:
        first, self.entity_kinds, self.by_value_kinds = MIXED[case]
        super().__init__(first)
        self.by_text[THIRD] = THIRD_ENTITIES

    def frame(self, dataframe: pd.DataFrame) -> EntityDetectionResult:
        frame = dataframe.copy()
        entities = [{"entities": self.by_text[str(text)]} for text in frame[COL_TEXT]]
        frame[COL_FINAL_ENTITIES] = pd.Series(
            [_encoded(e, k) for e, k in zip(entities, self.entity_kinds)], index=frame.index, dtype=object
        )
        frame[COL_ENTITIES_BY_VALUE] = pd.Series(
            [_encoded(_build_entities_by_value(e), k) for e, k in zip(entities, self.by_value_kinds)],
            index=frame.index,
            dtype=object,
        )
        return EntityDetectionResult(frame, [], tuple(range(len(frame))))


@pytest.mark.parametrize("mode", ["rewrite", "combined"])
@pytest.mark.parametrize("case", list(MIXED))
def test_every_kept_row_reaches_the_rewrite_seed_whatever_the_encoding_mix(
    case: str, mode: str, tmp_path: Path
) -> None:
    source = tmp_path / "in.csv"
    pd.DataFrame({"text": [TEXT, CONTROL, THIRD]}).to_csv(source, index=False)
    frames: list[pd.DataFrame] = []
    anonymizer = Anonymizer(
        detection_workflow=cast(EntityDetectionWorkflow, _MixedEncodingDetector(case)),
        rewrite_runner=_capturing(RewriteWorkflow(Mock()), frames),
        combined_rewrite_runner=cast(CombinedRewriteWorkflow, _capturing(CombinedRewriteWorkflow(Mock()), frames)),
    )
    result = anonymizer.run(
        config=AnonymizerConfig(rewrite=Rewrite(use_combined_graph=mode == "combined")),
        data=AnonymizerInput(source=str(source)),
    )

    assert all(f.step == "detect" for f in result.failed_records)
    kept = list(frames[0][COL_TEXT])
    assert kept == ([CONTROL, THIRD] if case in ("rejected_first", "malformed_first") else [TEXT, CONTROL, THIRD])
    seed = select_seed_cols(frames[0], [COL_FINAL_ENTITIES, COL_ENTITIES_BY_VALUE])
    assert [_requested_pairs(cell) for cell in seed[COL_ENTITIES_BY_VALUE]] == [PAIRS_BY_TEXT[t] for t in kept]
    assert [{(v, label) for _, _, label, v in _spans_with_values(cell)} for cell in seed[COL_FINAL_ENTITIES]] == [
        PAIRS_BY_TEXT[t] for t in kept
    ]


def _spans_with_values(raw: object) -> list[tuple[int, int, str, str]]:
    return [(e.start_position, e.end_position, e.label or "", e.value) for e in EntitiesSchema.from_raw(raw).entities]


MALFORMED_REASON = "malformed entities: payload failed schema validation"
# shape -> entities per text, or how the detector leaves its frame; no shape returns a by-value column
NO_BY_VALUE_SHAPES: dict[str, Any] = {
    "no_spans": {TEXT: [], CONTROL: []},
    "all_malformed": {TEXT: NULL_LABEL, CONTROL: [{**CONTROL_ENTITIES[1], "start_position": "x"}]},
    "no_spans_plus_malformed": {TEXT: NULL_LABEL, CONTROL: []},
    "no_entities_column": "no_entities_column",
    "every_row_failed": "every_row_failed",
}


class _ShapeWithoutByValueDetector(_FrameDetector):
    def __init__(self, shape: str) -> None:
        self.shape = NO_BY_VALUE_SHAPES[shape]

    def frame(self, dataframe: pd.DataFrame) -> EntityDetectionResult:
        frame = dataframe.copy()
        if self.shape == "every_row_failed":
            failed = [FailedRecord(str(i), "entity-detection", "detector_error") for i in range(len(frame))]
            return EntityDetectionResult(frame.iloc[0:0], failed, ())
        if self.shape != "no_entities_column":
            frame[COL_FINAL_ENTITIES] = pd.Series(
                [{"entities": self.shape[str(t)]} for t in frame[COL_TEXT]], index=frame.index, dtype=object
            )
        return EntityDetectionResult(frame, [], tuple(range(len(frame))))


def _expected_without_by_value(shape: str, mode: str, ids: list[str]) -> tuple[list[str], list[FailedRecord]]:
    """The output texts and failed records: unparsable rows fail at ``replace`` in Replace, at ``detect`` in Rewrite."""
    malformed = {"all_malformed": ["0", "1"], "no_spans_plus_malformed": ["0"]}.get(shape, [])
    if shape == "every_row_failed":
        return [], [FailedRecord(i, "entity-detection", "detector_error") for i in ("0", "1")]
    replace = mode in ("redact", "substitute")
    failed = [
        FailedRecord(ids[int(i)], "replace", MALFORMED_REASON)
        if replace
        else FailedRecord(ids[int(i)], "detect", "invalid_output")
        for i in malformed
    ]
    return [t for i, t in enumerate([TEXT, CONTROL]) if str(i) not in malformed], failed


@pytest.mark.parametrize("mode", ["redact", "substitute", "rewrite", "combined"])
@pytest.mark.parametrize("shape", list(NO_BY_VALUE_SHAPES))
def test_a_detector_without_a_by_value_column_completes_whatever_its_rows(
    shape: str, mode: str, tmp_path: Path
) -> None:
    source = tmp_path / "in.csv"
    pd.DataFrame({"text": [TEXT, CONTROL]}).to_csv(source, index=False)
    frames: list[pd.DataFrame] = []

    def run_workflow(self: NddAdapter, df: pd.DataFrame, **_: Any) -> WorkflowRunResult:
        out = df.copy().reset_index(drop=True)
        out[COL_REPLACEMENT_MAP] = _substitute_map(out)
        return WorkflowRunResult(dataframe=out, failed_records=[], source_positions=tuple(range(len(out))))

    anonymizer = Anonymizer(
        detection_workflow=_as_builtin(_ShapeWithoutByValueDetector(shape)),
        rewrite_runner=_capturing(RewriteWorkflow(Mock()), frames),
        combined_rewrite_runner=cast(CombinedRewriteWorkflow, _capturing(CombinedRewriteWorkflow(Mock()), frames)),
    )
    config = {
        "redact": AnonymizerConfig(replace=Redact()),
        "substitute": AnonymizerConfig(replace=Substitute()),
        "rewrite": AnonymizerConfig(rewrite=Rewrite()),
        "combined": AnonymizerConfig(rewrite=Rewrite(use_combined_graph=True)),
    }[mode]
    with patch.object(NddAdapter, "run_workflow", run_workflow):
        result = anonymizer.run(config=config, data=AnonymizerInput(source=str(source)))

    texts, failed = _expected_without_by_value(shape, mode, [_rid(source, 0), _rid(source, 1)])
    assert result.failed_records == failed
    column = "text_replaced" if mode in ("redact", "substitute") else "text_rewritten"
    assert (list(result.dataframe[column]) if column in result.dataframe.columns else []) == texts


@pytest.mark.parametrize("extra", ["occurs", "absent"])
@pytest.mark.parametrize("shape", ["good", "nested"])
def test_a_by_value_view_keeps_the_extra_pairs_that_occur_in_the_text(shape: str, extra: str, tmp_path: Path) -> None:
    """A row with spans gets the view of its kept spans plus every detector pair whose value occurs in its text. A
    pair whose value does not occur is dropped, so the rewrite's deterministic replacements stay ready."""
    entities = [NAME, CITY] if shape == "good" else SHAPES["nested_inner_first"][0]
    pair = ("lives", "other") if extra == "occurs" else ("Zurich", "city")
    view = _build_entities_by_value({"entities": [NAME, CITY]})
    view = {**view, "entities_by_value": [*view["entities_by_value"], {"value": pair[0], "labels": [pair[1]]}]}

    class ExtraPairDetector(_Detector):
        def frame(self, dataframe: pd.DataFrame) -> EntityDetectionResult:
            result = super().frame(dataframe)
            result.dataframe[COL_ENTITIES_BY_VALUE] = [view, *result.dataframe[COL_ENTITIES_BY_VALUE].iloc[1:]]
            return result

    source = tmp_path / "in.csv"
    pd.DataFrame({"text": [TEXT, CONTROL]}).to_csv(source, index=False)
    frames: list[pd.DataFrame] = []
    anonymizer = Anonymizer(
        detection_workflow=cast(EntityDetectionWorkflow, ExtraPairDetector(entities)),
        rewrite_runner=_capturing(RewriteWorkflow(Mock()), frames),
    )
    anonymizer.run(config=AnonymizerConfig(rewrite=Rewrite()), data=AnonymizerInput(source=str(source)))

    row = frames[0].to_dict("records")[0]
    assert _requested_pairs(row[COL_ENTITIES_BY_VALUE]) == KEPT_PAIRS | ({pair} if extra == "occurs" else set())
    if extra == "absent":
        pairs = _requested_pairs(row[COL_ENTITIES_BY_VALUE])
        prepared = _prepare_rewrite_tagged_text(
            {
                **row,
                COL_REPLACEMENT_MAP: _substitute_map(
                    pd.DataFrame({COL_ENTITIES_BY_VALUE: [row[COL_ENTITIES_BY_VALUE]]})
                )[0],
                COL_REWRITE_DISPOSITION_BLOCK: [
                    {"entity_value": v, "entity_label": lab, "protection_method_suggestion": "replace"}
                    for v, lab in pairs
                ],
                COL_TAG_NOTATION: "bracket",
            }
        )
        assert prepared[COL_REWRITE_REPLACEMENT_READY] is True
        assert not [v for v in PROTECTED if v in prepared[COL_REWRITE_BASELINE_TEXT]]


@pytest.mark.parametrize("mode", ["redact", "substitute", "rewrite"])
def test_the_trace_of_a_rebuilt_by_value_cell_writes_to_parquet(mode: str, tmp_path: Path) -> None:
    source = tmp_path / "in.csv"
    pd.DataFrame({"text": [TEXT, CONTROL]}).to_csv(source, index=False)

    def run_workflow(self: NddAdapter, df: pd.DataFrame, **_: Any) -> WorkflowRunResult:
        out = df.copy().reset_index(drop=True)
        out[COL_REPLACEMENT_MAP] = _substitute_map(out)
        return WorkflowRunResult(dataframe=out, failed_records=[], source_positions=tuple(range(len(out))))

    anonymizer = Anonymizer(
        detection_workflow=_as_builtin(_MixedByValueDetector(0)),
        rewrite_runner=_capturing(RewriteWorkflow(Mock()), []),
    )
    config = {
        "redact": AnonymizerConfig(replace=Redact()),
        "substitute": AnonymizerConfig(replace=Substitute()),
        "rewrite": AnonymizerConfig(rewrite=Rewrite()),
    }[mode]
    with patch.object(NddAdapter, "run_workflow", run_workflow):
        result = anonymizer.run(config=config, data=AnonymizerInput(source=str(source)))

    result.trace_dataframe.to_parquet(tmp_path / "trace.parquet")
    assert all(isinstance(cell, str) for cell in result.trace_dataframe[COL_ENTITIES_BY_VALUE])


@pytest.mark.parametrize("mode", ["rewrite", "combined"])
def test_a_missing_by_value_cell_on_a_malformed_row_fails_only_that_row(mode: str, tmp_path: Path) -> None:
    class NaByValueDetector(_Detector):
        def frame(self, dataframe: pd.DataFrame) -> EntityDetectionResult:
            result = super().frame(dataframe)
            result.dataframe[COL_ENTITIES_BY_VALUE] = pd.Series(
                [pd.NA, *result.dataframe[COL_ENTITIES_BY_VALUE].iloc[1:]], index=result.dataframe.index, dtype=object
            )
            return result

    source = tmp_path / "in.csv"
    pd.DataFrame({"text": [TEXT, CONTROL]}).to_csv(source, index=False)
    frames: list[pd.DataFrame] = []
    anonymizer = Anonymizer(
        detection_workflow=_as_builtin(NaByValueDetector(NULL_LABEL)),
        rewrite_runner=_capturing(RewriteWorkflow(Mock()), frames),
        combined_rewrite_runner=cast(CombinedRewriteWorkflow, _capturing(CombinedRewriteWorkflow(Mock()), frames)),
    )
    result = anonymizer.run(
        config=AnonymizerConfig(rewrite=Rewrite(use_combined_graph=mode == "combined")),
        data=AnonymizerInput(source=str(source)),
    )

    assert result.failed_records == [FailedRecord(_rid(source, 0), "detect", "invalid_output")]
    assert list(result.dataframe["text_rewritten"]) == ["rewritten"]


def test_a_spans_row_and_a_spans_free_row_in_one_batch_both_keep_their_reported_values(tmp_path: Path) -> None:
    """A row with spans and a pair outside them gets its spans' view plus that pair; a row with no spans keeps the
    detector's view. Every reported value that occurs in a row's text is in that row's view."""
    with_extra = _view(*FULL_PAIRS, ("lives", "other"))
    control_view = _build_entities_by_value({"entities": CONTROL_ENTITIES})

    class BothShapesDetector(_Detector):
        def frame(self, dataframe: pd.DataFrame) -> EntityDetectionResult:
            frame = dataframe.copy()
            frame[COL_FINAL_ENTITIES] = [{"entities": [NAME, CITY]}, {"entities": []}]
            frame[COL_ENTITIES_BY_VALUE] = [with_extra, control_view]
            return EntityDetectionResult(frame, [], tuple(range(len(frame))))

    source = tmp_path / "in.csv"
    pd.DataFrame({"text": [TEXT, CONTROL]}).to_csv(source, index=False)
    frames: list[pd.DataFrame] = []
    anonymizer = Anonymizer(
        detection_workflow=cast(EntityDetectionWorkflow, BothShapesDetector([])),
        rewrite_runner=_capturing(RewriteWorkflow(Mock()), frames),
    )
    anonymizer.run(config=AnonymizerConfig(rewrite=Rewrite()), data=AnonymizerInput(source=str(source)))

    rows = {row[COL_TEXT]: row for row in frames[0].to_dict("records")}
    assert _requested_pairs(rows[TEXT][COL_ENTITIES_BY_VALUE]) == KEPT_PAIRS | {("lives", "other")}
    assert _requested_pairs(rows[CONTROL][COL_ENTITIES_BY_VALUE]) == PAIRS_BY_TEXT[CONTROL]


FULL_PAIRS = (("Alice Smith", "full_name"), ("Paris", "city"))


def _view(*pairs: tuple[str, str]) -> dict[str, Any]:
    return {"entities_by_value": [{"value": v, "labels": [lab]} for v, lab in pairs]}


# shape -> (spans, or None for no final_entities column; the detector's by-value view) for TEXT
UNSPANNED: dict[str, tuple[list[dict[str, Any]] | None, dict[str, Any]]] = {
    "span_missing_a_listed_value": ([NAME], _view(*FULL_PAIRS)),
    "extra_pair_outside_the_spans": ([NAME, CITY], _view(*FULL_PAIRS, ("lives", "other"))),
    "spans_free_with_pairs": ([], _view(*FULL_PAIRS)),
    "no_entities_column": (None, _view(*FULL_PAIRS)),
}


class _UnspannedDetector(_Detector):
    def __init__(self, shape: str) -> None:
        super().__init__([])
        self.spans, self.view = UNSPANNED[shape]

    def frame(self, dataframe: pd.DataFrame) -> EntityDetectionResult:
        frame = dataframe.copy()
        texts = [str(t) for t in frame[COL_TEXT]]
        if self.spans is not None:
            frame[COL_FINAL_ENTITIES] = [{"entities": self.spans if t == TEXT else CONTROL_ENTITIES} for t in texts]
        frame[COL_ENTITIES_BY_VALUE] = [
            self.view if t == TEXT else _build_entities_by_value({"entities": CONTROL_ENTITIES}) for t in texts
        ]
        return EntityDetectionResult(frame, [], tuple(range(len(frame))))


@pytest.mark.parametrize("strategy", ["redact", "hash", "substitute"])
@pytest.mark.parametrize("shape", list(UNSPANNED))
def test_replace_fails_a_row_whose_reported_value_has_no_span(shape: str, strategy: str, tmp_path: Path) -> None:
    source = tmp_path / "in.csv"
    pd.DataFrame({"text": [TEXT, CONTROL]}).to_csv(source, index=False)

    def run_workflow(self: NddAdapter, df: pd.DataFrame, **_: Any) -> WorkflowRunResult:
        out = df.copy().reset_index(drop=True)
        out[COL_REPLACEMENT_MAP] = _substitute_map(out)
        return WorkflowRunResult(dataframe=out, failed_records=[], source_positions=tuple(range(len(out))))

    replace = {"redact": Redact(), "hash": Hash(key="s3cr3t"), "substitute": Substitute()}[strategy]
    anonymizer = Anonymizer(detection_workflow=cast(EntityDetectionWorkflow, _UnspannedDetector(shape)))
    with patch.object(NddAdapter, "run_workflow", run_workflow):
        result = anonymizer.run(config=AnonymizerConfig(replace=replace), data=AnonymizerInput(source=str(source)))

    control_kept = shape != "no_entities_column"
    assert result.failed_records == [FailedRecord(_rid(source, 0), "detect", "unspanned_value")] + (
        [] if control_kept else [FailedRecord(_rid(source, 1), "detect", "unspanned_value")]
    )
    assert not [v for v in PROTECTED for text in result.dataframe["text_replaced"] if v in text]


def _one_row_run(
    tmp_path: Path,
    text: str,
    spans: list[dict[str, Any]] | None,
    view: object,
    mode: str,
    *,
    custom: bool = True,
    replace_runner: Any = None,
) -> tuple[Any, list[pd.DataFrame]]:
    """``text`` through ``Anonymizer.run`` with a detector that returns ``spans`` (``None``: no entities column) and
    the by-value cell ``view``; Rewrite captures the rows its generation step is given."""
    source = tmp_path / "in.csv"
    pd.DataFrame({"text": [text]}).to_csv(source, index=False)

    class Detector(_FrameDetector):
        def frame(self, dataframe: pd.DataFrame) -> EntityDetectionResult:
            frame = dataframe.copy()
            if spans is not None:
                frame[COL_FINAL_ENTITIES] = [{"entities": spans}]
            frame[COL_ENTITIES_BY_VALUE] = pd.Series([view], index=frame.index, dtype=object)
            return EntityDetectionResult(frame, [], (0,))

        def run(self, dataframe: pd.DataFrame, **kwargs: object) -> Any:
            # The built-in leg returns its frame, as the built-in workflow does.
            return super().run(dataframe) if custom else self.frame(dataframe)

    def run_workflow(self: NddAdapter, df: pd.DataFrame, **_: Any) -> WorkflowRunResult:
        out = df.copy().reset_index(drop=True)
        out[COL_REPLACEMENT_MAP] = _substitute_map(out)
        return WorkflowRunResult(dataframe=out, failed_records=[], source_positions=tuple(range(len(out))))

    frames: list[pd.DataFrame] = []
    anonymizer = Anonymizer(
        detection_workflow=cast(EntityDetectionWorkflow, Detector()),
        rewrite_runner=_capturing(RewriteWorkflow(Mock()), frames),
        combined_rewrite_runner=cast(CombinedRewriteWorkflow, _capturing(CombinedRewriteWorkflow(Mock()), frames)),
        **({"replace_runner": replace_runner} if replace_runner is not None else {}),
    )
    if not custom:  # the built-in detection path, with this detector standing in for the built-in workflow
        anonymizer._detector_kind = "builtin"
    config = {
        "redact": AnonymizerConfig(replace=Redact()),
        "hash": AnonymizerConfig(replace=Hash(key="s3cr3t")),
        "annotate": AnonymizerConfig(replace=Annotate()),
        "substitute": AnonymizerConfig(replace=Substitute()),
        "rewrite": AnonymizerConfig(rewrite=Rewrite()),
        "combined": AnonymizerConfig(rewrite=Rewrite(use_combined_graph=True)),
    }[mode]
    with patch.object(NddAdapter, "run_workflow", run_workflow):
        return anonymizer.run(config=config, data=AnonymizerInput(source=str(source))), frames


def _at(text: str, value: str, label: str, nth: int = 0) -> dict[str, Any]:
    start = -1
    for _ in range(nth + 1):
        start = text.index(value, start + 1)
    return _e(value, label, start, start + len(value))


def _seed_values(frames: list[pd.DataFrame]) -> set[str]:
    return {v for v, _ in _requested_pairs(frames[0][COL_ENTITIES_BY_VALUE].iloc[0])}


@pytest.mark.parametrize(
    ("text", "span", "variant"),
    [
        ("Qa Paris met ROME.", ("Paris", "city"), ("rome", "city")),
        ("Qd José met Paris.", ("Paris", "city"), ("Jose", "first_name")),
    ],
    ids=["case", "accent"],
)
def test_a_case_or_accent_variant_pair_stays_in_the_rewrite_view(
    text: str, span: tuple[str, str], variant: tuple[str, str], tmp_path: Path
) -> None:
    """A listed pair occurs under the leftover check's matching rule (case and accents folded), not only byte for
    byte; it stays in the view the rewrite is given."""
    result, frames = _one_row_run(tmp_path, text, [_at(text, *span)], _view(span, variant), "rewrite")

    assert result.failed_records == []
    assert _seed_values(frames) == {span[0], variant[0]}


@pytest.mark.parametrize("view", ["complete", "absent", "stale"])
@pytest.mark.parametrize("strategy", ["redact", "substitute"])
def test_replace_fails_a_spans_unspanned_repeat_whatever_the_detectors_view(
    view: str, strategy: str, tmp_path: Path
) -> None:
    """The check reads the row's view after derivation, so a span value repeated outside every span fails the row
    whether the detector listed it, gave no view or gave a stale one."""
    text = "Qb Bob met Bob."
    cell = {"complete": _view(("Bob", "first_name")), "absent": None, "stale": _view(("Zurich", "city"))}[view]
    result, _ = _one_row_run(tmp_path, text, [_at(text, "Bob", "first_name")], cell, strategy)

    assert result.failed_records == [FailedRecord(_rid(tmp_path / "in.csv", 0), "detect", "unspanned_value")]


@pytest.mark.parametrize("strategy", ["redact", "hash", "annotate", "substitute"])
@pytest.mark.parametrize(
    ("text", "spans"),
    [
        ("Ann filed the Annual report.", [("Ann", "first_name")]),
        ("Eve is 34 and paid $340.", [("Eve", "first_name"), ("34", "age")]),
    ],
    ids=["embedded_word", "embedded_number"],
)
def test_a_value_inside_a_longer_word_is_not_an_unspanned_occurrence(
    text: str, spans: list[tuple[str, str]], strategy: str, tmp_path: Path
) -> None:
    entities = [_at(text, *s) for s in spans]
    result, _ = _one_row_run(tmp_path, text, entities, _build_entities_by_value({"entities": entities}), strategy)

    assert result.failed_records == []


@pytest.mark.parametrize("custom", [True, False], ids=["custom", "builtin"])
def test_the_replace_check_scope(custom: bool, tmp_path: Path) -> None:
    """A custom detector's missed repeat fails the row; the built-in detector's gets a span and is replaced."""
    text = "Bob met Bob's sister."
    result, _ = _one_row_run(tmp_path, text, [_at(text, "Bob", "first_name")], None, "redact", custom=custom)

    if custom:
        assert result.failed_records == [FailedRecord(_rid(tmp_path / "in.csv", 0), "detect", "unspanned_value")]
    else:
        assert result.failed_records == []
        assert list(result.dataframe["text_replaced"]) == ["[REDACTED_FIRST_NAME] met [REDACTED_FIRST_NAME]'s sister."]
        spans = result.dataframe["final_entities"].iloc[0]["entities"]
        assert [(e["start_position"], e["source"]) for e in spans] == [(0, "detector"), (8, "repeat")]


# Tokens that collide: a value inside a longer word, repeats, a case and an accent variant, adjacent values.
_TOKENS = [
    "Ann",
    "Annual",
    "Bob",
    "Bobsled",
    "34",
    "$340",
    "Paris",
    "ROME",
    "José",
    "Paris/Rome",
    "met",
    "in",
    "the",
    "王",
    "李明",
    "Alice Smith",
    "Alice",
]
_LABELS = {
    "Ann": "first_name",
    "Bob": "first_name",
    "34": "age",
    "Paris": "city",
    "ROME": "city",
    "José": "first_name",
    "王": "last_name",
    "李明": "full_name",
    "Alice Smith": "full_name",
    "Alice": "first_name",
}
_EXTRA_PAIRS = [("rome", "city"), ("Jose", "first_name"), ("Lee", "first_name"), ("Rome", "city"), ("Paris", "city")]


def _random_row(rng: random.Random) -> tuple[str, list[dict[str, Any]] | None, object]:
    text = "Q " + " ".join(rng.choice(_TOKENS) for _ in range(rng.randint(2, 7))) + "."
    spans: list[dict[str, Any]] = []
    for value, label in _LABELS.items():
        start = -1
        while (start := text.find(value, start + 1)) >= 0:
            whole = (start == 0 or not text[start - 1].isalnum()) and not text[start + len(value)].isalnum()
            if whole and rng.random() < 0.6:
                spans.append(_e(value, label, start, start + len(value)))
    spans.sort(key=lambda e: e["start_position"])
    pairs = {(e["value"], e["label"]) for e in spans if rng.random() < 0.8}
    pairs |= set(rng.sample(_EXTRA_PAIRS, rng.randint(0, 2)))
    view: object = _view(*sorted(pairs)) if rng.random() < 0.8 else None
    if view is not None and rng.random() < 0.5:
        view = json.dumps(view)
    return text, (None if not spans and rng.random() < 0.3 else spans), view


@pytest.mark.parametrize("mode", ["redact", "hash", "annotate", "substitute", "rewrite", "combined"])
def test_every_reported_value_is_protected_or_fails_its_row(mode: str, tmp_path: Path) -> None:
    """The by-value invariant over random rows: every value the detector reports (span or by-value pair) that occurs
    in the row's text, under the leftover check's matching rule, lies inside a kept span or is in the Rewrite view, or
    the row fails at ``detect``. A released Replace row has every such value inside a span."""
    rng = random.Random(f"432-{mode}")
    for _ in range(40):
        text, spans, view = _random_row(rng)
        result, frames = _one_row_run(tmp_path, text, spans, view, mode)
        listed = [v for v, _ in _requested_pairs(view)] if view is not None else []
        reported = [e["value"] for e in spans or []] + listed
        returned = EntitiesSchema.from_raw({"entities": spans or []})
        dropped = covered_spans(GraphScope().datum(text), returned)  # nested spans the detect step collapses
        kept = EntitiesSchema(entities=[e for i, e in enumerate(returned.entities) if i not in dropped])
        outside = {value for value in reported if _occurs_outside(text, kept, value)}
        if result.failed_records:
            assert [f.step for f in result.failed_records] == ["detect"], (text, spans, view)
        elif mode in ("rewrite", "combined"):
            assert outside <= (_seed_values(frames) if frames else set()), (text, spans, view)
        else:
            assert not outside, (text, spans, view)


def _occurs_outside(text: str, kept: EntitiesSchema, value: str) -> bool:
    """The test's oracle: the leftover check's rule, or, for a value it builds no matcher for, any plain occurrence
    that no kept span contains."""
    leftovers = count_values(text, entity_spans(text, kept.entities), "", [value])
    if leftovers:
        return any(lo.undetected_spans for lo in leftovers)
    spans = [(e.start_position, e.end_position) for e in kept.entities]
    start = text.find(value)
    while value and start >= 0:
        if not any(a <= start and start + len(value) <= b for a, b in spans):
            return True
        start = text.find(value, start + 1)
    return False


@pytest.mark.parametrize("mode", ["redact", "rewrite", "combined"])
def test_a_one_character_unspaced_value_is_kept_or_fails_the_row(mode: str, tmp_path: Path) -> None:
    """The leftover check's rule builds no matcher for a one-character value in an unspaced script (王); such a value
    fails closed: it stays in the Rewrite view, and Replace fails the row."""
    text = "Qj 王 met 李明."
    view = _view(("李明", "full_name"), ("王", "last_name"))
    result, frames = _one_row_run(tmp_path, text, [_at(text, "李明", "full_name")], view, mode)

    if mode == "redact":
        assert result.failed_records == [FailedRecord(_rid(tmp_path / "in.csv", 0), "detect", "unspanned_value")]
    else:
        assert result.failed_records == []
        assert _seed_values(frames) == {"李明", "王"}


class _OldRunner(ReplacementWorkflow):
    """An override written for the earlier signature: no ``detector`` keyword and no ``**kwargs``."""

    def run_with_detection(  # ty: ignore[invalid-method-override]
        self,
        dataframe: pd.DataFrame,
        *,
        detect: Any,
        replace_method: Any,
        model_configs: Any,
        selected_models: Any,
        preview_num_records: int | None = None,
    ) -> Any:
        raise AssertionError("not reached: the facade rejects the override before detection")


@pytest.mark.parametrize("custom", [False, True], ids=["builtin", "custom"])
def test_an_old_signature_run_with_detection_override_fails_early_naming_the_keyword(
    custom: bool, tmp_path: Path
) -> None:
    """With a custom detector the facade passed ``unspanned_check=True`` and the override raised ``TypeError``
    (#319 review 4644413727); the built-in leg ran. Both now fail before detection with an error that names
    ``detector`` and how to update the override."""
    text = "Bob met Ann."
    spans = [_at(text, "Bob", "first_name"), _at(text, "Ann", "first_name")]
    with pytest.raises(InvalidConfigError, match=r"_OldRunner\.run_with_detection does not accept the detector"):
        _one_row_run(
            tmp_path, text, spans, None, "redact", custom=custom, replace_runner=_OldRunner(llm_workflow=Mock())
        )


def test_an_override_that_takes_the_detector_keyword_receives_the_detector_kind(tmp_path: Path) -> None:
    kinds: list[str] = []

    class NewRunner(ReplacementWorkflow):
        def run_with_detection(self, dataframe: pd.DataFrame, **kwargs: Any) -> Any:
            kinds.append(kwargs["detector"])
            return super().run_with_detection(dataframe, **kwargs)

    text = "Bob met Ann."
    spans = [_at(text, "Bob", "first_name"), _at(text, "Ann", "first_name")]
    for custom in (False, True):
        _one_row_run(
            tmp_path, text, spans, None, "redact", custom=custom, replace_runner=NewRunner(llm_workflow=Mock())
        )
    assert kinds == ["builtin", "custom"]


def test_a_rewrite_runner_override_receives_no_detector_keyword(tmp_path: Path) -> None:
    """Rewrite names failures from the carried record ids, so its runner gets no detector keyword."""
    seen: list[set[str]] = []

    class Runner(RewriteWorkflow):
        def run_with_detection(self, dataframe: pd.DataFrame, **kwargs: Any) -> Any:
            seen.append(set(kwargs))
            return super().run_with_detection(dataframe, **kwargs)

    text = "Bob met Ann."
    frames: list[pd.DataFrame] = []
    source = tmp_path / "in.csv"
    pd.DataFrame({"text": [text]}).to_csv(source, index=False)
    anonymizer = Anonymizer(
        detection_workflow=builtin_stub(
            lambda df, **_: EntityDetectionResult(
                df.assign(**{COL_FINAL_ENTITIES: [{"entities": []}] * len(df)}), [], tuple(range(len(df)))
            )
        ),
        rewrite_runner=_capturing(Runner(Mock()), frames),
    )
    anonymizer.run(config=AnonymizerConfig(rewrite=Rewrite()), data=AnonymizerInput(source=str(source)))
    assert seen and "detector" not in seen[0]


@pytest.mark.parametrize("view", ["absent", "outer_only"])
@pytest.mark.parametrize("mode", ["redact", "hash", "annotate", "substitute", "rewrite", "combined"])
def test_a_collapsed_spans_value_outside_the_kept_spans_is_reported(view: str, mode: str, tmp_path: Path) -> None:
    """A nested span the detect step collapses still reports its value: "Alice" also stands alone, outside the kept
    "Alice Smith", so Replace fails the row and the Rewrite view holds it."""
    text = "Alice Smith met Alice."
    spans = [_at(text, "Alice Smith", "full_name"), _at(text, "Alice", "first_name")]
    cell = None if view == "absent" else _view(("Alice Smith", "full_name"))
    result, frames = _one_row_run(tmp_path, text, spans, cell, mode)

    if mode in ("rewrite", "combined"):
        assert result.failed_records == []
        assert _seed_values(frames) == {"Alice Smith", "Alice"}
    else:
        assert result.failed_records == [FailedRecord(_rid(tmp_path / "in.csv", 0), "detect", "unspanned_value")]


@pytest.mark.parametrize("mode", ["redact", "rewrite"])
def test_a_collapsed_spans_value_inside_a_kept_span_is_not_added(mode: str, tmp_path: Path) -> None:
    """ "Smith" occurs only inside the kept "Alice Smith": it is protected by that span and adds nothing."""
    text = "Alice Smith met Bob."
    spans = [_at(text, "Alice Smith", "full_name"), _at(text, "Smith", "last_name"), _at(text, "Bob", "first_name")]
    result, frames = _one_row_run(tmp_path, text, spans, None, mode)

    assert result.failed_records == []
    if mode == "rewrite":
        assert _seed_values(frames) == {"Alice Smith", "Bob"}


def test_an_unspanned_value_is_named_in_the_failure_log(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    text = "Qb Bob met Bob."
    with caplog.at_level(logging.DEBUG, logger="anonymizer"):
        result, _ = _one_row_run(tmp_path, text, [_at(text, "Bob", "first_name")], None, "redact")

    assert result.failed_records == [FailedRecord(_rid(tmp_path / "in.csv", 0), "detect", "unspanned_value")]
    assert "1 record(s) failed during pipeline processing: detect/unspanned_value=1." in caplog.text
