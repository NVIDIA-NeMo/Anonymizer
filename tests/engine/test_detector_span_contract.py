# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The span contract for a custom ``detection_workflow``, shape by shape, through the public API.

Identical spans collapse into one and a span contained in another gives way to the outer one; every other bad span
fails its record at ``detect`` with a named reason. Either way no protected value reaches the output.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, Rewrite
from anonymizer.config.replace_strategies import Hash, Redact, Substitute
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
from anonymizer.engine.detection.detect_step import _with_entities, accepted_mentions
from anonymizer.engine.detection.detection_workflow import (
    EntityDetectionResult,
    EntityDetectionWorkflow,
    _build_entities_by_value,
)
from anonymizer.engine.graph_core.authoring import GraphScope
from anonymizer.engine.ndd.adapter import FailedRecord, NddAdapter, WorkflowRunResult
from anonymizer.engine.rewrite.rewrite_generation import _prepare_rewrite_tagged_text
from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow
from anonymizer.engine.rewrite.workflow_utils import select_seed_cols
from anonymizer.engine.schemas import EntitiesSchema
from anonymizer.interface.anonymizer import Anonymizer

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


class _Detector:
    def __init__(self, entities: list[dict[str, Any]]) -> None:
        self.by_text = {TEXT: entities, CONTROL: CONTROL_ENTITIES}

    def run(self, dataframe: pd.DataFrame, **_: object) -> EntityDetectionResult:
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
        assert result.failed_records == [FailedRecord("0", "detect", expected)]
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

    method = {"redact": Redact(), "hash": Hash(), "substitute": Substitute()}[strategy]
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

    def run(self, dataframe: pd.DataFrame, **_: object) -> EntityDetectionResult:
        result = super().run(dataframe)
        frame = result.dataframe
        tagged = [{"entities": [{**e, "note": "own"} for e in self.by_text[str(t)]]} for t in frame[COL_TEXT]]
        frame[COL_FINAL_ENTITIES] = [json.dumps(e) for e in tagged]
        frame[COL_ENTITIES_BY_VALUE] = [json.dumps(_build_entities_by_value(e)) for e in tagged]
        return result


def _string_payload_anonymizer(tmp_path: Path, **kwargs: Any) -> tuple[Anonymizer, AnonymizerInput]:
    source = tmp_path / "in.csv"
    pd.DataFrame({"text": [TEXT, CONTROL]}).to_csv(source, index=False)  # the collapsed row comes first
    detector = _StringDetector(SHAPES["nested_inner_first"][0])
    return Anonymizer(detection_workflow=cast(EntityDetectionWorkflow, detector), **kwargs), AnonymizerInput(
        source=str(source)
    )


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

    assert result.failed_records == [FailedRecord("0", "detect", "mention_unlabelled")]
    assert not [v for v in PROTECTED for text in result.dataframe["text_replaced"] if v in text]


class _NoByValueDetector(_Detector):
    """A detector that returns ``final_entities`` only: no by-value cell for the first row, or no column at all."""

    def __init__(self, entities: list[dict[str, Any]], *, drop_column: bool) -> None:
        super().__init__(entities)
        self.drop_column = drop_column

    def run(self, dataframe: pd.DataFrame, **_: object) -> EntityDetectionResult:
        result = super().run(dataframe)
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
