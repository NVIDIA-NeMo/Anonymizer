# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Facade-level pins on detection: where dropped rows are reported, empty input, unresolved positions, and how a
caller's detector is driven. Detection's stages are due to become graph operations; these pin today's behaviour."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest

from anonymizer import Anonymizer
from anonymizer.config.anonymizer_config import (
    AnonymizerConfig,
    AnonymizerInput,
    Rewrite,
    TextRecord,
    TextRecordsInput,
)
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_ENTITIES_BY_VALUE,
    COL_FINAL_ENTITIES,
    COL_LEAKAGE_MASS,
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
from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow
from tests.parity.harness import RECORDS, Case, scripted

_DATA = TextRecordsInput(records=[TextRecord(id=i, text=t) for i, t in RECORDS])
_REPLACE = AnonymizerConfig(replace=Redact())
_REWRITE = AnonymizerConfig(rewrite=Rewrite())
_MODES = {"replace": _REPLACE, "rewrite": _REWRITE}


def _drive(anonymizer: Anonymizer, call: str, config: AnonymizerConfig, data: Any) -> Any:
    if call == "preview":
        return anonymizer.preview(config=config, data=data, num_records=len(RECORDS))
    return anonymizer.run(config=config, data=data)


def _steps(result: Any) -> list[tuple[str, str]]:
    return [(f.record_id, f.step) for f in result.failed_records]


class _StubRewrite(RewriteWorkflow):
    """Rewrite with the model stages stubbed: each row is rewritten to ``R<text>``."""

    def _produce(self, rows, **options):  # noqa: ANN001, ANN003, ANN202 - LLM stages stubbed
        def produce() -> None:
            rows.frame[COL_REWRITTEN_TEXT] = [f"R<{t}>" for t in rows.frame[COL_TEXT]]
            for col in (COL_UTILITY_SCORE, COL_LEAKAGE_MASS, COL_WEIGHTED_LEAKAGE_RATE):
                rows.frame[col] = 1.0 if col == COL_UTILITY_SCORE else 0.0
            rows.frame[COL_ANY_HIGH_LEAKED] = False

        return produce


def _first_word_entities(text: str) -> dict:
    word = text.split()[0]
    return {
        "entities": [{"id": "a", "value": word, "label": "first_name", "start_position": 0, "end_position": len(word)}]
    }


def _detected(frame: pd.DataFrame, positions: tuple[int, ...] | None = None) -> EntityDetectionResult:
    """Each row's first word as its one entity; ``positions=None`` reports none (the default ``()``)."""
    out = frame.copy()
    entities = [_first_word_entities(str(t)) for t in out[COL_TEXT]]
    out[COL_FINAL_ENTITIES] = entities
    out[COL_ENTITIES_BY_VALUE] = [_build_entities_by_value(e) for e in entities]
    return EntityDetectionResult(out, [], positions if positions is not None else ())


# --- 1 and 2: rows dropped inside detection are reported at the dropping workflow, never at ``detect`` -----------


@pytest.mark.parametrize("call", ["run", "preview"])
def test_replace_row_dropped_by_entity_detection_fails_at_entity_detection(call: str) -> None:
    case = Case("pin", lambda: _REPLACE, drops={"entity-detection": "Bob"})
    with scripted(case):
        result = _drive(Anonymizer(), call, _REPLACE, _DATA)

    assert _steps(result) == [("c", "entity-detection")]
    assert "Bob lives in Paris." not in list(result.dataframe["text"])


@pytest.mark.parametrize("call", ["run", "preview"])
def test_rewrite_rows_dropped_inside_detection_fail_at_their_workflow(call: str) -> None:
    case = Case("pin", lambda: _REWRITE, drops={"entity-detection": "sky", "latent-entity-detection": "Dora"})
    with scripted(case):
        result = _drive(Anonymizer(rewrite_runner=_StubRewrite(Mock())), call, _REWRITE, _DATA)

    assert sorted(_steps(result)) == [("b", "entity-detection"), ("d", "latent-entity-detection")]
    assert not {"sky", "Dora"} & {w for t in result.dataframe["text"] for w in t.split()}


# --- 3: empty input still calls the built-in detector ---------------------------------------------------------------


@pytest.mark.parametrize("mode", list(_MODES))
def test_empty_input_still_runs_the_built_in_detector(tmp_path: Path, mode: str) -> None:
    """The custom-detector form is ``test_run_calls_the_detector_on_empty_input``; this pins the built-in one."""
    csv_path = tmp_path / "input.csv"
    pd.DataFrame({"text": []}).to_csv(csv_path, index=False)
    with scripted(Case("pin", lambda: _MODES[mode])) as (script, _):
        result = Anonymizer().run(config=_MODES[mode], data=AnonymizerInput(source=str(csv_path)))

    assert result.dataframe.empty
    assert result.failed_records == []
    expected = [("entity-detection", 0), *([("latent-entity-detection", 0)] if mode == "rewrite" else [])]
    assert [(run["workflow"], run["seed_rows"]) for run in script.log if "detection" in run["workflow"]] == expected


# --- 4: a detection result with unresolved positions reroutes over the frame as returned ----------------------------


@pytest.mark.parametrize("mode", list(_MODES))
def test_unresolved_detection_positions_reroute_over_the_returned_frame(mode: str) -> None:
    """No reported positions and no record IDs: positions cannot be resolved, so the route replaces or rewrites the
    detection frame as returned (here reversed and missing its first row) instead of failing every row.

    This pins today's silent row loss: row ``a`` disappears with no failed record. Decision D4 keeps that until N1c,
    so N1c updates this test on purpose."""
    calls: list[int] = []

    class Unresolved:
        def run(self, dataframe: pd.DataFrame, **_: object) -> EntityDetectionResult:
            calls.append(len(dataframe))
            returned = dataframe.drop(columns=[c for c in dataframe.columns if c != COL_TEXT]).iloc[:0:-1]
            return _detected(returned.reset_index(drop=True))

    detector: Any = Unresolved()
    anonymizer = Anonymizer(detection_workflow=detector, rewrite_runner=_StubRewrite(Mock()))
    with scripted(Case("pin", lambda: _MODES[mode])):
        result = anonymizer.run(config=_MODES[mode], data=_DATA)

    returned = [t for _, t in RECORDS][:0:-1]
    column, expected = (
        ("text_replaced", [f"[REDACTED_FIRST_NAME] {t.split(' ', 1)[1]}" for t in returned])
        if mode == "replace"
        else ("text_rewritten", [f"R<{t}>" for t in returned])
    )
    assert calls == [len(RECORDS)]
    assert list(result.dataframe["text"]) == returned
    assert list(result.dataframe[column]) == expected
    assert result.failed_records == []


# --- 5: a Mock(spec=EntityDetectionWorkflow) and a run-overriding subclass are driven through their run ---------------


class _OverridingDetector(EntityDetectionWorkflow):
    def __init__(self) -> None:
        super().__init__(adapter=Mock())
        self.calls: list[int] = []

    def run(self, dataframe: pd.DataFrame, **_: object) -> EntityDetectionResult:
        self.calls.append(len(dataframe))
        return _detected(dataframe, tuple(range(len(dataframe))))


def _spec_mock() -> Mock:
    detector = Mock(spec=EntityDetectionWorkflow)
    detector.run.side_effect = lambda frame, **_: _detected(frame, tuple(range(len(frame))))
    return detector


@pytest.mark.parametrize("make", [_spec_mock, _OverridingDetector], ids=["spec-mock", "subclass"])
@pytest.mark.parametrize("mode", list(_MODES))
@pytest.mark.parametrize("call", ["run", "preview"])
def test_caller_detector_run_is_called_once_and_its_result_used(make: Any, mode: str, call: str) -> None:
    detector = make()
    anonymizer = Anonymizer(detection_workflow=detector, rewrite_runner=_StubRewrite(Mock()))
    with scripted(Case("pin", lambda: _MODES[mode])) as (script, _):
        result = _drive(anonymizer, call, _MODES[mode], _DATA)

    calls = detector.run.call_count if isinstance(detector, Mock) else len(detector.calls)
    assert calls == 1
    assert not [run for run in script.log if "detection" in run["workflow"]]
    texts = [t for _, t in RECORDS]
    expected = (
        [f"[REDACTED_FIRST_NAME] {t.split(' ', 1)[1]}" for t in texts]
        if mode == "replace"
        else [f"R<{t}>" for t in texts]
    )
    column = "text_replaced" if mode == "replace" else "text_rewritten"
    assert list(result.dataframe[column]) == expected
    assert result.failed_records == []
