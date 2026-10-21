# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Caller-supplied detectors with v0.4.0's exact ``run`` signature, through every public route.

``EntityDetectionWorkflow.run()`` gained ``entity_label_examples`` (#275) and the graph-only
``source_frame`` keyword after v0.4.0. A detector written against v0.4.0's
signature (no such parameters, no ``**kwargs``) must keep running, with those keywords withheld and
one counts-only warning when configured examples are dropped. Expectations are main with #311 applied.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest
from data_designer.config.models import ModelConfig
from data_designer.interface.data_designer import DataDesigner

from anonymizer import Annotate, AnonymizerConfig, AnonymizerInput, Hash, Redact, Rewrite, Substitute
from anonymizer.config.models import DetectionModelSelection
from anonymizer.engine.constants import COL_REPLACED_TEXT, COL_TEXT
from anonymizer.engine.detection._graph_profile import run_detection_redact
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult, EntityDetectionWorkflow
from anonymizer.engine.detection.run_arguments import POST_040_RUN_KEYWORDS, detection_run_arguments
from anonymizer.engine.ndd.adapter import NddAdapter
from tests.interface.test_substitute_route_compat import (
    _PLAIN,
    _TEXTS,
    _anonymizer,
    _custom_frame,
    scripted,  # noqa: F401  (pytest fixture)
)

pytestmark = pytest.mark.compat

_EXAMPLES = {"first_name": ["Zed"]}
_WARNING = (
    "entity_label_examples configured for 1 label(s) but withheld: the configured detection_workflow "
    "does not accept this keyword argument"
)
# main + #311: a 0.4.0-signature detector's entities are transformed like any caller detector's.
_EXPECTED = {
    "Redact": ["[REDACTED_FIRST_NAME] waved.", "[REDACTED_FIRST_NAME] stayed."],
    "Annotate": ["<Alice, first_name> waved.", "<Bob, first_name> stayed."],
    "Hash": ["<HASH_FIRST_NAME_3bc51062973c> waved.", "<HASH_FIRST_NAME_cd9fb1e148cc> stayed."],
    "Substitute": ["Maya waved.", "Theo stayed."],
}


class _Release040Detector:
    """Duck-types v0.4.0's EntityDetectionWorkflow.run() signature exactly."""

    def __init__(self) -> None:
        self.calls = 0

    def run(
        self,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: DetectionModelSelection,
        gliner_detection_threshold: float,
        validation_max_entities_per_call: int = 20,
        validation_excerpt_window_chars: int = 200,
        validation_single_chunk_full_text: bool = True,
        entity_labels: list[str] | None = None,
        excluded_entity_labels: list[str] | None = None,
        privacy_goal: object | None = None,
        data_summary: str | None = None,
        tag_latent_entities: bool = True,
        compute_grouped_entities: bool | None = None,
        preview_num_records: int | None = None,
    ) -> EntityDetectionResult:
        self.calls += 1
        return EntityDetectionResult(_custom_frame(dataframe, {"preview_num_records": preview_num_records}), [])


class _Release040Subclass(EntityDetectionWorkflow):
    """An EntityDetectionWorkflow subclass overriding run() with v0.4.0's exact signature."""

    def __init__(self, adapter: Any, *, positioned: bool = False) -> None:
        super().__init__(adapter)
        self.calls = 0
        self.positioned = positioned

    def run(  # ty: ignore[invalid-method-override]
        self,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: DetectionModelSelection,
        gliner_detection_threshold: float,
        validation_max_entities_per_call: int = 20,
        validation_excerpt_window_chars: int = 200,
        validation_single_chunk_full_text: bool = True,
        entity_labels: list[str] | None = None,
        excluded_entity_labels: list[str] | None = None,
        privacy_goal: object | None = None,
        data_summary: str | None = None,
        tag_latent_entities: bool = True,
        compute_grouped_entities: bool | None = None,
        preview_num_records: int | None = None,
    ) -> EntityDetectionResult:
        self.calls += 1
        frame = _custom_frame(dataframe, {"preview_num_records": preview_num_records})
        positions = tuple(range(len(frame))) if self.positioned else ()
        return EntityDetectionResult(frame, [], source_positions=positions)


def _detector(kind: str, designer: DataDesigner) -> _Release040Detector | _Release040Subclass:
    return _Release040Detector() if kind == "duck_typed" else _Release040Subclass(NddAdapter(designer))


def _result(
    designer: DataDesigner, detector: Any, mode: str, tmp_path: Path, config: AnonymizerConfig, texts: list[str]
):
    source = tmp_path / "input.csv"
    pd.DataFrame({"text": texts}).to_csv(source, index=False)
    data = AnonymizerInput(source=str(source), text_column="text")
    anonymizer = _anonymizer(designer, detection_workflow=detector)
    return (
        anonymizer.run(config=config, data=data)
        if mode == "run"
        else anonymizer.preview(config=config, data=data, num_records=len(texts))
    )


def _withheld_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "entity_label_examples configured" in r.getMessage()]


@pytest.mark.parametrize("mode", ["run", "preview"])
@pytest.mark.parametrize("kind", ["duck_typed", "subclass"])
@pytest.mark.parametrize("strategy", [Redact, Annotate, Hash, Substitute])
def test_release_040_detector_runs_replace_strategies_with_examples_withheld(
    scripted: tuple[DataDesigner, Counter[str]],  # noqa: F811
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    strategy: type[Redact | Annotate | Hash | Substitute],
    kind: str,
    mode: str,
) -> None:
    designer, calls = scripted
    detector = _detector(kind, designer)
    config = AnonymizerConfig(
        detect={"entity_labels": ["first_name"], "entity_label_examples": _EXAMPLES}, replace=strategy()
    )

    with caplog.at_level(logging.WARNING, logger="anonymizer"):
        result = _result(designer, detector, mode, tmp_path, config, _TEXTS)

    assert not result.failed_records
    assert result.dataframe["text_replaced"].tolist() == _EXPECTED[strategy.__name__]
    assert detector.calls == 1
    assert _withheld_warnings(caplog) == [_WARNING]
    assert "Zed" not in caplog.text
    assert calls == ({"health_check": 1, "map": 2} if strategy is Substitute else {})


@pytest.mark.parametrize("mode", ["run", "preview"])
@pytest.mark.parametrize("kind", ["duck_typed", "subclass"])
def test_release_040_detector_runs_rewrite_with_examples_withheld(
    scripted: tuple[DataDesigner, Counter[str]],  # noqa: F811
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    kind: str,
    mode: str,
) -> None:
    designer, calls = scripted
    detector = _detector(kind, designer)
    config = AnonymizerConfig(
        detect={"entity_labels": ["first_name"], "entity_label_examples": _EXAMPLES}, rewrite=Rewrite()
    )

    with caplog.at_level(logging.WARNING, logger="anonymizer"):
        result = _result(designer, detector, mode, tmp_path, config, _PLAIN)

    assert not result.failed_records
    assert result.dataframe[["text", "text_rewritten"]].to_dict("records") == [
        {"text": t, "text_rewritten": t} for t in _PLAIN
    ]
    assert detector.calls == 1
    assert _withheld_warnings(caplog) == [_WARNING]
    assert calls == {}


@pytest.mark.parametrize("kind", ["duck_typed", "subclass"])
def test_release_040_detector_without_examples_does_not_warn(
    scripted: tuple[DataDesigner, Counter[str]],  # noqa: F811
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    kind: str,
) -> None:
    designer, _ = scripted
    detector = _detector(kind, designer)

    with caplog.at_level(logging.WARNING, logger="anonymizer"):
        result = _result(designer, detector, "run", tmp_path, AnonymizerConfig(replace=Redact()), _TEXTS)

    assert result.dataframe["text_replaced"].tolist() == _EXPECTED["Redact"]
    assert _withheld_warnings(caplog) == []


@pytest.mark.parametrize("strategy", [Redact, Annotate, Hash])
def test_release_040_detector_through_graph_invocation_withholds_source_frame(
    strategy: type[Redact | Annotate | Hash],
) -> None:
    """The graph-invocation consumer adds source_frame; a 0.4.0 run() must not receive it."""
    detector = _Release040Subclass(Mock(), positioned=True)
    selected = DetectionModelSelection(
        entity_detector="d", entity_validator=["v"], entity_augmenter="a", latent_detector="l"
    )

    projected = run_detection_redact(
        pd.DataFrame({COL_TEXT: _TEXTS}),
        detection=detector,
        model_configs=[],
        selected_models=selected,
        detection_options={
            "gliner_detection_threshold": 0.3,
            "entity_labels": ["first_name"],
            "entity_label_examples": _EXAMPLES,
            "tag_latent_entities": False,
            "compute_grouped_entities": True,
        },
        replace_method=strategy(),
    )

    assert detector.calls == 1
    assert projected.dataframe[COL_REPLACED_TEXT].tolist() == _EXPECTED[strategy.__name__]


def test_builtin_run_keeps_every_argument() -> None:
    arguments = {"gliner_detection_threshold": 0.3, **dict.fromkeys(POST_040_RUN_KEYWORDS, object())}

    assert detection_run_arguments(EntityDetectionWorkflow(Mock()).run, arguments) == arguments
    assert detection_run_arguments(lambda frame, **kwargs: None, arguments) == arguments
    assert detection_run_arguments(_Release040Detector().run, arguments) == {"gliner_detection_threshold": 0.3}


def test_non_introspectable_run_fails_open() -> None:
    arguments = dict.fromkeys(POST_040_RUN_KEYWORDS, None)

    class _Opaque:
        @property
        def __signature__(self) -> Any:
            raise ValueError("no signature")

        def __call__(self, *args: Any, **kwargs: Any) -> None:
            return None

    assert detection_run_arguments(_Opaque(), arguments) == arguments
