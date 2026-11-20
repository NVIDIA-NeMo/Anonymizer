# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detector facts: one merge rule and one preflight check against the labels the detector is sent."""

from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import pytest
from data_designer.config.models import ModelConfig

from anonymizer.config.anonymizer_config import AnonymizerConfig, Detect, TextRecord, TextRecordsInput
from anonymizer.config.models import DetectionModelSelection
from anonymizer.config.regex import BuiltinRegex, RegexRule
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import COL_FINAL_ENTITIES, COL_RAW_DETECTED, COL_TEXT
from anonymizer.engine.detection.capabilities import (
    BUILTIN_REQUIRED_FACTS,
    GLINER_CONTRACT,
    DetectorCapabilities,
    check_detector_work,
    merge_capabilities,
    resolve_detector_labels,
)
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult, EntityDetectionWorkflow
from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.interface.errors import InvalidConfigError

_DATA = TextRecordsInput(records=[TextRecord(id="r1", text="Alice")])


def _custom_anonymizer() -> tuple[Anonymizer, Mock]:
    detector = Mock(spec=EntityDetectionWorkflow)
    frame = pd.DataFrame({COL_TEXT: ["Alice"], COL_FINAL_ENTITIES: [{"entities": []}]})
    detector.run.return_value = EntityDetectionResult(dataframe=frame, failed_records=[])
    return Anonymizer(detection_workflow=detector), detector


def _fixed(*labels: str) -> DetectorCapabilities:
    return DetectorCapabilities(source="configuration", labels_fixed=True, labels=frozenset(labels))


def test_fixed_label_detector_lacking_a_sent_label_rejects_before_detection() -> None:
    anonymizer, detector = _custom_anonymizer()
    anonymizer._declare_detector(_fixed("first_name"))
    config = AnonymizerConfig(detect=Detect(entity_labels=["first_name", "email"]), replace=Redact())
    with pytest.raises(InvalidConfigError):
        anonymizer.run(config=config, data=_DATA)
    with pytest.raises(InvalidConfigError):
        anonymizer.validate_config(config)
    detector.run.assert_not_called()


@pytest.mark.parametrize(("additional", "accepted"), [(False, True), (True, False)])
def test_regex_constrained_label_outside_fixed_set_does_not_count(additional: bool, accepted: bool) -> None:
    anonymizer, _ = _custom_anonymizer()
    anonymizer._declare_detector(_fixed("first_name"))
    rule = RegexRule(label="ticket", pattern=r"TKT-\d+", detect_additional_matches=additional)
    config = AnonymizerConfig(
        detect=Detect(entity_labels=["first_name", "ticket"], regex_rules=[rule]), replace=Redact()
    )
    if accepted:
        anonymizer.validate_config(config)
    else:
        with pytest.raises(InvalidConfigError):
            anonymizer.validate_config(config)


def test_every_label_regex_constrained_with_empty_fixed_set_is_accepted(
    stub_detector_model_configs: list[ModelConfig],
    stub_detection_model_selection: DetectionModelSelection,
) -> None:
    anonymizer, detector = _custom_anonymizer()
    anonymizer._declare_detector(_fixed())
    rule = RegexRule(label="ticket", pattern=r"TKT-\d+", detect_additional_matches=False)
    detect = Detect(entity_labels=["ticket"], regex_rules=[rule])
    anonymizer.run(config=AnonymizerConfig(detect=detect, replace=Redact()), data=_DATA)
    detector.run.assert_called_once()

    _, columns = EntityDetectionWorkflow(adapter=Mock())._build_detection_spec(
        model_configs=stub_detector_model_configs,
        selected_models=stub_detection_model_selection,
        gliner_detection_threshold=0.3,
        entity_labels=detect.entity_labels,
        regex_rules=detect.regex_rules,
    )
    raw = next(column for column in columns if column.name == COL_RAW_DETECTED)
    assert resolve_detector_labels(detect) == []
    assert raw.skip is not None


@pytest.mark.parametrize(
    "other",
    [
        DetectorCapabilities(source="discovery", offset_unit="utf16_code_unit"),
        DetectorCapabilities(source="discovery", labels_fixed=True, labels=frozenset({"first_name"})),
    ],
)
def test_disagreeing_sources_raise_on_declaration(other: DetectorCapabilities) -> None:
    anonymizer = Anonymizer()
    with pytest.raises(InvalidConfigError, match="discovery"):
        anonymizer._declare_detector(other)


def test_agreeing_or_silent_sources_merge_to_the_declared_values() -> None:
    merged = merge_capabilities(
        [
            GLINER_CONTRACT,
            DetectorCapabilities(source="discovery", offset_unit="code_point"),
            DetectorCapabilities(source="discovery"),
        ]
    )
    assert (merged.offset_unit, merged.labels_fixed, merged.labels) == ("code_point", False, None)
    labels = merge_capabilities([_fixed(" First_Name "), _fixed("first_name")])
    assert labels.labels == frozenset({"first_name"})
    with pytest.raises(InvalidConfigError):
        merge_capabilities([_fixed("first_name"), _fixed("email")])


def test_builtin_kind_requires_declared_offset_unit() -> None:
    undeclared = DetectorCapabilities(source="configuration", labels_fixed=False)
    with pytest.raises(InvalidConfigError, match="offset_unit"):
        check_detector_work(undeclared, detector_labels=["first_name"], required=BUILTIN_REQUIRED_FACTS)
    check_detector_work(undeclared, detector_labels=[], required=BUILTIN_REQUIRED_FACTS)


def test_custom_kind_with_nothing_declared_runs() -> None:
    anonymizer, detector = _custom_anonymizer()
    anonymizer.run(config=AnonymizerConfig(replace=Redact()), data=_DATA)
    detector.run.assert_called_once()


def test_resolve_detector_labels_matches_the_detector_request(
    stub_detector_model_configs: list[ModelConfig],
    stub_detection_model_selection: DetectionModelSelection,
) -> None:
    detect = Detect(
        regex_rules=[
            BuiltinRegex(label="email", detect_additional_matches=False),
            RegexRule(label="ticket", pattern=r"TKT-\d+", detect_additional_matches=False),
            RegexRule(label="badge", pattern=r"B-\d+"),
        ],
        excluded_entity_labels=["city"],
    )
    models, _ = EntityDetectionWorkflow(adapter=Mock())._build_detection_spec(
        model_configs=stub_detector_model_configs,
        selected_models=stub_detection_model_selection,
        gliner_detection_threshold=0.3,
        entity_labels=detect.entity_labels,
        excluded_entity_labels=detect.excluded_entity_labels,
        builtin_regexes=detect.builtin_regexes,
        regex_rules=detect.regex_rules,
    )
    detector = next(model for model in models if model.alias == stub_detection_model_selection.entity_detector)
    extra_body = detector.inference_parameters.extra_body
    assert extra_body is not None
    sent = extra_body["labels"]
    assert resolve_detector_labels(detect) == sent
    assert "badge" in sent
    assert not {"email", "ticket", "city"} & set(sent)
