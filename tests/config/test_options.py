# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The read-only option views: a frozen config satisfies them, and raw fields are not on them."""

from __future__ import annotations

from typing import Any

from anonymizer.config.anonymizer_config import Detect, DetectorDeclaration, Rewrite
from anonymizer.config.options import DetectionOptions, RewriteOptions
from anonymizer.config.rewrite import default_privacy_goal


def _read(options: DetectionOptions) -> tuple[Any, ...]:
    return (
        options.scope,
        options.strict_labels,
        options.validator_examples,
        options.augmenter_examples,
        options.validation_max_entities_per_call,
        options.validation_excerpt_window_chars,
    )


def test_detect_satisfies_the_detection_options_view() -> None:
    detect = Detect(
        entity_labels=["first_name", "vendor_key"],
        entity_label_examples={"vendor_key": ["acme_live_1"]},
        validation_max_entities_per_call=7,
        validation_excerpt_window_chars=11,
    )
    scope, strict, validator_examples, augmenter_examples, max_entities, window = _read(detect)
    assert scope.effective_labels == ("first_name", "vendor_key")
    assert strict is True
    assert validator_examples["vendor_key"] == ("acme_live_1",)
    assert augmenter_examples == {"vendor_key": ("acme_live_1",)}
    assert (max_entities, window) == (7, 11)


def _raw_labels(options: DetectionOptions) -> object:
    return options.entity_labels  # ty: ignore[unresolved-attribute]


def _raw_detector(options: DetectionOptions) -> object:
    return options.detector  # ty: ignore[unresolved-attribute]


def test_raw_field_read_through_the_view_is_a_type_error() -> None:
    detect = Detect(entity_labels=["first_name"], detector=DetectorDeclaration(offset_unit="code_point"))
    assert _raw_labels(detect) == ["first_name"]
    assert _raw_detector(detect) == DetectorDeclaration(offset_unit="code_point")


def _goal(options: RewriteOptions) -> str:
    return options.privacy_goal.protect


def _raw_risk(options: RewriteOptions) -> object:
    return options.risk_tolerance  # ty: ignore[unresolved-attribute]


def test_rewrite_satisfies_the_rewrite_options_view() -> None:
    rewrite = Rewrite()
    assert _goal(rewrite) == default_privacy_goal().protect
    assert _raw_risk(rewrite) == rewrite.risk_tolerance
