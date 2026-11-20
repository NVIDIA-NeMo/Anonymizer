# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Semantic facts a detector's sources declare, merged under one rule and checked before detection runs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from anonymizer.config.anonymizer_config import Detect, resolve_effective_detection_labels
from anonymizer.config.entity_labels import normalize_entity_label
from anonymizer.config.regex import BuiltinRegex, RegexRule
from anonymizer.engine.detection.regex_detection import resolve_regex_constrained_labels
from anonymizer.interface.errors import InvalidConfigError

OffsetUnit = Literal["code_point", "utf16_code_unit", "utf8_byte"]
DetectorFact = Literal["offset_unit", "labels_fixed", "labels"]

_FACTS: tuple[DetectorFact, ...] = ("offset_unit", "labels_fixed", "labels")

# The built-in path parses wire offsets and sends labels, so it needs both facts declared.
BUILTIN_REQUIRED_FACTS: frozenset[DetectorFact] = frozenset({"offset_unit", "labels_fixed"})


@dataclass(frozen=True, slots=True)
class DetectorCapabilities:
    """Semantic facts one source declares about a detector. None means this source does not declare the fact."""

    source: Literal["discovery", "configuration"]
    offset_unit: OffsetUnit | None = None
    labels_fixed: bool | None = None  # True: fixed vocabulary; False: takes caller-selected labels
    labels: frozenset[str] | None = None  # normalized; required when labels_fixed is True

    def __post_init__(self) -> None:
        if self.labels is not None:
            object.__setattr__(self, "labels", frozenset(normalize_entity_label(label) for label in self.labels))


GLINER_CONTRACT = DetectorCapabilities(source="configuration", offset_unit="code_point", labels_fixed=False)


def merge_capabilities(sources: Sequence[DetectorCapabilities]) -> DetectorCapabilities:
    """Merge sources fact by fact; every source declaring a fact must declare the same value."""
    if not sources:
        return DetectorCapabilities(source="configuration")
    merged: dict[str, Any] = {}
    conflicts: list[str] = []
    for fact in _FACTS:
        declaring = [source for source in sources if getattr(source, fact) is not None]
        values = {getattr(source, fact) for source in declaring}
        if len(values) > 1:
            names = ", ".join(sorted({source.source for source in declaring}))
            conflicts.append(f"{fact} (declared differently by: {names})")
        elif values:
            merged[fact] = values.pop()
    if conflicts:
        raise InvalidConfigError("Detector declarations conflict on " + "; ".join(conflicts) + ".")
    return DetectorCapabilities(source=sources[0].source, **merged)


def check_detector_work(
    caps: DetectorCapabilities, *, detector_labels: Sequence[str], required: frozenset[DetectorFact]
) -> None:
    """Reject detector work the declared facts do not cover; a skipped detector needs nothing."""
    if not detector_labels:
        return
    missing = sorted(fact for fact in required if getattr(caps, fact) is None)
    if missing:
        raise InvalidConfigError(f"The detector does not declare: {', '.join(missing)}.")
    if caps.labels_fixed:
        if caps.labels is None:
            raise InvalidConfigError("The detector declares a fixed label set but not its labels.")
        unsupported = {normalize_entity_label(label) for label in detector_labels} - caps.labels
        if unsupported:
            raise InvalidConfigError(
                f"The detector's fixed label set lacks {len(unsupported)} of the labels detection would send it."
            )


def detector_labels_from(
    labels: Iterable[str], *, builtin_regexes: bool, rules: list[BuiltinRegex | RegexRule]
) -> list[str]:
    """Return ``labels`` minus regex-constrained labels, in order: the labels sent to the detector."""
    labels = list(labels)
    constrained = {
        normalize_entity_label(label)
        for label in resolve_regex_constrained_labels(labels=labels, builtin_regexes=builtin_regexes, rules=rules)
    }
    return [label for label in labels if normalize_entity_label(label) not in constrained]


def resolve_detector_labels(detect: Detect) -> list[str]:
    """Return the labels the detector is sent for ``detect``."""
    labels = resolve_effective_detection_labels(
        detect.entity_labels,
        regex_rules=detect.regex_rules,
        excluded_entity_labels=detect.excluded_entity_labels,
    )
    return detector_labels_from(labels, builtin_regexes=detect.builtin_regexes, rules=detect.regex_rules)
