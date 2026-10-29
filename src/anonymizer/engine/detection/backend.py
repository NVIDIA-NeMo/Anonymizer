# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import SupportsFloat, SupportsIndex, SupportsInt

from anonymizer.engine.schemas import (
    AdmissionRoute,
    DetectionResult,
    Observation,
    SourceView,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _GlinerDetectionBackend:
    """Translate the existing GLiNER response into source-bound observations."""

    source_instance: str
    source_kind: str = "initial_detector"

    def to_detection_result(
        self,
        *,
        detection_result_id: str,
        source_view: SourceView,
        canonical_text: str,
        raw_output: object,
    ) -> DetectionResult:
        """Convert raw output to observations, skipping malformed or mismatched spans.

        Whitespace is trimmed from source bounds before publishing each value.
        GLiNER derives text and offsets from the same span, so a text mismatch
        indicates inconsistent source text or offset mapping at the integration
        boundary. Skip that span, log only its offsets, and retain other spans.
        """
        if source_view.end_position > len(canonical_text):
            raise ValueError(
                f"source view end {source_view.end_position} exceeds canonical text length {len(canonical_text)}"
            )

        view_text = canonical_text[source_view.start_position : source_view.end_position]
        raw_entities = _extract_raw_entities(raw_output)
        candidates: list[tuple[int, int, str, float | None, int]] = []
        for raw_index, raw_entity in enumerate(raw_entities):
            raw_text = raw_entity.get("text")
            raw_label = raw_entity.get("label")
            detector_text = raw_text.strip() if isinstance(raw_text, str) else ""
            label = raw_label.strip() if isinstance(raw_label, str) else ""
            view_relative_start = _coerce_int(raw_entity.get("start"))
            view_relative_end = _coerce_int(raw_entity.get("end"))
            score = _coerce_finite_float(raw_entity.get("score"))
            if not detector_text or not label:
                continue
            if (
                view_relative_start is None
                or view_relative_end is None
                or view_relative_start < 0
                or view_relative_end <= view_relative_start
                or view_relative_end > len(view_text)
            ):
                continue
            canonical_start = source_view.start_position + view_relative_start
            canonical_end = source_view.start_position + view_relative_end
            source_value = canonical_text[canonical_start:canonical_end]
            if detector_text.casefold() != source_value.strip().casefold():
                logger.warning(
                    "Skipping GLiNER entity whose text does not match source offsets [%d, %d)",
                    canonical_start,
                    canonical_end,
                )
                continue
            canonical_start += len(source_value) - len(source_value.lstrip())
            canonical_end -= len(source_value) - len(source_value.rstrip())
            candidates.append((canonical_start, canonical_end, label, score, raw_index))

        sorted_candidates = sorted(
            candidates,
            key=lambda item: (
                item[0],
                item[1],
                item[2].strip().casefold(),
                item[3] is None,
                -(item[3] or 0.0),
                item[4],
            ),
        )
        observations = tuple(
            Observation(
                detection_result_id=detection_result_id,
                observation_id=f"o{observation_index}",
                source_text_ref=source_view.source_text_ref,
                value=canonical_text[start_position:end_position],
                label=label,
                start_position=start_position,
                end_position=end_position,
                source_kind=self.source_kind,
                source_instance=self.source_instance,
                admission_route=AdmissionRoute.llm_validation,
                native_score=score,
            )
            for observation_index, (start_position, end_position, label, score, _) in enumerate(
                sorted_candidates,
                start=1,
            )
        )
        return DetectionResult(
            detection_result_id=detection_result_id,
            source_view=source_view,
            source_kind=self.source_kind,
            source_instance=self.source_instance,
            observations=observations,
        )


def _extract_raw_entities(raw_output: object) -> list[dict[str, object]]:
    if isinstance(raw_output, str):
        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Failed to parse GLiNER JSON output (error=%s, length=%d)",
                exc.msg,
                len(raw_output),
            )
            return []
    elif isinstance(raw_output, dict):
        payload = raw_output
    else:
        return []
    if not isinstance(payload, dict):
        return []
    raw_entities = payload.get("entities", [])
    if not isinstance(raw_entities, list):
        return []
    return [item for item in raw_entities if isinstance(item, dict)]


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if not isinstance(value, (str, bytes, bytearray, SupportsInt, SupportsIndex)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (str, bytes, bytearray, SupportsFloat, SupportsIndex)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
