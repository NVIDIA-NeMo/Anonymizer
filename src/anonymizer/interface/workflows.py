# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocols for workflows passed to `Anonymizer`, and the types they receive and return."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from anonymizer.config.options import DetectionOptions, RunContext
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult

if TYPE_CHECKING:
    import pandas as pd

    from anonymizer.config.rewrite import PrivacyGoal

__all__ = ["DetectionOptions", "DetectionWorkflow", "EntityDetectionResult", "RunContext"]


class DetectionWorkflow(Protocol):
    """A detection workflow, as passed to `Anonymizer(detection_workflow=...)`.

    Input frame: one row per target. The text is in the column `"__nemo_anonymizer_text_input__"` and the row id
    in `"_anonymizer_record_id"`.

    Result: `dataframe` carries `"final_entities"` for each returned row, and `source_positions` maps each returned
    row to its input row. The Rewrite route also reads `"_entities_by_value"`, `"tagged_text"` and, when
    `latent_goal` is set, `"_latent_entities"`. Rows that failed go in `failed_records`.
    """

    def run(
        self,
        dataframe: pd.DataFrame,
        *,
        options: DetectionOptions,
        context: RunContext,
        latent_goal: PrivacyGoal | None = None,
    ) -> EntityDetectionResult:
        """Detect entities in `dataframe`. `options` is the run's copy of `Detect`; `latent_goal` is the Rewrite
        privacy goal, or None when there is no latent-entity stage."""
        ...
