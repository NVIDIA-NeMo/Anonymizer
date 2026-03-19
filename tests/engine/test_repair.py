# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock

from data_designer.config.column_configs import CustomColumnConfig
from data_designer.config.models import ModelConfig

from anonymizer.config.models import RewriteModelSelection
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import COL_REWRITTEN_TEXT
from anonymizer.engine.rewrite.repair import RepairWorkflow

_STUB_PRIVACY_GOAL = PrivacyGoal(
    protect="Direct identifiers, quasi-identifier combinations, and latent inferences",
    preserve="General utility, content quality, and semantic meaning of the original text",
)


def test_repair_columns_pipeline(
    stub_model_configs: list[ModelConfig],
    stub_rewrite_model_selection: RewriteModelSelection,
) -> None:
    wf = RepairWorkflow(adapter=Mock())
    cols = wf.columns(
        selected_models=stub_rewrite_model_selection,
        privacy_goal=_STUB_PRIVACY_GOAL,
        effective_threshold=1.0,
    )

    assert len(cols) == 2
    assert isinstance(cols[0], CustomColumnConfig)
    assert isinstance(cols[1], CustomColumnConfig)
    assert cols[1].name == COL_REWRITTEN_TEXT
