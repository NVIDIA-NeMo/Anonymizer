# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.plugins import Plugin, PluginType

judge_column_plugin = Plugin(
    config_qualified_name="anonymizer.engine.workflow_columns.evaluation.judge.config.JudgeColumnConfig",
    impl_qualified_name="anonymizer.engine.workflow_columns.evaluation.judge.impl.JudgeColumnGenerator",
    plugin_type=PluginType.COLUMN_GENERATOR,
)
