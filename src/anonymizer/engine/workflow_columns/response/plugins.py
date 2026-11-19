# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.plugins import Plugin, PluginType

response_column_plugin = Plugin(
    config_qualified_name="anonymizer.engine.workflow_columns.response.config.ResponseColumnConfig",
    impl_qualified_name="anonymizer.engine.workflow_columns.response.impl.ResponseColumnGenerator",
    plugin_type=PluginType.COLUMN_GENERATOR,
)
