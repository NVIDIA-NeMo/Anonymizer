# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.plugins import Plugin, PluginType

detection_transform_plugin = Plugin(
    config_qualified_name="anonymizer.engine.workflow_columns.detection.config.DetectionTransformConfig",
    impl_qualified_name="anonymizer.engine.workflow_columns.detection.impl.DetectionTransformGenerator",
    plugin_type=PluginType.COLUMN_GENERATOR,
)

chunked_validation_plugin = Plugin(
    config_qualified_name="anonymizer.engine.workflow_columns.detection.config.ChunkedValidationConfig",
    impl_qualified_name="anonymizer.engine.workflow_columns.detection.impl.ChunkedValidationGenerator",
    plugin_type=PluginType.COLUMN_GENERATOR,
)
