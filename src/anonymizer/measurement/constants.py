# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

MEASUREMENT_SCHEMA_VERSION = 1
DEFAULT_MEASUREMENT_ENV_PREFIX = "ANONYMIZER_MEASUREMENT_"
DD_TRACE_MODES = {"none", "last_message", "all_messages"}
DDTraceMode = Literal["none", "last_message", "all_messages"]
