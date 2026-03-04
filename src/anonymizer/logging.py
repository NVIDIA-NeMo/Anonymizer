# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logging utilities for the Anonymizer package."""

from __future__ import annotations

import logging

LOG_INDENT = "  |-- "

# Suppress DataDesigner's verbose logs by default. Users can see them by
# lowering the data_designer logger to DEBUG:
#   logging.getLogger("data_designer").setLevel(logging.DEBUG)
logging.getLogger("data_designer").setLevel(logging.WARNING)
