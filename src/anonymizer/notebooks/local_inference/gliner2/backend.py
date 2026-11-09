# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility alias for the native serving backend."""

from __future__ import annotations

import sys

from anonymizer.engine.local_inference.gliner2 import backend as _implementation

sys.modules[__name__] = _implementation
