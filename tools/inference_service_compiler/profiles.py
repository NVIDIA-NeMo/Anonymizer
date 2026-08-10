# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Human-authored profile transport for the inference service compiler."""

from __future__ import annotations

import tomllib
from pathlib import Path

from inference_service_compiler.models import InferenceIntent


def load_profile(path: Path) -> InferenceIntent:
    """Load and validate one TOML inference-service profile."""
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    return InferenceIntent.model_validate(payload)
