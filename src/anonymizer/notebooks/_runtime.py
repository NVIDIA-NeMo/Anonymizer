# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility wrappers for native GLiNER2."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from anonymizer.engine.local_inference.gliner2.contract import DeviceRequest
from anonymizer.engine.local_inference.runtime import stop_local_runtime
from anonymizer.interface.factory import NativeGliner
from anonymizer.interface.factory import create_anonymizer as create_public_anonymizer

if TYPE_CHECKING:
    from data_designer.config.models import ModelProvider

    from anonymizer.interface.anonymizer import Anonymizer


def create_anonymizer(
    *,
    model_configs: str | Path | None = None,
    model_providers: list[ModelProvider] | str | Path | None = None,
    gliner_device: DeviceRequest = "auto",
) -> Anonymizer:
    """Construct a client using the native singleton."""
    return create_public_anonymizer(
        gliner=NativeGliner(device=gliner_device),
        model_configs=model_configs,
        model_providers=model_providers,
    )


__all__ = ["create_anonymizer", "stop_local_runtime"]
