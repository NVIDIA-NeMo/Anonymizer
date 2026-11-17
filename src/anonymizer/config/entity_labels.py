# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


def normalize_entity_label(label: str) -> str:
    """Return the canonical identity used for entity-label comparisons."""
    return label.strip().casefold()
