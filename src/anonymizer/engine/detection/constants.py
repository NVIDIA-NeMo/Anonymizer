# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


def _jinja(col: str) -> str:
    """Wrap a column constant in Jinja2 template syntax: ``{{ col }}``."""
    return "{{ " + col + " }}"
