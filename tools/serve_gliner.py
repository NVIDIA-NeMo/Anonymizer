# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility entrypoint for the notebook/development GLiNER2 server."""

from __future__ import annotations

from anonymizer.notebooks._runtime import run_development_server

if __name__ == "__main__":
    run_development_server()
