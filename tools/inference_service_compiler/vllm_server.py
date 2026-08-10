# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Internal process entry point for the programmatic vLLM server factory."""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_ROOT))

from inference_service_compiler.vllm_factory import run_server  # noqa: E402

if __name__ == "__main__":
    run_server(sys.argv[1:])
