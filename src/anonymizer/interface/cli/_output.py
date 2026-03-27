# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from anonymizer.engine.io.writer import write_output
from anonymizer.interface.results import AnonymizerResult, PreviewResult


def write_result(result: AnonymizerResult | PreviewResult, output_path: str | Path) -> Path:
    """Write the result dataframe to a file and return the resolved path."""
    return write_output(result.dataframe, output_path)
