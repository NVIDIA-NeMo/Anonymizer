# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from anonymizer.engine.io.writer import write_output
from anonymizer.interface.results import AnonymizerResult, PreviewResult


def format_summary(result: AnonymizerResult | PreviewResult) -> str:
    """Return a human-readable one-line summary of an anonymization result."""
    num_records = len(result.dataframe)
    num_failures = len(result.failed_records)
    return f"Processed {num_records} record(s), {num_failures} failure(s)."


def write_result(result: AnonymizerResult | PreviewResult, output_path: str | Path) -> Path:
    """Write the result dataframe to a file and return the resolved path."""
    return write_output(result.dataframe, output_path)
