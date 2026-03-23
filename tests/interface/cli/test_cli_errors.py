# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pytest
import tyro

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput


@dataclasses.dataclass
class _RunArgs:
    config: AnonymizerConfig
    data: AnonymizerInput
    output: str | None = None


def test_invalid_source_exits(tmp_path: Path) -> None:
    """A non-existent source path causes SystemExit due to Pydantic validation."""
    with pytest.raises(SystemExit):
        tyro.cli(
            _RunArgs,
            args=[
                "--data.source",
                str(tmp_path / "nonexistent.csv"),
                "config.replace:redact",
            ],
        )


def test_missing_required_args_exits() -> None:
    """Omitting the required --data.source flag causes SystemExit."""
    with pytest.raises(SystemExit):
        tyro.cli(
            _RunArgs,
            args=["config.replace:redact"],
        )


def test_threshold_out_of_range_exits(tmp_path: Path) -> None:
    """gliner_threshold=2.0 violates the le=1.0 constraint → SystemExit."""
    csv_file = tmp_path / "data.csv"
    pd.DataFrame({"text": ["hello"]}).to_csv(csv_file, index=False)

    with pytest.raises(SystemExit):
        tyro.cli(
            _RunArgs,
            args=[
                "--data.source",
                str(csv_file),
                "--config.detect.gliner-threshold",
                "2.0",
                "config.replace:redact",
            ],
        )


def test_both_modes_set_exits(tmp_path: Path) -> None:
    """Setting both replace and rewrite violates the model_validator → SystemExit."""
    csv_file = tmp_path / "data.csv"
    pd.DataFrame({"text": ["hello"]}).to_csv(csv_file, index=False)

    with pytest.raises(SystemExit):
        tyro.cli(
            _RunArgs,
            args=[
                "--data.source",
                str(csv_file),
                "config.replace:redact",
                "config.rewrite:rewrite",
                "--config.rewrite.privacy-goal.protect",
                "Direct identifiers and quasi-identifiers",
                "--config.rewrite.privacy-goal.preserve",
                "Semantic meaning and general utility",
            ],
        )
