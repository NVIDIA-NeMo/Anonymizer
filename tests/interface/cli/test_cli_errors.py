# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from anonymizer.config.anonymizer_config import AnonymizerConfig, Rewrite
from anonymizer.config.replace_strategies import Redact
from anonymizer.interface.cli.main import app


def test_invalid_source_exits(tmp_path: Path) -> None:
    """A non-existent source path causes SystemExit due to Pydantic validation."""
    with pytest.raises(SystemExit) as exc_info:
        app(
            [
                "run",
                "--source",
                str(tmp_path / "nonexistent.csv"),
                "--replace",
                "redact",
            ]
        )
    assert exc_info.value.code != 0


def test_missing_required_args_exits() -> None:
    """Omitting the required --source flag causes SystemExit."""
    with pytest.raises(SystemExit):
        app(["run", "--replace", "redact"])


def test_threshold_out_of_range_exits(tmp_path: Path) -> None:
    """gliner_threshold=2.0 violates the le=1.0 constraint → SystemExit."""
    csv_file = tmp_path / "data.csv"
    pd.DataFrame({"text": ["hello"]}).to_csv(csv_file, index=False)

    with pytest.raises(SystemExit) as exc_info:
        app(
            [
                "run",
                "--source",
                str(csv_file),
                "--replace",
                "redact",
                "--gliner-threshold",
                "2.0",
            ]
        )
    assert exc_info.value.code != 0


def test_both_modes_set_exits() -> None:
    """Setting both replace and rewrite on AnonymizerConfig violates the model_validator."""
    with pytest.raises(PydanticValidationError):
        AnonymizerConfig(replace=Redact(), rewrite=Rewrite())
