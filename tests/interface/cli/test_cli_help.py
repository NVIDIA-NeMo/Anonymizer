# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from anonymizer.interface.cli.main import app


@pytest.mark.parametrize("subcommand", ["run", "preview", "validate"])
def test_help_exits_zero(subcommand: str, capsys: pytest.CaptureFixture[str]) -> None:
    """Each subcommand prints help and exits 0."""
    with pytest.raises(SystemExit) as exc:
        app([subcommand, "--help"])
    assert exc.value.code == 0


def test_console_script_starts_with_data_designer_telemetry_validation(tmp_path: Path) -> None:
    """The real CLI must not set a deployment type rejected by Data Designer."""
    env = os.environ.copy()
    env.pop("ANONYMIZER_USAGE_TYPE", None)
    env.pop("NEMO_DEPLOYMENT_TYPE", None)
    console_script = Path(sys.executable).parent / "anonymizer"

    completed = subprocess.run(
        [console_script, "--help"],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "NeMo Anonymizer CLI" in completed.stdout
