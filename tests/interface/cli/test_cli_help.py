# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from anonymizer.interface.cli.main import app


@pytest.mark.parametrize("subcommand", ["run", "preview", "validate"])
def test_help_exits_zero(subcommand: str, capsys: pytest.CaptureFixture[str]) -> None:
    """Each subcommand prints help and exits 0."""
    with pytest.raises(SystemExit) as exc:
        app([subcommand, "--help"])
    assert exc.value.code == 0
