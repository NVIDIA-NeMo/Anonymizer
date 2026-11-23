# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest entry for the spec programs: every check of every program must pass."""

from __future__ import annotations

import importlib

import pytest

from tests.spec_programs.run_all import PROGRAMS


@pytest.mark.parametrize("name", PROGRAMS)
def test_spec_program_passes(name: str) -> None:
    lines = importlib.import_module(f"tests.spec_programs.{name}").program()
    assert lines, f"{name} reported no checks"
    failures = [line for line in lines if line["result"] != "PASS"]
    assert not failures, failures
