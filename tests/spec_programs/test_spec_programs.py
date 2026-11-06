# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest entry for the spec programs: every live check must PASS."""

from __future__ import annotations

from tests.spec_programs.run_all import collect


def test_spec_programs_pass() -> None:
    failures = [line for line in collect() if line["result"] == "FAIL"]
    assert not failures, failures
