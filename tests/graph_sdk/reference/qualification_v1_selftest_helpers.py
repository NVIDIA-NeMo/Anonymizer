# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Case lookup helpers shared by qualification reference self-tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import cast

# Preserve the original module-loading adapter used by these self-tests.
SPEC = importlib.util.spec_from_file_location(
    "qualification_v1_reference", Path(__file__).with_name("qualification_v1.py")
)
assert SPEC is not None and SPEC.loader is not None
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


def case(cid: str) -> reference.Obj:
    return next(x for x in reference.CASES if x["case_id"] == cid)


def result(cid: str) -> reference.Obj:
    return case(cid)["expected"]


def row(cid: str, target: str = "A") -> reference.Obj:
    return next(x for x in cast(list[reference.Json], result(cid)["targets"]) if x["target"] == target)
