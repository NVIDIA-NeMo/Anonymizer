# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No CLI module defines a top-level function or class twice. ruff's F811 does not flag a redefined name that
matches its dummy-variable pattern, which every underscore-prefixed name does, so the check is made here."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pytest

import anonymizer.interface.cli as cli

_MODULES = sorted(Path(cli.__file__).parent.glob("*.py"))


@pytest.mark.parametrize("path", _MODULES, ids=lambda path: path.name)
def test_no_top_level_name_is_defined_twice(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = Counter(
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )
    assert [name for name, count in names.items() if count > 1] == []
