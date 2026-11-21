# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The engine raises the public error types without importing the interface layer."""

from __future__ import annotations

import ast
import pickle
from pathlib import Path

import pytest

import anonymizer
import anonymizer.engine.errors as engine_errors
import anonymizer.interface.errors as interface_errors

_ENGINE = Path(engine_errors.__file__).parent


def test_no_engine_module_imports_the_interface_layer() -> None:
    importers = []
    for path in sorted(_ENGINE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            names += [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
            if any(name == "anonymizer.interface" or name.startswith("anonymizer.interface.") for name in names):
                importers.append(path.relative_to(_ENGINE).as_posix())
    assert importers == []


@pytest.mark.parametrize("name", interface_errors.__all__)
def test_each_public_error_is_one_class_under_its_public_name(name: str) -> None:
    cls = getattr(interface_errors, name)
    assert cls is getattr(engine_errors, name)
    assert getattr(anonymizer, name, cls) is cls  # the package exports every one but AnonymizerWorkflowError
    assert cls.__module__ == "anonymizer.interface.errors"
    assert type(pickle.loads(pickle.dumps(cls("message")))) is cls
    assert issubclass(cls, interface_errors.AnonymizerError)
