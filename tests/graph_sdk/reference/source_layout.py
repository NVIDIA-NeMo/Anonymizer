# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source inventory and isolation checks for the split reference implementations."""

from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
from pathlib import Path

REFERENCE_DIR = Path(__file__).parent


def implementation_paths(name: str) -> tuple[Path, ...]:
    """Return the facade and every module in its private implementation package."""
    return (REFERENCE_DIR / f"{name}.py", *sorted((REFERENCE_DIR / f"_{name}").glob("*.py")))


def source_digest(name: str, *, self_tests: bool = False) -> str:
    """Hash sorted relative paths, NUL separators, and complete source bytes."""
    paths = (
        tuple(REFERENCE_DIR.glob(f"test_{name}*.py")) + tuple(REFERENCE_DIR.glob(f"{name}_selftest*.py"))
        if self_tests
        else implementation_paths(name)
    )
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(REFERENCE_DIR).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def assert_reference_imports(name: str) -> None:
    """Check every implementation import against stdlib and its own package."""
    own_package = f"tests.graph_sdk.reference._{name}"
    for path in implementation_paths(name):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            imports = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            for imported in imports:
                assert (
                    imported.split(".")[0] in sys.stdlib_module_names
                    or imported == "__future__"
                    or imported == own_package
                    or imported.startswith(own_package + ".")
                ), (path, imported)


def assert_isolated_generation(name: str) -> None:
    """Generate without importing product code or unrelated reference packages."""
    script = """
import builtins, importlib, pathlib, sys
name, root = sys.argv[1:]
sys.path.insert(0, root)
original_import = builtins.__import__
allowed = ('tests', 'tests.graph_sdk', 'tests.graph_sdk.reference',
           'tests.graph_sdk.reference.' + name)
private = 'tests.graph_sdk.reference._' + name

def guarded(module, *args, **kwargs):
    if module == 'anonymizer' or module.startswith('anonymizer.') or 'donor' in module.lower():
        raise AssertionError(module)
    if module.startswith('tests') and module not in allowed and module != private and not module.startswith(private + '.'):
        raise AssertionError(module)
    return original_import(module, *args, **kwargs)

builtins.__import__ = guarded
module = importlib.import_module('tests.graph_sdk.reference.' + name)
assert module.canonical_bytes(module.generate_cases()) == module.canonical_bytes(module.generate_cases())
assert not any(key == 'anonymizer' or key.startswith('anonymizer.') for key in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, name, str(REFERENCE_DIR.parents[2])],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
