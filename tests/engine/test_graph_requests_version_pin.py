# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tomllib
from pathlib import Path

from anonymizer.engine.ndd._graph_requests import REQUIRED_DATA_DESIGNER_VERSION

_PYPROJECT_PATH = Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_pyproject_pins_data_designer_to_required_version() -> None:
    """``pyproject.toml`` must declare an exact pin matching the runtime gate.

    The runtime gate in ``_graph_requests.py`` only accepts
    ``REQUIRED_DATA_DESIGNER_VERSION``; if ``pyproject.toml`` allows a
    different (or wider) range, a fresh install can resolve a DataDesigner
    version that the gate then rejects at runtime. This test keeps the two
    declarations from drifting apart.
    """
    pyproject = tomllib.loads(_PYPROJECT_PATH.read_text())
    dependencies = pyproject["project"]["dependencies"]

    data_designer_specifiers = [dep for dep in dependencies if dep.split()[0].startswith("data-designer")]
    assert len(data_designer_specifiers) == 1, (
        f"expected exactly one data-designer dependency declaration, found {data_designer_specifiers}"
    )

    expected = f"data-designer=={REQUIRED_DATA_DESIGNER_VERSION}"
    assert data_designer_specifiers[0] == expected, (
        f"pyproject.toml declares {data_designer_specifiers[0]!r}, expected {expected!r} "
        "to match REQUIRED_DATA_DESIGNER_VERSION in _graph_requests.py"
    )
