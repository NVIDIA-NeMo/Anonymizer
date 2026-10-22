# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pin the public ``anonymizer.graph`` export surface independently of defining modules."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import FunctionType

import anonymizer.graph as graph
from tests.graph_sdk.public_export_facts import export_facts

_REFERENCE = Path(__file__).parent / "export_surface" / "public_exports.json"


def test_export_order_and_names_are_pinned() -> None:
    assert list(json.loads(_REFERENCE.read_text())) == graph.__all__


def test_export_facts_are_pinned() -> None:
    expected = json.loads(_REFERENCE.read_text())
    actual = export_facts()
    mismatched = {name: (expected[name], actual[name]) for name in expected if expected[name] != actual[name]}
    assert mismatched == {}


def test_exports_resolve_to_one_object_per_name() -> None:
    for name in graph.__all__:
        value = getattr(graph, name)
        assert getattr(graph, name) is value
        if isinstance(value, (type, FunctionType)):
            assert pickle.loads(pickle.dumps(value)) is value
