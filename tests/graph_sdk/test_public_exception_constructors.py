# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pin the constructor signatures of every exception exported by ``anonymizer.graph``."""

from __future__ import annotations

import inspect

import pytest

import anonymizer.graph as graph

_EXPECTED = {
    "call": {
        "BindingRejected": "(code: 'EffectCode') -> 'None'",
        "ContractViolation": "(code: 'ValidationCode') -> 'None'",
        "EffectRejected": "(code: 'EffectCode') -> 'None'",
        "ExecutionRejected": "(code: 'EffectCode') -> 'None'",
        "NestedEventLoopError": "<builtin>",
        "PreparationRejected": "(code: 'PreparationCode') -> 'None'",
    },
    "init": {
        "BindingRejected": "(self, code: 'EffectCode') -> 'None'",
        "ContractViolation": "(self, code: 'ValidationCode') -> 'None'",
        "EffectRejected": "(self, code: 'EffectCode') -> 'None'",
        "ExecutionRejected": "(self, code: 'EffectCode') -> 'None'",
        "NestedEventLoopError": "(self, /, *args, **kwargs)",
        "PreparationRejected": "(self, code: 'PreparationCode') -> 'None'",
    },
}


def _signature(value: object) -> str:
    try:
        return str(inspect.signature(value))  # ty: ignore[invalid-argument-type]
    except ValueError:
        return "<builtin>"


def _exception_exports() -> list[str]:
    return [
        name
        for name in graph.__all__
        if isinstance(getattr(graph, name), type) and issubclass(getattr(graph, name), BaseException)
    ]


def test_exported_exception_constructors_are_pinned() -> None:
    init = {name: _signature(getattr(graph, name).__init__) for name in _exception_exports()}
    call = {name: _signature(getattr(graph, name)) for name in _exception_exports()}
    if any("recording" in getattr(graph, name).__init__.__qualname__ for name in _exception_exports()):
        pytest.skip("a rejection-recording plugin replaced exception __init__ without preserving its signature")
    assert {"init": init, "call": call} == _EXPECTED
