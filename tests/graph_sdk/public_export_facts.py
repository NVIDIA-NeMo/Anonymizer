# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Location-independent facts about every ``anonymizer.graph`` export."""

from __future__ import annotations

import enum
import inspect
import pickle
from types import FunctionType
from typing import Any

import anonymizer.graph as graph


def _signature(value: object) -> str:
    # Exception constructors are not authoring surface, and test tooling may wrap them to record raises.
    if not callable(value) or (isinstance(value, type) and issubclass(value, BaseException)):
        return "<none>"
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return "<none>"


def _pickle_outcome(value: object) -> str:
    try:
        restored = pickle.loads(pickle.dumps(value))
    except Exception as error:  # noqa: BLE001 - the error type is the recorded fact
        return f"raises {type(error).__name__}"
    if restored is value and isinstance(value, (type, FunctionType)):
        return "identity"
    # typing aliases and enum members are cached objects; equality is the stable fact.
    return "equal" if restored == value else "different"


def export_facts() -> dict[str, dict[str, Any]]:
    facts: dict[str, dict[str, Any]] = {}
    for name in graph.__all__:
        value = getattr(graph, name)
        entry: dict[str, Any] = {
            "kind": type(value).__name__,
            "qualname": getattr(value, "__qualname__", None),
            "signature": _signature(value),
            "pickle": _pickle_outcome(value),
        }
        if isinstance(value, type):
            entry["mro"] = [item.__qualname__ for item in value.__mro__]
            entry["repr_owner"] = next(c.__qualname__ for c in value.__mro__ if "__repr__" in vars(c))
            entry["eq_owner"] = next(c.__qualname__ for c in value.__mro__ if "__eq__" in vars(c))
            if issubclass(value, enum.Enum):
                entry["members"] = {member.name: [repr(member), _pickle_outcome(member)] for member in value}
            if issubclass(value, BaseException):
                entry["instance_pickle"] = _pickle_outcome(_exception_sample(value))
        facts[name] = entry
    return facts


def _exception_sample(value: type[BaseException]) -> BaseException | None:
    for args in ((), ("x",)):
        try:
            return value(*args)
        except Exception:  # noqa: BLE001
            continue
    return None
