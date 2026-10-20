# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checked construction: every owner-keyed graph value is complete and keyed at construction."""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import pkgutil
from functools import cache
from typing import Any

import pytest

import anonymizer.engine.graph_sdk as graph_sdk
from anonymizer.engine.graph_sdk._request_values import (
    BindingAssociation,
    BindingDeclarationId,
    BindingId,
    PhysicalRequestPolicy,
    RequestPolicyBinding,
)
from anonymizer.engine.graph_sdk.executor import AdmittedExecutionPlan, ExecutionResult
from anonymizer.engine.graph_sdk.resources import ResourceLease
from tests.graph_sdk.evidence_fixtures import _execute_assessment

# Bump deliberately when a new owner-keyed value is added; a silent change means a class escaped this suite.
EXPECTED_OWNER_KEYED = 32


def _owner_keyed() -> dict[str, type[Any]]:
    found: dict[str, type[Any]] = {}
    for info in pkgutil.walk_packages(graph_sdk.__path__, graph_sdk.__name__ + "."):
        for value in vars(importlib.import_module(info.name)).values():
            if isinstance(value, type) and dataclasses.is_dataclass(value):
                names = {field.name for field in dataclasses.fields(value)}
                # `_key` is declared but excluded from fields(): an InitVar owner key.
                if "_key" in value.__dataclass_fields__ and "_key" not in names and names:
                    found[f"{value.__module__.rsplit('.', 1)[-1]}.{value.__qualname__}"] = value
    return dict(sorted(found.items()))


OWNER_KEYED = _owner_keyed()


def _owner_key(kind: type[Any]) -> object:
    post_init = kind.__post_init__
    names = [name for name in post_init.__code__.co_names if name.endswith("_KEY")]
    assert len(names) == 1, names
    return post_init.__globals__[names[0]]


def _arguments(kind: type[Any]) -> dict[str, object]:
    return {field.name: object() for field in dataclasses.fields(kind) if field.init}


def _required(kind: type[Any]) -> list[str]:
    return [
        field.name
        for field in dataclasses.fields(kind)
        if field.init and field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING
    ]


@cache
def _execution() -> tuple[AdmittedExecutionPlan, ExecutionResult]:
    return asyncio.run(_execute_assessment())


def _valid(kind: type[Any]) -> Any:
    """A real instance: owner factories where construction also validates ingress, else the real key."""
    if kind is AdmittedExecutionPlan:
        return _execution()[0]
    if kind is ExecutionResult:
        return _execution()[1]
    if kind is ResourceLease:
        return ResourceLease.create(owner="caller", safe_detachment="forbidden", handle=object())
    declaration = BindingDeclarationId.new(binding=BindingId.new(), ordinal=0)
    if kind is BindingDeclarationId:
        return declaration
    if kind is RequestPolicyBinding:
        policy = PhysicalRequestPolicy(
            visibility="none", pre_dispatch_control="none", retry_owner="none", replay="never", max_attempts=1
        )
        return RequestPolicyBinding.create(
            association=BindingAssociation(declaration=declaration), policies=frozenset({policy})
        )
    arguments = _arguments(kind)
    if kind.__name__ == "ObservationReceipt":  # consistency is checked at construction
        arguments.update(schema_version=1, pairs=(), delivery="not_configured", delivered_pairs=0)
    elif kind.__name__ == "DiagnosisSummary":
        arguments.update(reasons=())
    return kind(_key=_owner_key(kind), **arguments)


def test_owner_keyed_inventory_is_pinned() -> None:
    assert len(OWNER_KEYED) == EXPECTED_OWNER_KEYED, sorted(OWNER_KEYED)


@pytest.mark.parametrize("kind", OWNER_KEYED.values(), ids=OWNER_KEYED.keys())
def test_construction_requires_the_owner_key(kind: type[Any]) -> None:
    with pytest.raises(TypeError, match="'_key'"):
        kind(**_arguments(kind))
    with pytest.raises(TypeError, match="created by their owner"):
        kind(_key=object(), **_arguments(kind))


@pytest.mark.parametrize("kind", OWNER_KEYED.values(), ids=OWNER_KEYED.keys())
def test_every_required_field_is_enforced(kind: type[Any]) -> None:
    key = _owner_key(kind)
    for name in _required(kind):
        arguments = _arguments(kind)
        del arguments[name]
        with pytest.raises(TypeError, match=f"missing 1 required keyword-only argument: '{name}'"):
            kind(_key=key, **arguments)


@pytest.mark.parametrize("kind", OWNER_KEYED.values(), ids=OWNER_KEYED.keys())
def test_copies_require_the_owner_key(kind: type[Any]) -> None:
    value = _valid(kind)
    assert type(value) is kind
    with pytest.raises(ValueError, match="InitVar '_key' must be specified"):
        dataclasses.replace(value)
    assert type(dataclasses.replace(value, _key=_owner_key(kind))) is kind
