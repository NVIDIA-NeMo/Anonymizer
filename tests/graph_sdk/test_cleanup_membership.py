# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Handle-key membership is independent of the surviving cleanup inventories."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from anonymizer.engine.graph_sdk.resources import ResourceId
from anonymizer.graph import EffectCode, EffectRejected
from anonymizer.graph._values import InvocationId
from tests.graph_sdk.child_qualification_fixtures import result_with
from tests.graph_sdk.test_delegated_execution import _execute
from tests.graph_sdk.test_delegated_readiness import _qualify
from tests.graph_sdk.test_delegated_worker import _loop_and_pool
from tests.graph_sdk.test_foreign_optional_tail import foreign_optional
from tests.graph_sdk.test_foreign_resources import fixture
from tests.graph_sdk.test_optional_tail import optional_fixture


def test_stateless_public_execution_needs_no_resource_registration():
    execution, services, _ = optional_fixture(readiness="available")
    result = asyncio.run(_execute(execution, services))
    assert result.implementation_resources == result.cleanup == result.cleanup_associations == ()
    assert len(_qualify(execution, result).qualified) == 1


async def executed(loop, pool, *, optional, failed=True):
    if optional:
        execution, services, _, clients, _, adapters, q = await foreign_optional(loop, pool)
        selected = next(a for a in adapters.values() if a._inner is q)
    else:
        execution, services, _, clients, _, adapters = await fixture(loop, pool)
        owner = next(p for p in execution.policies if p.delegated)
        node = execution.context.prepared.reservation_recipe[owner.delegated.child_slots[0]].template
        implementation = next(p.implementations[0].implementation for p in execution.policies if p.node == node)
        selected = adapters[implementation]
    if failed:
        selected._inner.close_result = "exception"
    result = await _execute(execution, services)
    assert all(c.closes == 1 for c in clients)
    assert len(_qualify(execution, result).qualified) == int(not failed)
    return execution, result, selected.lease.resource


@pytest.mark.parametrize("optional", [False, True])
@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("registration", ["retained", "selected_removed", "all_removed", "no_resource", "foreign_id"])
def test_paired_deletion_cannot_erase_declared_resource_even_when_membership_is_changed(optional, failed, registration):
    async def run():
        async with _loop_and_pool() as (loop, pool):
            execution, result, resource = await executed(loop, pool, optional=optional, failed=failed)
            rows = result.implementation_resources
            if registration == "selected_removed":
                rows = tuple(row for row in rows if row.resource != resource)
            elif registration == "all_removed":
                rows = ()
            elif registration == "no_resource":
                original = next(row for row in rows if row.resource == resource)
                forged = replace(original)
                object.__setattr__(forged, "resource", None)
                rows = tuple(forged if row is original else row for row in rows)
            elif registration == "foreign_id":
                rows = tuple(
                    replace(row, resource=ResourceId.new()) if row.resource == resource else row for row in rows
                )
            changed = result_with(
                result,
                implementation_resources=rows,
                cleanup=tuple(f for f in result.cleanup if f.resource != resource),
                cleanup_associations=tuple(a for a in result.cleanup_associations if a.resource != resource),
            )
            if registration == "no_resource":
                with pytest.raises(EffectRejected) as caught:
                    _qualify(execution, changed)
                assert caught.value.code == EffectCode.INVALID_TYPE
            else:
                qualified = _qualify(execution, changed)
                assert not qualified.qualified
                assert all("inconsistent_attribution" in target.withholding for target in qualified.targets)
            assert len(_qualify(execution, result).qualified) == int(not failed)

    asyncio.run(run())


@pytest.mark.parametrize(
    "mutation",
    [
        "fact_owner",
        "registered_owner",
        "foreign_invocation",
        "duplicate_key",
        "alias_owner",
        "detachment",
        "foreign_key",
        "stateless_resource",
    ],
)
def test_registered_handle_owner_and_complete_domain_are_checked(mutation):
    async def run():
        async with _loop_and_pool() as (loop, pool):
            execution, result, resource = await executed(loop, pool, optional=True)
            rows, cleanup = result.implementation_resources, result.cleanup
            q = next(row for row in rows if row.resource == resource)
            if mutation == "fact_owner":
                cleanup = tuple(
                    replace(f, owner="caller", disposition="left_open") if f.resource == resource else f
                    for f in cleanup
                )
            elif mutation == "registered_owner":
                rows = tuple(replace(row, owner="caller") if row is q else row for row in rows)
                cleanup = tuple(
                    replace(f, owner="caller", disposition="left_open") if f.resource == resource else f
                    for f in cleanup
                )
            elif mutation == "foreign_invocation":
                rows = tuple(
                    replace(row, invocation=InvocationId.new(plan=result.record.plan)) if row is q else row
                    for row in rows
                )
            elif mutation == "duplicate_key":
                rows = (*rows, q)
            elif mutation == "alias_owner":
                # The caller worker and SDK client cannot share contradictory lease metadata.
                rows = tuple(replace(row, resource=resource) if row.owner == "caller" else row for row in rows)
            elif mutation == "foreign_key":
                from anonymizer.engine.graph_sdk.capabilities import ImplementationRef

                rows = tuple(
                    replace(row, implementation=ImplementationRef(name="foreign", revision=1)) if row is q else row
                    for row in rows
                )
            elif mutation == "stateless_resource":
                selected = next(
                    item
                    for policy in execution.policies
                    for item in policy.implementations
                    if item.capability.resource_lifetime == "stateless"
                )
                rows = (
                    *rows,
                    replace(
                        q,
                        implementation=selected.implementation,
                        operation=selected.capability.operation,
                        configuration=selected.configuration,
                    ),
                )
            else:
                rows = tuple(
                    replace(row, safe_detachment="independent_after_dispatch") if row is q else row for row in rows
                )
            changed = result_with(result, implementation_resources=rows, cleanup=cleanup)
            qualified = _qualify(execution, changed)
            assert not qualified.qualified
            assert all("inconsistent_attribution" in t.withholding for t in qualified.targets)

    asyncio.run(run())


@pytest.mark.parametrize("field", ["implementation_resources", "cleanup", "cleanup_associations"])
def test_resource_inventories_are_bounded_before_indexes(field):
    async def run():
        async with _loop_and_pool() as (loop, pool):
            execution, result, _ = await executed(loop, pool, optional=True, failed=False)
            limit = execution.context.prepared.limits.max_capabilities
            changed = result_with(result, **{field: (getattr(result, field)[0],) * (limit + 1)})
            with pytest.raises(EffectRejected) as caught:
                _qualify(execution, changed)
            assert caught.value.code == EffectCode.LIMIT_EXCEEDED

    asyncio.run(run())


@pytest.mark.parametrize("field,value", [("resource", "not-an-id"), ("owner", "invented"), ("configuration", object())])
def test_raw_registration_fields_are_rechecked(field, value):
    async def run():
        async with _loop_and_pool() as (loop, pool):
            execution, result, resource = await executed(loop, pool, optional=True, failed=False)
            original = next(row for row in result.implementation_resources if row.resource == resource)
            forged = replace(original)
            object.__setattr__(forged, field, value)
            changed = result_with(
                result,
                implementation_resources=tuple(
                    forged if row is original else row for row in result.implementation_resources
                ),
            )
            with pytest.raises(EffectRejected):
                _qualify(execution, changed)

    asyncio.run(run())


@pytest.mark.parametrize("shape", ["list", "object", "missing"])
def test_registration_container_cannot_be_removed_or_retyped(shape):
    async def run():
        async with _loop_and_pool() as (loop, pool):
            execution, result, _ = await executed(loop, pool, optional=True, failed=False)
            changed = result_with(result)
            if shape == "missing":
                object.__delattr__(changed, "implementation_resources")
            else:
                object.__setattr__(changed, "implementation_resources", [] if shape == "list" else object())
            with pytest.raises(EffectRejected) as caught:
                _qualify(execution, changed)
            assert caught.value.code == EffectCode.INVALID_TYPE

    asyncio.run(run())


@pytest.mark.parametrize(
    "mutation",
    [
        "equivalent",
        "foreign",
        "deep_configuration",
        "raw_fields",
        "raw_implementation",
        "raw_operation",
        "deep_operation",
    ],
)
def test_retained_keys_resolve_before_untrusted_hashing(mutation):
    from anonymizer.engine.graph_sdk.capabilities import ConfigField, ConfigNull, ConfigSequence, FrozenConfig

    async def run():
        async with _loop_and_pool() as (loop, pool):
            execution, result, resource = await executed(loop, pool, optional=True, failed=False)
            original = next(row for row in result.implementation_resources if row.resource == resource)
            row = replace(
                original,
                implementation=replace(original.implementation),
                operation=replace(original.operation),
                configuration=replace(original.configuration),
            )
            if mutation == "foreign":
                row = replace(
                    row, configuration=FrozenConfig(fields=(ConfigField(name="foreign", value=ConfigNull()),))
                )
            elif mutation in {"deep_configuration", "deep_operation"}:
                value = ConfigNull()
                for _ in range(1500):
                    value = ConfigSequence(items=(value,))
                assert execution.context.prepared.limits.max_config_depth == 10
                if mutation == "deep_configuration":
                    row = replace(row, configuration=FrozenConfig(fields=(ConfigField(name="oversize", value=value),)))
                else:
                    object.__setattr__(row.operation, "name", value)
            elif mutation == "raw_fields":
                object.__setattr__(row.configuration, "fields", [])
            elif mutation == "raw_implementation":
                object.__setattr__(row.implementation, "name", [])
            elif mutation == "raw_operation":
                object.__setattr__(row.operation, "outcomes", [])
            changed = result_with(
                result,
                implementation_resources=tuple(row if r is original else r for r in result.implementation_resources),
            )
            qualified = _qualify(execution, changed)
            assert len(qualified.qualified) == int(mutation == "equivalent")
            if mutation != "equivalent":
                assert qualified.verified == ()
                assert all("inconsistent_attribution" in target.withholding for target in qualified.targets)
            assert len(_qualify(execution, result).qualified) == 1

    asyncio.run(run())


def test_key_matching_is_bounded_by_admitted_configuration_shape():
    from anonymizer.engine.graph_sdk.capabilities import (
        ConfigField,
        ConfigNull,
        ConfigSequence,
        FrozenConfig,
        config_size,
    )
    from anonymizer.engine.graph_sdk.qualification import _matches_admitted_key

    def config(depth):
        value = ConfigNull()
        for _ in range(depth - 2):
            value = ConfigSequence(items=(value,))
        return FrozenConfig(fields=(ConfigField(name="nested", value=value),))

    admitted = config(10)
    assert config_size(admitted) == (10, 11)
    assert _matches_admitted_key(admitted, config(10))
    assert not _matches_admitted_key(admitted, config(9))
    assert not _matches_admitted_key(admitted, config(11))
    assert not _matches_admitted_key(admitted, config(1502))
    assert not _matches_admitted_key(admitted, FrozenConfig(fields=()))
    assert not _matches_admitted_key(
        admitted, FrozenConfig(fields=(*admitted.fields, ConfigField(name="z", value=ConfigNull())))
    )


@pytest.mark.parametrize("missing", [False, True])
def test_registration_count_at_exact_domain_and_one_less(missing):
    async def run():
        async with _loop_and_pool() as (loop, pool):
            execution, result, _ = await executed(loop, pool, optional=True, failed=False)
            rows = result.implementation_resources
            changed = result_with(result, implementation_resources=rows[:-1] if missing else rows)
            assert len(_qualify(execution, changed).qualified) == int(not missing)

    asyncio.run(run())
