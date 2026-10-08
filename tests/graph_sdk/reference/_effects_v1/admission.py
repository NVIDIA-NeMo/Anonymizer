# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent effects_v1 reference: admission."""

from __future__ import annotations

import json
from typing import cast

from tests.graph_sdk.reference._effects_v1.materialization import (
    _natural,
)
from tests.graph_sdk.reference._effects_v1.materialization_admission import (
    _admit_materializations,
)
from tests.graph_sdk.reference._effects_v1.model import (
    RUNTIME_CONDITIONS,
    Object,
    _array,
    _expected_mapping_keys,
    _mapping_key,
    _object,
    _strings,
)
from tests.graph_sdk.reference._effects_v1.request_facts import (
    _reject,
)


def admit(declaration: Object) -> Object:
    allowed = {
        "admission_limits",
        "binding_targets",
        "declared_outcomes",
        "execution_policies",
        "outcome_categories",
        "capability_catalog",
        "runtime_mappings",
        "targets",
        "materializations",
        "materialization_limits",
        "root_artifacts",
        "root_input_types",
    }
    if set(declaration) - allowed:
        return _reject("invalid_value")
    materialized = _admit_materializations(declaration)
    if materialized is not None:
        return materialized
    policies_value = declaration.get("execution_policies", [])
    if not isinstance(policies_value, list):
        return _reject("invalid_type")
    catalog = declaration.get("capability_catalog", [])
    if not isinstance(catalog, list):
        return _reject("invalid_type")
    limits = declaration.get("admission_limits", {"max_capabilities": len(catalog)})
    if not isinstance(limits, dict) or not _natural(limits.get("max_capabilities")):
        return _reject("invalid_type")
    if len(catalog) > cast(int, limits["max_capabilities"]):
        return _reject("limit_exceeded")
    if any(not isinstance(raw, dict) for raw in catalog):
        return _reject("invalid_type")
    catalog_ids = [_object(raw).get("implementation") for raw in catalog]
    if len(catalog_ids) != len(set(cast(list[str], catalog_ids))):
        return _reject("duplicate")
    if any(not isinstance(raw, dict) for raw in policies_value):
        return _reject("invalid_type")
    policies = [_object(raw) for raw in policies_value]
    if any(policy.get("kind") not in ("local", "external", "decision") for policy in policies):
        return _reject("invalid_value")
    if any(policy.get("owner", "I0") != "I0" for policy in policies):
        return _reject("foreign_owner")
    nodes = [policy.get("node") for policy in policies]
    if len(nodes) != len({json.dumps(node, sort_keys=True) for node in nodes}):
        return _reject("duplicate")
    if any(not isinstance(policy.get("implementations"), list) for policy in policies):
        return _reject("invalid_type")
    if any(not policy["implementations"] for policy in policies):
        return _reject("implementation_count")
    if any(policy.get("retry_owner") == "implementation" for policy in policies):
        return _reject("unsupported")
    declared_values = _strings(declaration.get("declared_outcomes", []))
    if len(declared_values) != len(set(declared_values)):
        return _reject("duplicate")
    declared_outcomes = set(declared_values)
    mappings = declaration.get("runtime_mappings", [])
    if not isinstance(mappings, list) or any(not isinstance(value, dict) for value in mappings):
        return _reject("invalid_type")
    typed_mappings = [_object(value) for value in mappings]
    mapping_fields = {"category", "condition", "failure", "outcome", "reported_outcome"}
    if any(set(value) != mapping_fields for value in typed_mappings):
        return _reject("invalid_value")
    if any(value.get("condition") not in RUNTIME_CONDITIONS for value in typed_mappings):
        return _reject("invalid_value")
    mapping_keys = [_mapping_key(value) for value in typed_mappings]
    if len(mapping_keys) != len(set(mapping_keys)):
        return _reject("duplicate")
    if any(value.get("outcome") not in declared_outcomes | {None} for value in typed_mappings):
        return _reject("unsupported")
    categories = _object(declaration.get("outcome_categories", {}))
    if any(
        value.get("outcome") is not None and categories.get(cast(str, value["outcome"])) != value.get("category")
        for value in typed_mappings
    ):
        return _reject("contradictory")
    if any(value.get("outcome") is None and value.get("category") == "success" for value in typed_mappings):
        return _reject("contradictory")
    if any(
        value.get("condition") == "request_inconsistent" and value.get("category") != "inconsistent"
        for value in typed_mappings
    ):
        return _reject("contradictory")
    if any(
        value.get("condition") == "cancel_before_start"
        and (value.get("outcome") is not None or value.get("category") != "blocked")
        for value in typed_mappings
    ):
        return _reject("contradictory")
    targets = set(_strings(declaration.get("targets", [])))
    binding_targets = _strings(declaration.get("binding_targets", []))
    if any(target not in targets for target in binding_targets):
        return _reject("foreign_owner")
    for policy in policies:
        implementations = _array(policy["implementations"])
        if policy["kind"] in ("local", "decision") and len(implementations) != 1:
            return _reject("implementation_count")
        if policy.get("retry_owner") == "implementation":
            return _reject("retry_owner")
        if any(_object(item).get("physical_policy") != policy.get("physical_policy") for item in implementations):
            return _reject("changed_failover_policy")
    if len(policies) == 1:
        policy = policies[0]
        outcomes_value = policy.get("result_outcomes", [])
        if not isinstance(outcomes_value, list) or any(not isinstance(value, str) for value in outcomes_value):
            return _reject("invalid_type")
        outcomes = cast(list[str], outcomes_value)
        if len(outcomes) != len(set(outcomes)):
            return _reject("duplicate")
        if policy["kind"] in ("local", "external") and not outcomes:
            return _reject("missing")
        if policy["kind"] == "decision" and outcomes:
            return _reject("contradictory")
        expected_keys = _expected_mapping_keys(cast(str, policy["kind"]), outcomes)
        actual_keys = set(mapping_keys)
        if expected_keys - actual_keys:
            return _reject("missing")
        if actual_keys - expected_keys:
            return _reject("extra")
    return {"status": "accepted"}


def recheck_capabilities(declaration: Object) -> Object:
    admitted = _object(declaration["admitted"])
    initial = admit(admitted)
    if initial["status"] != "accepted":
        return initial
    supplied = _array(declaration["capability_catalog"])
    current = {_object(raw)["implementation"]: _object(raw) for raw in supplied}
    for raw in _array(admitted["capability_catalog"]):
        expected = _object(raw)
        if current.get(expected["implementation"]) != expected:
            return _reject("changed_failover_policy")
    return {"status": "accepted"}
