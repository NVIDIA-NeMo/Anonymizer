# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent effects_v1 reference: materialization admission."""

from __future__ import annotations

from typing import cast

from tests.graph_sdk.reference._effects_v1.materialization import (
    _natural,
)
from tests.graph_sdk.reference._effects_v1.model import (
    Json,
    Object,
    _array,
    _object,
)
from tests.graph_sdk.reference._effects_v1.request_facts import (
    _reject,
)


def _admit_materializations(declaration: Object) -> Object | None:
    raw_declarations = declaration.get("materializations", [])
    if not isinstance(raw_declarations, list):
        return _reject("invalid_type")
    if not raw_declarations:
        return None
    raw_limits = declaration.get("materialization_limits")
    required_limits = {
        "max_artifact_bytes",
        "max_artifacts",
        "max_collection_items",
        "max_declarations",
        "max_provenance_edges",
    }
    if not isinstance(raw_limits, dict) or set(raw_limits) != required_limits:
        return _reject("invalid_type")
    limits = _object(raw_limits)
    if any(not _natural(limits[key]) for key in required_limits):
        return _reject("invalid_type")
    if len(raw_declarations) > cast(int, limits["max_declarations"]):
        return _reject("limit_exceeded")
    if any(not isinstance(raw, dict) for raw in raw_declarations):
        return _reject("invalid_type")
    specs = [_object(raw) for raw in raw_declarations]
    base_fields = {
        "association",
        "declaration",
        "item_type",
        "kind",
        "max_bytes",
        "max_items",
        "node",
        "output_type",
        "path",
        "port",
        "source",
        "target",
        "selector_inputs",
    }
    if any(
        set(spec)
        != (
            base_fields
            | ({"version_selection"} if "version_selection" in spec else set())
            | ({"publication"} if spec.get("version_selection") == "latest" else set())
        )
        for spec in specs
    ):
        return _reject("invalid_value")
    if any(
        spec["path"] not in ("initial", "adaptive") or spec["kind"] not in ("single", "collection") for spec in specs
    ):
        return _reject("invalid_value")
    if any(
        not all(
            isinstance(spec[key], str) and spec[key]
            for key in ("association", "item_type", "node", "output_type", "port", "target")
        )
        or not _natural(spec["max_items"])
        or not _natural(spec["max_bytes"])
        for spec in specs
    ):
        return _reject("invalid_type")
    if any(
        spec["path"] == "initial" and (not isinstance(spec["declaration"], str) or not isinstance(spec["source"], str))
        for spec in specs
    ):
        return _reject("missing")
    if any(
        spec["path"] == "adaptive" and (spec["declaration"] is not None or spec["source"] is not None) for spec in specs
    ):
        return _reject("contradictory")
    if any(
        spec["path"] == "initial"
        and (
            spec.get("version_selection", "exact_one") not in ("exact_one", "latest")
            or (spec.get("version_selection") == "latest" and spec.get("kind") != "single")
            or (spec.get("version_selection") == "latest" and cast(int, spec["max_items"]) <= 0)
            or (
                spec.get("version_selection") == "latest"
                and (
                    not isinstance(spec.get("publication"), dict)
                    or set(_object(spec["publication"]))
                    != {"activation", "artifact_type", "inputs", "node", "outcome", "output_port"}
                    or not all(
                        isinstance(_object(spec["publication"]).get(field), str)
                        and bool(_object(spec["publication"])[field])
                        for field in ("activation", "artifact_type", "node", "outcome", "output_port")
                    )
                    or _object(spec["publication"]).get("node") != spec["node"]
                    or _object(spec["publication"]).get("inputs") != [spec["port"]]
                )
            )
        )
        for spec in specs
    ):
        return _reject("contradictory")
    if any(
        spec.get("selector_inputs") != ([f"RootInputKey:{spec['target']}:input"] if spec["path"] == "adaptive" else [])
        for spec in specs
    ):
        return _reject("invalid_provenance")
    identities = [tuple(spec[key] for key in ("path", "target", "node", "port", "declaration")) for spec in specs]
    if len(identities) != len(set(identities)):
        return _reject("duplicate")
    roots = [_object(raw) for raw in _array(declaration.get("root_artifacts", []))]
    if any(
        set(root) != {"artifact_type", "port", "target"}
        or root["target"] not in {spec["target"] for spec in specs}
        or not all(isinstance(root[field], str) and root[field] for field in root)
        for root in roots
    ):
        return _reject("foreign_owner")
    if len([(root["target"], root["port"]) for root in roots]) != len(
        set((root["target"], root["port"]) for root in roots)
    ):
        return _reject("duplicate")
    if any(
        spec["kind"] == "single"
        and (
            spec["output_type"] != spec["item_type"]
            or (spec.get("version_selection", "exact_one") == "exact_one" and spec["max_items"] != 1)
        )
        for spec in specs
    ):
        return _reject("contradictory")
    if any(
        spec["kind"] == "collection" and (spec["output_type"] == spec["item_type"] or cast(int, spec["max_items"]) < 1)
        for spec in specs
    ):
        return _reject("contradictory")
    schemas: dict[str, tuple[Json, Json]] = {}
    for spec in specs:
        shape = (spec["kind"], spec["item_type"])
        output = cast(str, spec["output_type"])
        if output in schemas and schemas[output] != shape:
            return _reject("contradictory")
        schemas[output] = shape
    collection_types = {cast(str, spec["output_type"]) for spec in specs if spec["kind"] == "collection"}
    if any(spec["item_type"] in collection_types for spec in specs):
        return _reject("contradictory")
    roots = declaration.get("root_input_types", [])
    if not isinstance(roots, list) or any(not isinstance(value, str) for value in roots):
        return _reject("invalid_type")
    if collection_types & set(cast(list[str], roots)):
        return _reject("contradictory")
    return None
