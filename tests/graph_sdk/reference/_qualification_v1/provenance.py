# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent qualification_v1 reference: provenance."""

from __future__ import annotations

from typing import cast

from tests.graph_sdk.reference._qualification_v1.model import (
    Obj,
    arr,
    obj,
    reject,
)


def validate_provenance(d: Obj, s: Obj) -> Obj | None:
    targets = [cast(str, value) for value in arr(d["targets"])]
    arts = obj(s["artifacts"])
    ports = obj(s["ports"])
    prov = obj(s["provenance"])
    materialized_by_artifact: dict[str, list[Obj]] = {}
    for raw_value in prov.values():
        value = obj(raw_value)
        if value.get("source") in ("bound_input", "map_item"):
            materialized_by_artifact.setdefault(cast(str, value["artifact"]), []).append(value)
    artifact_lineages: dict[tuple[str, str], list[tuple[str, Obj]]] = {}
    for ref, raw_artifact in arts.items():
        artifact_value = obj(raw_artifact)
        artifact_lineages.setdefault(
            (cast(str, artifact_value["invocation"]), cast(str, artifact_value["key"])), []
        ).append((ref, artifact_value))
    for (invocation, _), versions in artifact_lineages.items():
        if len(versions) < 2:
            continue
        if len({obj(value)["version"] for _, value in versions}) != len(versions):
            return reject("duplicate")
        identities = []
        for ref, artifact_value in versions:
            owners = materialized_by_artifact.get(ref, [])
            if len(owners) != 1:
                return reject("missing")
            owner = owners[0]
            if owner["source"] == "bound_input":
                binding = obj(owner["binding_artifact"])
                if binding.get("version") != artifact_value["version"]:
                    return reject("contradictory")
                identities.append(
                    (
                        "initial",
                        invocation,
                        binding.get("declaration"),
                        binding.get("key"),
                        owner.get("target"),
                    )
                )
            else:
                if owner.get("item_version") != artifact_value["version"]:
                    return reject("contradictory")
                expander_entry = obj(obj(s["entries"]).get(cast(str, owner.get("expander")), {}))
                member_entry = obj(obj(s["entries"]).get(cast(str, owner.get("member")), {}))
                if not expander_entry:
                    return reject("missing")
                if (
                    owner.get("target") not in targets
                    or expander_entry.get("target") != owner.get("target")
                    or member_entry.get("target") != owner.get("target")
                ):
                    return reject("foreign_owner")
                identities.append(
                    (
                        "map",
                        invocation,
                        owner.get("expander"),
                        owner.get("target"),
                        owner.get("item_key"),
                    )
                )
        if len(set(identities)) != 1:
            return reject("contradictory")
    allowed_sources = {
        "bound_input",
        "initial_collection",
        "map_item",
        "operation_output",
        "root_input",
    }
    bindings = [obj(x) for x in arr(d["binding_inputs"])]
    collections = [obj(x) for x in arr(d["initial_collections"])]
    map_inputs = [obj(x) for x in arr(d["map_inputs"])]
    receipt = obj(s["binding_receipt"])
    receipt_sources = [obj(source) for source in arr(receipt.get("sources"))]
    receipt_artifacts = [obj(item) for item in arr(receipt.get("artifacts"))]
    if receipt:
        source_sites = [
            {field: source[field] for field in ("declaration", "node", "port", "source", "target")}
            for source in receipt_sources
        ]
        if len({cast(str, source["declaration"]) for source in receipt_sources}) != len(receipt_sources):
            return reject("duplicate")
        expected_source_sites = [
            {field: binding[field] for field in ("declaration", "node", "port", "source", "target")}
            for binding in bindings
        ]
        if source_sites != expected_source_sites:
            return reject("contradictory")
        artifacts_by_declaration: dict[str, list[Obj]] = {}
        for item in receipt_artifacts:
            artifacts_by_declaration.setdefault(cast(str, item["declaration"]), []).append(item)
        for binding in bindings:
            retained = artifacts_by_declaration.get(cast(str, binding["declaration"]), [])
            if binding["materialization"] == "single" and binding["version_selection"] == "exact_one":
                if len(retained) != 1:
                    return reject("contradictory")
            elif binding["materialization"] == "single" and binding["version_selection"] == "latest":
                keys = {item["key"] for item in retained}
                if not retained or len(keys) != 1:
                    return reject("contradictory")
            if binding["version_selection"] != "latest":
                continue
            requests = [
                request
                for request in map(obj, obj(s["binding_requests"]).values())
                if obj(request.get("association")).get("declaration") == binding["declaration"]
            ]
            if len(requests) != 1:
                return reject("foreign_owner" if obj(s["binding_requests"]) else "missing")
            request = requests[0]
            association = obj(request["association"])
            if association != {field: binding[field] for field in ("declaration", "node", "port", "source", "target")}:
                return reject("foreign_owner")
            if (
                request.get("phase") != "settled"
                or obj(request.get("terminal")).get("outcome") != "retrieved"
                or obj(request.get("settlement")) != {"remote_stopped": True, "usage": "known"}
            ):
                return reject("missing")
            resource = cast(str, request["resource"])
            cleanup_association = obj(obj(s["binding_cleanup_associations"]).get(resource, {}))
            cleanup = obj(obj(s["binding_cleanups"]).get(resource, {}))
            if not cleanup_association or not cleanup:
                return reject("missing")
            if cleanup_association != {"owner": "sdk", "purpose": "accounting", "targets": [binding["target"]]}:
                return reject("foreign_owner")
            if cleanup.get("disposition") != "closed":
                return reject("contradictory")
        for item in receipt_artifacts:
            site = {field: item[field] for field in ("declaration", "node", "port", "source", "target")}
            if site not in source_sites:
                return reject("foreign_owner")
    entries = obj(s["entries"])
    memberships = obj(s["memberships"])
    admitted_nodes = obj(d["node_kinds"])
    output_dependencies = {
        (item["node"], item["outcome"], item["port"]): item for item in map(obj, arr(d["output_dependencies"]))
    }
    input_producers = obj(s["input_producers"])
    admitted_input_ports: dict[str, set[str]] = {}
    for dependency in output_dependencies.values():
        admitted_input_ports.setdefault(cast(str, dependency["node"]), set()).update(
            cast(str, value) for value in arr(dependency["inputs"])
        )
    for production in map(obj, arr(d["productions"])):
        admitted_input_ports.setdefault(cast(str, production["node"]), set()).update(
            cast(str, value) for value in arr(production["consumed_ports"])
        )
        if production["subject_source"] == "input":
            admitted_input_ports.setdefault(cast(str, production["node"]), set()).add(
                cast(str, production["subject_port"])
            )
    for raw_subgraph in arr(d["subgraphs"]):
        subgraph = obj(raw_subgraph)
        admitted_input_ports.setdefault(cast(str, subgraph["node"]), set()).add(cast(str, subgraph["input_port"]))
        if subgraph["body_source"] == "node_output":
            admitted_input_ports.setdefault(cast(str, subgraph["body_node"]), set()).add(
                cast(str, subgraph["body_input"])
            )
    expected_input_keys = {
        key
        for key, raw_port in ports.items()
        if obj(raw_port).get("port") in admitted_input_ports.get(cast(str, obj(raw_port).get("node")), set())
        and obj(raw_port).get("activation") in entries
    }
    dynamic_item_ports = {cast(str, map_input["item_input"]) for map_input in map_inputs}
    expected_input_keys.update(
        key
        for key, raw_port in ports.items()
        if obj(raw_port).get("port") in dynamic_item_ports
        and obj(entries.get(cast(str, obj(raw_port).get("activation")), {})).get("parent") is not None
    )
    if expected_input_keys - set(input_producers):
        return reject("missing")
    if set(input_producers) - expected_input_keys:
        return reject("contradictory")
    for key, raw_input in input_producers.items():
        input_fact = obj(raw_input)
        port_fact = obj(ports.get(key, {}))
        producer_fact = obj(prov.get(cast(str, input_fact.get("producer")), {}))
        if producer_fact and producer_fact.get("artifact") not in arts:
            return reject("missing")
        if producer_fact and producer_fact.get("target") not in targets:
            return reject("foreign_owner")
        if (
            input_fact.get("activation") != port_fact.get("activation")
            or input_fact.get("node") != port_fact.get("node")
            or input_fact.get("port") != port_fact.get("port")
            or input_fact.get("target") != port_fact.get("target")
            or producer_fact.get("artifact") != port_fact.get("artifact")
            or producer_fact.get("target") != port_fact.get("target")
        ):
            return reject("contradictory")
    for producer, raw_collection in obj(s["collection_values"]).items():
        collection = obj(raw_collection)
        producer_fact = obj(prov.get(producer, {}))
        if not producer_fact:
            return reject("missing")
        activation = cast(str, producer_fact.get("activation"))
        source_entry = obj(entries.get(activation, {}))
        owners = [
            map_input
            for map_input in map_inputs
            if map_input.get("expander") == source_entry.get("node")
            and map_input.get("outcome") == source_entry.get("state_outcome")
            and map_input.get("membership_port") == producer_fact.get("port")
            and producer == f"OP:{activation}:{collection.get('target')}:{map_input.get('membership_port')}"
        ]
        if len(owners) != 1:
            return reject("contradictory")
        if (
            producer_fact.get("source") != "operation_output"
            or producer_fact.get("artifact") != collection.get("artifact")
            or producer_fact.get("target") != collection.get("target")
        ):
            return reject("contradictory")
    for provenance_key, raw_value in prov.items():
        value = obj(raw_value)
        source = value.get("source")
        if source not in allowed_sources:
            return reject("unsupported")
        if value.get("target") not in targets:
            return reject("foreign_owner")
        artifact_ref = cast(str, value.get("artifact"))
        artifact_value = obj(arts.get(artifact_ref, {}))
        if not artifact_value:
            return reject("missing")
        if artifact_value.get("invocation") != obj(d["execution"])["invocation"]:
            return reject("foreign_owner")
        if artifact_value.get("target") not in (value.get("target"), "shared"):
            return reject("foreign_owner")
        if type(value.get("decision")) is not bool:
            return reject("invalid_type")
        if source in ("operation_output", "root_input"):
            if any(
                value.get(field) is not None
                for field in (
                    "binding_artifact",
                    "declaration",
                    "expander",
                    "item_key",
                    "item_version",
                    "member",
                    "root_target",
                )
            ):
                return reject("contradictory")
            source_matches = [
                fact
                for fact in map(obj, ports.values())
                if fact.get("activation") == value.get("activation")
                and fact.get("artifact") == artifact_ref
                and fact.get("node") == value.get("node")
                and fact.get("port") == value.get("port")
                and fact.get("target") == value.get("target")
            ]
            source_entry = obj(entries.get(cast(str, value.get("activation")), {}))
            retained_undispatched_root = (
                source == "root_input" and source_entry.get("closed_unstarted") is True and not source_matches
            )
            if len(source_matches) != 1 and not retained_undispatched_root:
                return reject("contradictory")
            map_output = any(
                map_input.get("expander") == value.get("node")
                and map_input.get("outcome") == source_entry.get("state_outcome")
                and map_input.get("membership_port") == value.get("port")
                for map_input in map_inputs
            )
            structural_output = next(
                (
                    obj(item)
                    for item in arr(d["subgraphs"])
                    if obj(item).get("node") == value.get("node")
                    and obj(item).get("outcome") == source_entry.get("state_outcome")
                    and obj(item).get("port") == value.get("port")
                ),
                {},
            )
            if (
                admitted_nodes.get(cast(str, value.get("node"))) != "operation"
                and not map_output
                and not structural_output
            ) or (source_entry.get("node") != value.get("node") or source_entry.get("target") != value.get("target")):
                return reject("contradictory")
            if source == "root_input" and arr(value["parents"]):
                return reject("contradictory")
            if source == "operation_output":
                if structural_output:
                    if structural_output.get("body_source") == "workflow_input":
                        wrapper_input = obj(
                            input_producers.get(f"{value.get('activation')}|{structural_output.get('input_port')}", {})
                        )
                        actual_parents = [cast(str, parent) for parent in arr(value["parents"])]
                        if (
                            len(actual_parents) != 1
                            or actual_parents[0] != wrapper_input.get("producer")
                            or obj(prov.get(actual_parents[0], {})).get("artifact") != artifact_ref
                            or wrapper_input.get("target") != value.get("target")
                        ):
                            return reject("contradictory")
                        continue
                    body_parents = [
                        key
                        for key, candidate_raw in prov.items()
                        if obj(candidate_raw).get("source") == "operation_output"
                        and obj(candidate_raw).get("node") == structural_output.get("body_node")
                        and obj(candidate_raw).get("port") == structural_output.get("body_port")
                        and obj(candidate_raw).get("target") == value.get("target")
                        and obj(entries.get(cast(str, obj(candidate_raw).get("activation")), {})).get("parent")
                        == value.get("activation")
                    ]
                    if len(arr(value["parents"])) != len(set(cast(str, x) for x in arr(value["parents"]))) or set(
                        cast(str, x) for x in arr(value["parents"])
                    ) != set(body_parents):
                        return reject("contradictory")
                    body_parent = obj(prov.get(body_parents[0], {})) if len(body_parents) == 1 else {}
                    body_entry = obj(entries.get(cast(str, body_parent.get("activation")), {}))
                    body_input = obj(
                        input_producers.get(
                            f"{body_parent.get('activation')}|{structural_output.get('body_input')}", {}
                        )
                    )
                    if (
                        body_parent.get("artifact") != artifact_ref
                        or body_entry.get("node") != structural_output.get("body_node")
                        or body_input.get("target") != value.get("target")
                    ):
                        return reject("contradictory")
                    continue
                dependency = output_dependencies.get(
                    (value.get("node"), source_entry.get("state_outcome"), value.get("port"))
                ) or output_dependencies.get((value.get("node"), "ok", value.get("port")))
                if dependency is None:
                    return reject("missing")
                producer_facts = [
                    obj(input_producers.get(f"{value.get('activation')}|{input_port}", {}))
                    for input_port in arr(dependency["inputs"])
                ]
                if any(
                    not fact or fact.get("node") != value.get("node") or fact.get("target") != value.get("target")
                    for fact in producer_facts
                ):
                    return reject("missing")
                expected_parents = {cast(str, fact["producer"]) for fact in producer_facts}
                actual_parents = [cast(str, parent) for parent in arr(value["parents"])]
                if len(actual_parents) != len(set(actual_parents)) or set(actual_parents) != expected_parents:
                    return reject("contradictory")
                identity_input = dependency.get("identity_input")
                if identity_input is not None:
                    identity_fact = obj(input_producers.get(f"{value.get('activation')}|{identity_input}", {}))
                    identity_parent = obj(prov.get(cast(str, identity_fact.get("producer")), {}))
                    if identity_parent.get("artifact") != artifact_ref:
                        return reject("contradictory")
                elif any(obj(prov.get(parent, {})).get("artifact") == artifact_ref for parent in expected_parents):
                    return reject("contradictory")
        if source == "bound_input":
            binding = obj(value.get("binding_artifact"))
            if set(binding) != {"declaration", "key", "version"}:
                return reject("invalid_type")
            if (
                type(binding.get("key")) is not int
                or cast(int, binding["key"]) < 0
                or type(binding.get("version")) is not int
                or cast(int, binding["version"]) <= 0
            ):
                return reject("invalid_value")
            identity = {
                "declaration": binding.get("declaration"),
                "node": value.get("node"),
                "port": value.get("port"),
                "target": value.get("target"),
            }
            binding_matches = [
                admitted
                for admitted in bindings
                if all(admitted.get(field) == field_value for field, field_value in identity.items())
            ]
            if len(binding_matches) != 1:
                return reject("contradictory")
            receipt_match = [
                item
                for item in receipt_artifacts
                if item.get("declaration") == binding.get("declaration")
                and item.get("key") == binding.get("key")
                and item.get("version") == binding.get("version")
                and item.get("node") == value.get("node")
                and item.get("port") == value.get("port")
                and item.get("target") == value.get("target")
            ]
            if len(receipt_match) != 1:
                return reject("missing")
            if any(
                value.get(field) is not None
                for field in (
                    "activation",
                    "declaration",
                    "expander",
                    "item_key",
                    "item_version",
                    "member",
                    "root_target",
                )
            ):
                return reject("contradictory")
            if value.get("decision") is not False or arr(value["parents"]):
                return reject("contradictory")
        if source == "initial_collection":
            identity = {
                "declaration": value.get("declaration"),
                "node": value.get("node"),
                "port": value.get("port"),
                "target": value.get("target"),
            }
            if identity not in collections:
                return reject("contradictory")
            if any(
                value.get(field) is not None
                for field in (
                    "activation",
                    "binding_artifact",
                    "expander",
                    "item_key",
                    "item_version",
                    "member",
                    "root_target",
                )
            ):
                return reject("contradictory")
            expected_parents = {
                key
                for key, parent_raw in prov.items()
                if obj(parent_raw).get("source") == "bound_input"
                and obj(obj(parent_raw).get("binding_artifact")).get("declaration") == value.get("declaration")
                and obj(parent_raw).get("node") == value.get("node")
                and obj(parent_raw).get("port") == value.get("port")
                and obj(parent_raw).get("target") == value.get("target")
            }
            expected_inventory = {
                (
                    item.get("declaration"),
                    item.get("key"),
                    item.get("version"),
                    item.get("node"),
                    item.get("port"),
                    item.get("target"),
                )
                for item in receipt_artifacts
                if item.get("declaration") == value.get("declaration")
                and item.get("node") == value.get("node")
                and item.get("port") == value.get("port")
                and item.get("target") == value.get("target")
            }
            actual_inventory = {
                (
                    obj(obj(prov[parent]).get("binding_artifact")).get("declaration"),
                    obj(obj(prov[parent]).get("binding_artifact")).get("key"),
                    obj(obj(prov[parent]).get("binding_artifact")).get("version"),
                    obj(prov[parent]).get("node"),
                    obj(prov[parent]).get("port"),
                    obj(prov[parent]).get("target"),
                )
                for parent in expected_parents
            }
            if (
                set(cast(str, parent) for parent in arr(value["parents"])) != expected_parents
                or actual_inventory != expected_inventory
            ):
                return reject("contradictory")
            if value.get("decision") is not False:
                return reject("contradictory")
        if source == "map_item":
            member = cast(str, value.get("member"))
            expander = cast(str, value.get("expander"))
            member_entry = obj(entries.get(member, {}))
            expander_entry = obj(entries.get(expander, {}))
            if not expander_entry or not member_entry:
                return reject("missing")
            matches = [
                map_input
                for map_input in map_inputs
                if map_input.get("expander") == expander_entry.get("node")
                and map_input.get("outcome") == expander_entry.get("state_outcome")
            ]
            membership = obj(memberships.get(expander, {}))
            item_key = value.get("item_key")
            item_version = value.get("item_version")
            parent_keys = [cast(str, parent) for parent in arr(value["parents"])]
            if len(matches) != 1:
                return reject("contradictory")
            map_input = matches[0]
            if (
                value.get("target") not in targets
                or expander_entry.get("target") != value.get("target")
                or member_entry.get("target") != value.get("target")
                or membership.get("target") != value.get("target")
            ):
                return reject("foreign_owner")
            expected_parent = f"OP:{expander}:{value.get('target')}:{map_input['membership_port']}"
            if (
                value.get("activation") is not None
                or value.get("binding_artifact") is not None
                or value.get("declaration") is not None
                or value.get("node") is not None
                or value.get("root_target") is not None
                or value.get("port") != map_input["item_input"]
                or value.get("decision") is not False
                or type(item_key) is not int
                or item_key < 0
                or type(item_version) is not int
                or item_version <= 0
                or expander_entry.get("node_kind") not in ("operation", "container")
                or member_entry.get("parent") != expander
                or member not in arr(membership.get("members"))
                or len(parent_keys) != 1
                or parent_keys[0] != expected_parent
            ):
                return reject("contradictory")
            parent_fact = obj(prov.get(parent_keys[0], {}))
            collection = obj(obj(s["collection_values"]).get(parent_keys[0], {}))
            member_port = obj(ports.get(f"{member}|{map_input['item_input']}", {}))
            retained_undispatched_item = member_entry.get("closed_unstarted") is True and not member_port
            membership_members = [cast(str, item) for item in arr(membership.get("members"))]
            ordered_members = sorted(membership_members, key=lambda item: cast(int, obj(entries[item])["occurrence"]))
            occurrences = [obj(entries[item]).get("occurrence") for item in ordered_members]
            if len(occurrences) != len(set(occurrences)):
                return reject("contradictory")
            member_index = ordered_members.index(member) if member in ordered_members else -1
            items = [obj(item) for item in arr(collection.get("items"))]
            if not any(item.get("key") == item_key and item.get("version") == item_version for item in items):
                return reject("missing")
            if (
                parent_fact.get("source") != "operation_output"
                or parent_fact.get("activation") != expander
                or parent_fact.get("node") != map_input["expander"]
                or parent_fact.get("port") != map_input["membership_port"]
                or parent_fact.get("target") != value.get("target")
                or collection.get("artifact") != parent_fact.get("artifact")
                or collection.get("target") != value.get("target")
                or len(items) != len(ordered_members)
                or member_index < 0
                or items[member_index].get("key") != item_key
                or items[member_index].get("version") != item_version
                or not retained_undispatched_item
                and (
                    member_port.get("artifact") != artifact_ref
                    or member_port.get("target") != value.get("target")
                    or member_port.get("node") != member_entry.get("node")
                )
            ):
                return reject("contradictory")
        if value.get("decision") is True and not any(
            fact.get("artifact") == artifact_ref
            and fact.get("role") == "decision"
            and fact.get("target") == value.get("target")
            for fact in map(obj, ports.values())
        ):
            return reject("contradictory")
    for binding in bindings:
        if binding["materialization"] != "single" or binding["version_selection"] != "latest":
            continue
        inventory = [item for item in receipt_artifacts if item.get("declaration") == binding["declaration"]]
        maximum = max(cast(int, item["version"]) for item in inventory)
        selected_owners = [
            (key, value)
            for key, value in map(lambda pair: (pair[0], obj(pair[1])), prov.items())
            if value.get("source") == "bound_input"
            and obj(value.get("binding_artifact")).get("declaration") == binding["declaration"]
            and obj(value.get("binding_artifact")).get("version") == maximum
        ]
        if len(selected_owners) != 1:
            return reject("missing")
        selected_key, selected_owner = selected_owners[0]
        destination_ports = [
            value
            for value in map(obj, ports.values())
            if value.get("node") == binding["node"]
            and value.get("port") == binding["port"]
            and value.get("target") == binding["target"]
        ]
        if len(destination_ports) != 1:
            return reject("missing")
        destination = destination_ports[0]
        producer = obj(input_producers.get(f"{destination.get('activation')}|{binding['port']}", {}))
        if destination.get("artifact") != selected_owner.get("artifact") or producer.get("producer") != selected_key:
            return reject("contradictory")
    return None
