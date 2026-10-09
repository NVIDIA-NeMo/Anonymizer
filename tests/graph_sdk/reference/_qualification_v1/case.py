# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent qualification_v1 reference: case."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import cast

from tests.graph_sdk.reference._qualification_v1.declarations import (
    admit,
    set_output_dependency,
)
from tests.graph_sdk.reference._qualification_v1.events import (
    artifact,
    binding_receipt,
    binding_ref,
    input_producer,
    port,
    provenance,
)
from tests.graph_sdk.reference._qualification_v1.model import (
    Json,
    Obj,
    arr,
    obj,
)
from tests.graph_sdk.reference._qualification_v1.qualification import (
    reduce,
)

VERSIONED_CASES = {
    "validity/candidate_stale",
    "validity/candidate_stale_no_assessment",
    "validity/evidence_stale",
    "validity/consumed_stale",
    "validity/stale_precedes_unknown",
    "validity/evidence_output_replaced",
    "selective/a_only",
    "selective/b_only",
    "selective/a_b",
    "selective/a_only_stale",
    "selective/b_only_stale",
    "selective/a_b_stale",
    "selective/decision_stale",
    "provenance/version_edge",
    "lineage/initial_two_versions",
    "lineage/two_selected_current_versions",
    "lineage/invented_version",
    "lineage/declaration_crossover",
    "lineage/target_crossover",
    "lineage/invocation_crossover",
    "lineage/artifacts_exact",
    "lineage/artifacts_one_over",
    "lineage/bytes_exact",
    "lineage/bytes_one_over",
    "lineage/provenance_exact",
    "lineage/provenance_one_over",
    "lineage/rejected_publication_rollback",
    "lineage/latest_older_selected",
    "lineage/latest_missing_binding_request",
    "lineage/latest_foreign_binding_association",
    "lineage/latest_missing_binding_settlement",
    "lineage/latest_missing_binding_cleanup",
    "lineage/latest_foreign_cleanup_target",
    "validity/final_candidate_sibling_substitution",
}


def _replace_refs(value: Json, replacements: Mapping[str, str]) -> Json:
    if isinstance(value, str):
        return replacements.get(value, value)
    if isinstance(value, list):
        return [_replace_refs(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_refs(item, replacements) for key, item in value.items()}
    return value


def _materialize_initial_versions(case_id: str, d: Obj, e: list[Obj]) -> None:
    artifacts = [item for item in e if item.get("kind") == "artifact"]
    by_key: dict[str, list[Obj]] = {}
    for value in artifacts:
        by_key.setdefault(cast(str, value["key"]), []).append(value)
    revisions_by_key = {
        cast(str, value["key"]): value
        for value in e
        if value.get("kind") == "revision" and value.get("collection") == "artifacts"
    }
    if case_id == "provenance/version_edge":
        old = next(value for value in artifacts if value["ref"] == "XAv0")
        e.insert(e.index(old) + 1, artifact("XAv1", "A", "artifact"))
        revisions_by_key["XA"]["value"] = 1
        by_key["XA"] = [old, next(value for value in e if value.get("ref") == "XAv1")]
    for key, revision in revisions_by_key.items():
        values = by_key.get(key, [])
        if values and revision["value"] not in {value["version"] for value in values} and case_id in VERSIONED_CASES:
            template = values[0]
            ref = f"{key}v{revision['value']}"
            created = artifact(ref, cast(str, template["target"]), "artifact")
            e.insert(e.index(template) + 1, created)
            values.append(created)
            by_key[key] = values
    replacements: dict[str, str] = {}
    shifts: dict[str, int] = {}
    for key, values in by_key.items():
        if len(values) > 1 and min(cast(int, value["version"]) for value in values) == 0:
            shifts[key] = 1
            for value in values:
                replacements[cast(str, value["ref"])] = f"{key}v{cast(int, value['version']) + 1}"
    if replacements:
        updated = cast(list[Obj], _replace_refs(e, replacements))
        e[:] = updated
        for value in e:
            if value.get("kind") == "artifact" and value.get("key") in shifts:
                value["version"] = cast(int, value["version"]) + shifts[cast(str, value["key"])]
            if (
                value.get("kind") == "revision"
                and value.get("collection") == "artifacts"
                and value.get("key") in shifts
            ):
                value["value"] = cast(int, value["value"]) + shifts[cast(str, value["key"])]
    artifacts = [item for item in e if item.get("kind") == "artifact"]
    by_key = {}
    for value in artifacts:
        by_key.setdefault(cast(str, value["key"]), []).append(value)
    multi = {key: values for key, values in by_key.items() if len(values) > 1}
    if not multi:
        return
    existing_materialized = {
        cast(str, value["artifact"])
        for value in e
        if value.get("kind") == "provenance" and value.get("source") in ("bound_input", "map_item")
    }
    pending = {
        key: values
        for key, values in multi.items()
        if any(cast(str, value["ref"]) not in existing_materialized for value in values)
    }
    if not pending:
        return
    sites: list[tuple[str, str, str, str]] = []
    receipt_artifacts: list[tuple[str, int, int]] = []
    lineage_facts: list[Obj] = []
    lifecycle: list[Obj] = []
    selections: dict[str, tuple[str, str, str]] = {}
    for index, (key, values) in enumerate(sorted(pending.items())):
        declaration_id = f"DVER{index}"
        target = cast(str, values[0]["target"])
        refs = {cast(str, value["ref"]) for value in values}
        root_source = next(
            (
                value
                for value in e
                if value.get("kind") == "provenance"
                and value.get("source") == "root_input"
                and value.get("artifact") in refs
            ),
            None,
        )
        port_name = "evidence_version" if root_source is None and key == "EA" else cast(str, obj(root_source)["port"])
        node = "N"
        site = (declaration_id, node, port_name, target)
        sites.append(site)
        association = {
            "declaration": declaration_id,
            "node": node,
            "port": port_name,
            "source": f"SRC:{declaration_id}@1",
            "target": target,
        }
        request = f"BINDREQ:{declaration_id}"
        resource = f"BINDRES:{declaration_id}"
        lifecycle.extend(
            [
                {
                    "association": association,
                    "kind": "binding_reserve",
                    "policy": "P0",
                    "purpose": "initial_binding",
                    "request": request,
                    "resource": resource,
                },
                {"kind": "binding_dispatch", "request": request},
                {"kind": "binding_result", "outcome": "retrieved", "request": request},
                {
                    "kind": "binding_settlement",
                    "remote_stopped": True,
                    "request": request,
                    "usage": "known",
                },
            ]
        )
        arr(d["binding_inputs"]).append(
            {
                "declaration": declaration_id,
                "materialization": "single",
                "node": node,
                "port": port_name,
                "source": f"SRC:{declaration_id}@1",
                "target": target,
                "version_selection": "latest",
            }
        )
        for value in sorted(values, key=lambda item: cast(int, item["version"])):
            version = cast(int, value["version"])
            receipt_artifacts.append((declaration_id, 0, version))
            producer = f"BOUND:{key}:{version}"
            lineage_facts.append(
                provenance(
                    producer,
                    cast(str, value["ref"]),
                    target,
                    source="bound_input",
                    node=node,
                    port_name=port_name,
                    binding_artifact=binding_ref(declaration_id, 0, version),
                )
            )
        selected = max(values, key=lambda item: cast(int, item["version"]))
        selections[key] = (cast(str, selected["ref"]), f"BOUND:{key}:{selected['version']}", port_name)
    ownership: list[Obj] = []
    for declaration_id, _, _, target in sites:
        resource = f"BINDRES:{declaration_id}"
        ownership.extend(
            [
                {
                    "kind": "binding_cleanup_association",
                    "owner": "sdk",
                    "purpose": "accounting",
                    "resource": resource,
                    "targets": [target],
                },
                {"disposition": "closed", "kind": "binding_cleanup", "resource": resource},
            ]
        )
    e[0:0] = [*lifecycle, binding_receipt(sites, receipt_artifacts), *ownership]
    first_provenance = next(i for i, value in enumerate(e) if value.get("kind") == "provenance")
    e[first_provenance:first_provenance] = lineage_facts

    # Initial ``latest`` retains every version but supplies only the numeric
    # maximum to the scalar destination.  The input producer is its exact
    # BoundInputKey; no collection or synthetic scalar extraction exists.
    for key, values in sorted(multi.items()):
        selected_ref, bound_key, port_name = selections[key]
        refs = {cast(str, value["ref"]) for value in values}
        root_sources = [
            value
            for value in e
            if value.get("kind") == "provenance"
            and value.get("source") == "root_input"
            and value.get("artifact") in refs
        ]
        for source in root_sources:
            source_key = cast(str, source["key"])
            for value in e:
                if value.get("kind") == "input_producer" and value.get("producer") == source_key:
                    value["producer"] = bound_key
                if value.get("kind") == "provenance":
                    value["parents"] = [
                        bound_key if parent == source_key else parent for parent in arr(value["parents"])
                    ]
            e.remove(source)
        for value in e:
            if value.get("kind") == "port" and value.get("artifact") in refs:
                value["artifact"] = selected_ref
            if value.get("kind") == "assessment":
                value["consumed"] = {
                    consumed_port: selected_ref if consumed_ref in refs else consumed_ref
                    for consumed_port, consumed_ref in obj(value["consumed"]).items()
                }
                if value.get("evidence_artifact") in refs:
                    value["evidence_artifact"] = selected_ref
                if value.get("subject_artifact") in refs:
                    value["subject_artifact"] = selected_ref
            if value.get("kind") == "final" and value.get("candidate") in refs:
                value["candidate"] = selected_ref
            if (
                value.get("kind") == "provenance"
                and value.get("artifact") in refs
                and value.get("source") == "operation_output"
            ):
                value["artifact"] = selected_ref
            if value.get("kind") == "provenance" and value.get("key") == "EVID:A" and value.get("artifact") in refs:
                value["artifact"] = selected_ref

    if case_id in {"validity/evidence_stale", "validity/evidence_output_replaced"}:
        evidence_values = sorted(multi["EA"], key=lambda item: cast(int, item["version"]))
        selected = evidence_values[-1]
        selected_ref = cast(str, selected["ref"])
        bound_key = f"BOUND:EA:{selected['version']}"
        assessment_fact = next(value for value in e if value.get("kind") == "assessment")
        assessment_fact["consumed"] = {"evidence_version": selected_ref}
        production = obj(arr(d["productions"])[0])
        production["consumed_ports"] = ["evidence_version"]
        obj(arr(d["requirements"])[0])["consumed_ports"] = ["evidence_version"]
        evidence_output = next(
            value for value in e if value.get("kind") == "provenance" and value.get("key") == "EVID:A"
        )
        next(value for value in e if value.get("kind") == "port" and value.get("port") == "evidence")["artifact"] = (
            selected_ref
        )
        evidence_output["artifact"] = selected_ref
        insert_at = e.index(evidence_output)
        e[insert_at:insert_at] = [
            port("ROOT:A", "A", "evidence_version", selected_ref, "evidence"),
            input_producer("ROOT:A", "A", "N", "evidence_version", bound_key),
        ]
        evidence_output["parents"] = [bound_key]
        set_output_dependency(d, "N", "evidence", ("evidence_version",), identity_input="evidence_version")

    if case_id == "provenance/version_edge":
        selected_ref, _, _ = selections["XA"]
        next(value for value in e if value.get("kind") == "port" and value.get("port") == "version")["artifact"] = (
            selected_ref
        )
        next(value for value in e if value.get("kind") == "provenance" and value.get("key") == "VERSION")[
            "artifact"
        ] = selected_ref
        next(
            value
            for value in e
            if value.get("kind") == "revision" and value.get("collection") == "artifacts" and value.get("key") == "XA"
        )["value"] = max(cast(int, value["version"]) for value in multi["XA"])

    stale_keys = {
        "validity/candidate_stale": "A",
        "validity/candidate_stale_no_assessment": "A",
        "validity/evidence_stale": "EA",
        "validity/evidence_output_replaced": "EA",
        "validity/consumed_stale": "XA",
        "validity/stale_precedes_unknown": "XA",
        "selective/b_only": "YA",
        "selective/a_b": "YA",
        "selective/a_only_stale": "XA",
        "selective/b_only_stale": "YA",
        "selective/a_b_stale": "YA",
        "selective/decision_stale": "YA",
    }
    if case_id in stale_keys:
        stale_key = stale_keys[case_id]
        next(
            value
            for value in e
            if value.get("kind") == "revision"
            and value.get("collection") == "artifacts"
            and value.get("key") == stale_key
        )["value"] = min(cast(int, value["version"]) for value in multi[stale_key])

    if case_id == "lineage/two_selected_current_versions":
        seal = next(i for i, value in enumerate(e) if value.get("kind") == "seal_revision")
        current = next(value for value in e if value.get("kind") == "revision" and value.get("key") == "XA")
        e.insert(seal, {**current, "value": 2})
    elif case_id == "lineage/invented_version":
        next(value for value in e if value.get("kind") == "revision" and value.get("key") == "XA")["value"] = 3
    elif case_id == "lineage/declaration_crossover":
        second = next(value for value in e if value.get("kind") == "provenance" and value.get("key") == "BOUND:XA:2")
        obj(second["binding_artifact"])["declaration"] = "OTHER"
    elif case_id == "lineage/target_crossover":
        next(value for value in e if value.get("kind") == "artifact" and value.get("ref") == "XAv2")["target"] = "B"
    elif case_id == "lineage/invocation_crossover":
        next(value for value in e if value.get("kind") == "artifact" and value.get("ref") == "XAv2")["invocation"] = (
            "I1"
        )
    if case_id == "lineage/latest_older_selected":
        values = sorted(multi["XA"], key=lambda item: cast(int, item["version"]))
        older = cast(str, values[0]["ref"])
        older_key = f"BOUND:XA:{values[0]['version']}"
        selected = cast(str, values[-1]["ref"])
        for value in e:
            if value.get("kind") == "port" and value.get("port") == "context" and value.get("artifact") == selected:
                value["artifact"] = older
            if value.get("kind") == "input_producer" and value.get("port") == "context":
                value["producer"] = older_key
            if value.get("kind") == "provenance":
                value["parents"] = [
                    older_key if parent == f"BOUND:XA:{values[-1]['version']}" else parent
                    for parent in arr(value["parents"])
                ]
            if value.get("kind") == "assessment":
                value["consumed"] = {
                    port_name: older if artifact_ref == selected else artifact_ref
                    for port_name, artifact_ref in obj(value["consumed"]).items()
                }
    elif case_id == "lineage/latest_missing_binding_request":
        e[:] = [value for value in e if value.get("kind") != "binding_reserve"]
    elif case_id == "lineage/latest_foreign_binding_association":
        reserve = next(value for value in e if value.get("kind") == "binding_reserve")
        obj(reserve["association"])["declaration"] = "OTHER"
    elif case_id == "lineage/latest_missing_binding_settlement":
        e[:] = [value for value in e if value.get("kind") != "binding_settlement"]
    elif case_id == "lineage/latest_missing_binding_cleanup":
        e[:] = [value for value in e if value.get("kind") != "binding_cleanup"]
    elif case_id == "lineage/latest_foreign_cleanup_target":
        association = next(value for value in e if value.get("kind") == "binding_cleanup_association")
        association["targets"] = ["B"]
    elif case_id == "validity/final_candidate_sibling_substitution":
        values = sorted(multi["A"], key=lambda item: cast(int, item["version"]))
        next(value for value in e if value.get("kind") == "final")["candidate"] = values[0]["ref"]


def case(
    family: str, name: str, d: Obj, e: list[Obj], boundary: str = "qualification", alternates: Sequence[list[Obj]] = ()
) -> Obj:
    case_id = f"{family}/{name}"
    if case_id in VERSIONED_CASES:
        _materialize_initial_versions(case_id, d, e)
    c: Obj = {
        "boundary": boundary,
        "case_id": case_id,
        "declaration": d,
        "events": e,
        "expected": admit(d) if boundary == "admission" else reduce(d, e),
        "family": family,
        "traces": [],
    }
    neutral_only = {
        "admission/binding_foreign_node",
        "admission/duplicate_binding_declaration",
        "admission/foreign_dependency",
        "admission/initial_collection_without_binding",
        "assessment/consumed_port",
        "assessment/duplicate_coverage",
        "assessment/extra_coverage",
        "assessment/foreign_consumed",
        "assessment/foreign_subject",
        "assessment/foreign_target",
        "assessment/incomplete_coverage",
        "assessment/wrong_kind_coverage",
        "authentication/entry_target",
        "authentication/terminal_target",
        "joins/evidence_port_swap",
        "joins/subject_port_swap",
        "map_item_evidence/expansion_failed",
        "map_item_evidence/expansion_open",
        "map_item_evidence/expansion_overflow",
        "map_item_evidence/member_blocked_unreached",
        "map_item_evidence/two_maps_cross_owner",
        "map_item_evidence/wrong_subject_artifact",
        "membership/duplicate_member",
        "membership/foreign_target",
        "structural/node_kind_mismatch",
    }
    c["comparison_scope"] = "neutral_only" if case_id in neutral_only else "production_boundary"
    c["traces"] = [{"events": x, "expected": reduce(d, x), "name": f"alternate_{i}"} for i, x in enumerate(alternates)]
    return c


def mutate(events: list[Obj], kind: str, field: str, value: Json, index: int = 0) -> list[Obj]:
    x = deepcopy(events)
    matches = [e for e in x if e.get("kind") == kind]
    matches[index][field] = value
    return x
