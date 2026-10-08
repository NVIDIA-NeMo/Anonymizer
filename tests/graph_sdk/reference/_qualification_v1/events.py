# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent qualification_v1 reference: events."""

from __future__ import annotations

from collections.abc import Sequence

from tests.graph_sdk.reference._qualification_v1.declarations import (
    declaration,
    set_output_dependency,
)
from tests.graph_sdk.reference._qualification_v1.model import (
    Json,
    Obj,
    arr,
    obj,
)


def artifact(ref: str, target: str, role: str) -> Obj:
    return {
        "bytes": 1,
        "invocation": "I0",
        "key": ref.rsplit("v", 1)[0],
        "kind": "artifact",
        "ref": ref,
        "role": role,
        "target": target,
        "version": int(ref.rsplit("v", 1)[1]),
    }


def entry(
    a: str,
    t: str,
    node: str = "N",
    closed: bool = False,
    *,
    node_kind: str = "operation",
    occurrence: int | None = None,
    parent: str | None = None,
    state_category: str = "success",
    state_outcome: str | None = "ok",
) -> Obj:
    suffix = a.rstrip("0123456789")
    derived_occurrence = int(a[len(suffix) :]) if len(suffix) < len(a) else 0
    return {
        "activation": a,
        "closed_unstarted": closed,
        "kind": "entry",
        "node": node,
        "node_kind": node_kind,
        "occurrence": derived_occurrence if occurrence is None else occurrence,
        "parent": parent,
        "state_category": state_category,
        "state_outcome": state_outcome,
        "target": t,
    }


def terminal(
    a: str,
    t: str,
    category: str = "success",
    outcome: str | None = "ok",
    *,
    structural: bool = False,
    attempt: str | None = None,
    reasons: Sequence[str] = (),
) -> Obj:
    actual_attempt = None if structural or category in ("blocked", "inconsistent") else attempt or f"TASK:{a}"
    default_reason = {
        "blocked": "prerequisite",
        "cancelled": "cancel_requested",
        "failure": "execution_failed",
        "inconsistent": "contradictory",
        "lost": "transport_lost",
    }
    actual_reasons = list(reasons) if reasons else ([] if category == "success" else [default_reason[category]])
    return {
        "activation": a,
        "attempt": actual_attempt,
        "category": category,
        "kind": "terminal",
        "outcome": outcome,
        "reasons": actual_reasons,
        "structural": structural,
        "target": t,
    }


def port(a: str, t: str, name: str, ref: str, role: str, node: str = "N") -> Obj:
    return {"activation": a, "artifact": ref, "kind": "port", "node": node, "port": name, "role": role, "target": t}


def input_producer(a: str, t: str, node: str, port_name: str, producer: str) -> Obj:
    return {
        "activation": a,
        "kind": "input_producer",
        "node": node,
        "port": port_name,
        "producer": producer,
        "target": t,
    }


def binding_ref(declaration: str, key: int = 0, version: int = 1) -> Obj:
    return {"declaration": declaration, "key": key, "version": version}


def binding_declaration(
    *sites: tuple[str, str, str, str],
    collections: Sequence[str] = (),
    targets: Sequence[str] = ("A",),
) -> Obj:
    d = declaration(targets=targets)
    d["binding_inputs"] = [
        {
            "declaration": identity,
            "materialization": "collection" if identity in collections else "single",
            "node": node,
            "port": port_name,
            "source": f"SRC:{identity}@1",
            "target": target,
            "version_selection": "exact_one",
        }
        for identity, node, port_name, target in sites
    ]
    d["initial_collections"] = [
        {field: obj(site)[field] for field in ("declaration", "node", "port", "target")}
        for site in arr(d["binding_inputs"])
        if obj(site).get("declaration") in collections
    ]
    if sites:
        set_output_dependency(
            d,
            "N",
            "result",
            (*tuple(f"bound_{i}" for i, _ in enumerate(sites)), "subject"),
            identity_input="subject",
        )
    return d


def binding_receipt(
    sites: Sequence[tuple[str, str, str, str]],
    artifacts: Sequence[tuple[str, int, int]],
) -> Obj:
    by_identity = {identity: (node, port_name, target) for identity, node, port_name, target in sites}
    return {
        "artifacts": [
            {
                "declaration": identity,
                "key": key,
                "node": by_identity[identity][0],
                "port": by_identity[identity][1],
                "source": f"SRC:{identity}@1",
                "target": by_identity[identity][2],
                "version": version,
            }
            for identity, key, version in artifacts
        ],
        "binding": "BIND0",
        "kind": "binding_receipt",
        "sources": [
            {
                "declaration": identity,
                "node": node,
                "port": port_name,
                "source": f"SRC:{identity}@1",
                "target": target,
                "terminal": "bound",
            }
            for identity, node, port_name, target in sites
        ],
        "terminal": "success",
    }


def provenance(
    key: str,
    ref: str,
    t: str,
    parents: Sequence[str] = (),
    decision: bool = False,
    *,
    source: str = "operation_output",
    node: str | None = "N",
    port_name: str | None = "result",
    root_target: str | None = None,
    activation: str | None = None,
    binding_artifact: Obj | str | None = None,
    declaration: str | None = None,
    expander: str | None = None,
    item_key: int | None = None,
    item_version: int | None = None,
    member: str | None = None,
) -> Obj:
    return {
        "artifact": ref,
        "activation": activation,
        "binding_artifact": binding_artifact,
        "declaration": declaration,
        "decision": decision,
        "expander": expander,
        "item_key": item_key,
        "item_version": item_version,
        "key": key,
        "kind": "provenance",
        "node": node,
        "member": member,
        "parents": list(parents),
        "port": port_name,
        "root_target": root_target,
        "source": source,
        "target": t,
    }


def assessment(t: str, **changes: Json) -> Obj:
    x: Obj = {
        "activation": f"ROOT:{t}",
        "authenticated_factory": "EXEC:I0",
        "consumed": {"context": f"X{t}v0"},
        "coverage": ["K0", "K1"],
        "environment": {"absences": {"Q0": 1}, "configurations": {"N": "c0"}, "state": {"read": 1}},
        "evidence_artifact": f"E{t}v0",
        "evidence_port": "evidence",
        "finding": "satisfied",
        "kind": "assessment",
        "node": "N",
        "outcome": "ok",
        "promise": "P",
        "subject_artifact": f"{t}v0",
        "subject_port": "subject",
        "target": t,
    }
    x.update(changes)
    x["fact"] = x.get("fact", f"F:{t}:{x['promise']}")
    return x


def assessment_submission(t: str, promise: str = "P") -> Obj:
    return {"fact": f"F:{t}:{promise}", "kind": "assessment_submission"}


def final(t: str, **changes: Json) -> Obj:
    x: Obj = {
        "activation": f"ROOT:{t}",
        "candidate": f"{t}v0",
        "kind": "final",
        "node": "N",
        "outcome": "ok",
        "port": "result",
        "producer": f"OUT:{t}",
        "target": t,
    }
    x.update(changes)
    return x


def revisions(targets: Sequence[str], **changes: int) -> list[Obj]:
    out: list[Obj] = []
    for t in targets:
        for key, value in ((t, 0), (f"E{t}", 0), (f"X{t}", 0)):
            out.append({"collection": "artifacts", "key": key, "kind": "revision", "value": changes.get(key, value)})
    out += [
        {"collection": "absences", "key": "Q0", "kind": "revision", "value": changes.get("Q0", 1)},
        {"collection": "configurations", "key": "N", "kind": "revision", "value": changes.get("N", "c0")},
        {"collection": "state", "key": "read", "kind": "revision", "value": changes.get("read", 1)},
        {"kind": "seal_revision"},
    ]
    return out


def base_events(targets: Sequence[str], *, assess: bool = True) -> list[Obj]:
    e = []
    for t in targets:
        root = f"ROOT:{t}"
        e.extend(
            [
                artifact(f"{t}v0", t, "candidate"),
                artifact(f"E{t}v0", t, "evidence"),
                artifact(f"X{t}v0", t, "artifact"),
                entry(root, t),
                terminal(root, t),
                port(root, t, "subject", f"{t}v0", "candidate"),
                port(root, t, "evidence", f"E{t}v0", "evidence"),
                port(root, t, "context", f"X{t}v0", "artifact"),
                port(root, t, "result", f"{t}v0", "candidate"),
                provenance(
                    f"ROOT:{t}:subject",
                    f"{t}v0",
                    t,
                    source="root_input",
                    port_name="subject",
                    activation=root,
                ),
                provenance(
                    f"ROOT:{t}:context",
                    f"X{t}v0",
                    t,
                    source="root_input",
                    port_name="context",
                    activation=root,
                ),
                input_producer(root, t, "N", "subject", f"ROOT:{t}:subject"),
                input_producer(root, t, "N", "context", f"ROOT:{t}:context"),
                provenance(
                    f"OUT:{t}",
                    f"{t}v0",
                    t,
                    (f"ROOT:{t}:subject", f"ROOT:{t}:context"),
                    activation=root,
                ),
                provenance(
                    f"EVID:{t}",
                    f"E{t}v0",
                    t,
                    (f"ROOT:{t}:context",),
                    port_name="evidence",
                    activation=root,
                ),
                final(t),
            ]
        )
        if assess:
            e.append(assessment(t))
            e.append(assessment_submission(t))
    root_membership: Obj = {
        "closed": True,
        "expansion_outcome": None,
        "kind": "membership",
        "members": [f"ROOT:{t}" for t in targets],
        "parent": None,
        "status": "closed",
        "target": None,
    }
    e.append(root_membership)
    e.extend(revisions(targets))
    return e
