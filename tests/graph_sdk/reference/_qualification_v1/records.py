# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent qualification_v1 reference: records."""

from __future__ import annotations

from typing import cast

from tests.graph_sdk.reference._qualification_v1.model import (
    REASON_CODES,
    TERMINAL_CATEGORIES,
    Obj,
    arr,
    obj,
    reject,
)


def initial(d: Obj) -> Obj:
    e = obj(d["execution"])
    return {
        "artifacts": {},
        "assessment_facts": {},
        "assessment_submissions": [],
        "binding_receipt": None,
        "binding_requests": {},
        "binding_cleanup_associations": {},
        "binding_cleanups": {},
        "cleanup_associations": {},
        "cleanups": {},
        "collection_values": {},
        "entries": {},
        "finals": {},
        "memberships": {},
        "input_producers": {},
        "ports": {},
        "provenance": {},
        "requests": {},
        "reservations": {},
        "revision": {"absences": {}, "artifacts": {}, "configurations": {}, "state": {}},
        "revision_sealed": False,
        "terminals": {},
        "execution": {"factory": e["factory"], "graph": e["graph"], "invocation": e["invocation"], "plan": e["plan"]},
    }


def add_unique(store: Obj, key: str, value: Obj) -> Obj | None:
    if key in store:
        return reject("duplicate")
    store[key] = value
    return None


def advance(s: Obj, e: Obj) -> Obj | None:
    k = e.get("kind")
    if k == "binding_receipt":
        if set(e) != {"artifacts", "binding", "kind", "sources", "terminal"}:
            return reject("invalid_value")
        if s["binding_receipt"] is not None:
            return reject("duplicate")
        if e.get("terminal") not in ("success", "partial"):
            return reject("contradictory")
        sources = [obj(source) for source in arr(e.get("sources"))]
        artifacts = [obj(artifact_value) for artifact_value in arr(e.get("artifacts"))]
        source_fields = {"declaration", "node", "port", "source", "target", "terminal"}
        artifact_fields = {"declaration", "key", "node", "port", "source", "target", "version"}
        if any(set(source) != source_fields or source.get("terminal") != "bound" for source in sources):
            return reject("invalid_value")
        if any(set(artifact_value) != artifact_fields for artifact_value in artifacts):
            return reject("invalid_value")
        references = [
            (artifact_value.get("declaration"), artifact_value.get("key"), artifact_value.get("version"))
            for artifact_value in artifacts
        ]
        if len(references) != len(set(references)):
            return reject("duplicate")
        if any(
            type(artifact_value.get("key")) is not int
            or cast(int, artifact_value["key"]) < 0
            or type(artifact_value.get("version")) is not int
            or cast(int, artifact_value["version"]) <= 0
            for artifact_value in artifacts
        ):
            return reject("invalid_value")
        s["binding_receipt"] = {
            "artifacts": artifacts,
            "binding": e["binding"],
            "sources": sources,
            "terminal": e["terminal"],
        }
        return None
    if k == "artifact":
        return add_unique(
            obj(s["artifacts"]),
            cast(str, e["ref"]),
            {x: e[x] for x in ("bytes", "invocation", "key", "target", "version")},
        )
    if k == "binding_reserve":
        if set(e) != {"association", "kind", "policy", "purpose", "request", "resource"}:
            return reject("invalid_value")
        if e.get("purpose") != "initial_binding" or e.get("policy") != "P0":
            return reject("contradictory")
        return add_unique(
            obj(s["binding_requests"]),
            cast(str, e["request"]),
            {
                "association": e["association"],
                "phase": "reserved",
                "policy": e["policy"],
                "purpose": e["purpose"],
                "resource": e["resource"],
                "settlement": None,
                "terminal": None,
            },
        )
    if k == "binding_dispatch":
        request = obj(obj(s["binding_requests"]).get(cast(str, e.get("request")), {}))
        if set(e) != {"kind", "request"} or request.get("phase") != "reserved":
            return reject("missing")
        request["phase"] = "dispatched"
        return None
    if k == "binding_result":
        request = obj(obj(s["binding_requests"]).get(cast(str, e.get("request")), {}))
        if set(e) != {"kind", "outcome", "request"} or request.get("phase") != "dispatched":
            return reject("missing")
        if e.get("outcome") != "retrieved" or request.get("terminal") is not None:
            return reject("contradictory")
        request["terminal"] = {"condition": "result", "outcome": e["outcome"]}
        request["phase"] = "terminal"
        return None
    if k == "binding_settlement":
        request = obj(obj(s["binding_requests"]).get(cast(str, e.get("request")), {}))
        if set(e) != {"kind", "remote_stopped", "request", "usage"} or request.get("phase") != "terminal":
            return reject("missing")
        if e.get("remote_stopped") is not True or e.get("usage") != "known":
            return reject("contradictory")
        request["settlement"] = {"remote_stopped": True, "usage": "known"}
        request["phase"] = "settled"
        return None
    if k == "port":
        return add_unique(
            obj(s["ports"]),
            f"{e['activation']}|{e['port']}",
            {x: e[x] for x in ("activation", "artifact", "node", "port", "role", "target")},
        )
    if k == "input_producer":
        if set(e) != {"activation", "kind", "node", "port", "producer", "target"}:
            return reject("invalid_value")
        return add_unique(
            obj(s["input_producers"]),
            f"{e['activation']}|{e['port']}",
            {x: e[x] for x in ("activation", "node", "port", "producer", "target")},
        )
    if k == "entry":
        if type(e.get("occurrence")) is not int or cast(int, e["occurrence"]) < 0:
            return reject("invalid_value")
        return add_unique(
            obj(s["entries"]),
            cast(str, e["activation"]),
            {
                x: e[x]
                for x in (
                    "closed_unstarted",
                    "node",
                    "node_kind",
                    "occurrence",
                    "parent",
                    "state_category",
                    "state_outcome",
                    "target",
                )
            },
        )
    if k == "terminal":
        if type(e.get("structural")) is not bool:
            return reject("invalid_type")
        category = e.get("category")
        attempt = e.get("attempt")
        reasons = arr(e.get("reasons"))
        if category not in TERMINAL_CATEGORIES or any(reason not in REASON_CODES for reason in reasons):
            return reject("invalid_value")
        if e["structural"] is True and attempt is not None:
            return reject("contradictory")
        if e["structural"] is False and category in ("success", "failure", "cancelled", "lost") and attempt is None:
            return reject("contradictory")
        if (category == "success" and reasons) or (category != "success" and not reasons):
            return reject("contradictory")
        return add_unique(
            obj(s["terminals"]),
            cast(str, e["activation"]),
            {x: e[x] for x in ("attempt", "category", "outcome", "reasons", "structural", "target")},
        )
    if k == "reservation":
        return add_unique(
            obj(s["reservations"]), cast(str, e["activation"]), {x: e[x] for x in ("parent", "selected", "target")}
        )
    if k == "membership":
        key = "__ROOT__" if e["parent"] is None else cast(str, e["parent"])
        return add_unique(
            obj(s["memberships"]),
            key,
            {x: e[x] for x in ("closed", "expansion_outcome", "members", "parent", "status", "target")},
        )
    if k == "result_owner":
        s["execution"] = {x: e[x] for x in ("factory", "graph", "invocation", "plan")}
        return None
    if k == "assessment":
        fields = (
            "activation",
            "authenticated_factory",
            "consumed",
            "coverage",
            "environment",
            "evidence_artifact",
            "evidence_port",
            "finding",
            "node",
            "outcome",
            "promise",
            "subject_artifact",
            "subject_port",
            "target",
        )
        return add_unique(obj(s["assessment_facts"]), cast(str, e["fact"]), {x: e[x] for x in fields})
    if k == "assessment_submission":
        if set(e) != {"fact", "kind"} or not isinstance(e.get("fact"), str):
            return reject("invalid_type")
        arr(s["assessment_submissions"]).append(cast(str, e["fact"]))
        return None
    if k == "final":
        return add_unique(
            obj(s["finals"]),
            cast(str, e["target"]),
            {x: e[x] for x in ("activation", "candidate", "node", "outcome", "port", "producer")},
        )
    if k == "provenance":
        provenance_fields = {
            "activation",
            "artifact",
            "binding_artifact",
            "declaration",
            "decision",
            "expander",
            "item_key",
            "item_version",
            "key",
            "kind",
            "member",
            "node",
            "parents",
            "port",
            "root_target",
            "source",
            "target",
        }
        if set(e) != provenance_fields:
            return reject("invalid_value")
        parent_keys = [cast(str, parent) for parent in arr(e["parents"])]
        if len(parent_keys) != len(set(parent_keys)):
            return reject("duplicate")
        prov = obj(s["provenance"])
        if any(parent not in prov for parent in arr(e["parents"])):
            return reject("missing")
        return add_unique(
            prov,
            cast(str, e["key"]),
            {
                x: e[x]
                for x in (
                    "activation",
                    "artifact",
                    "binding_artifact",
                    "declaration",
                    "decision",
                    "expander",
                    "item_key",
                    "item_version",
                    "node",
                    "member",
                    "parents",
                    "port",
                    "root_target",
                    "source",
                    "target",
                )
            },
        )
    if k == "collection_value":
        if set(e) != {"artifact", "items", "kind", "producer", "target"}:
            return reject("invalid_value")
        items = arr(e.get("items"))
        if any(
            set(obj(item)) != {"key", "version"}
            or type(obj(item).get("key")) is not int
            or cast(int, obj(item)["key"]) < 0
            or type(obj(item).get("version")) is not int
            or cast(int, obj(item)["version"]) <= 0
            for item in items
        ):
            return reject("invalid_value")
        identities = [(obj(item)["key"], obj(item)["version"]) for item in items]
        if identities != sorted(identities) or len(identities) != len(set(identities)):
            return reject("contradictory")
        return add_unique(
            obj(s["collection_values"]),
            cast(str, e["producer"]),
            {"artifact": e["artifact"], "items": items, "target": e["target"]},
        )
    if k == "revision":
        if s["revision_sealed"] is True:
            return reject("contradictory")
        group = cast(str, e["collection"])
        store = obj(obj(s["revision"])[group])
        return add_unique(store, cast(str, e["key"]), {"value": e["value"]})
    if k == "seal_revision":
        s["revision_sealed"] = True
        return None
    if k == "request_attempt":
        associations = arr(e["associations"])
        if len(associations) != len(set(cast(str, item) for item in associations)):
            return reject("duplicate")
        return add_unique(
            obj(s["requests"]),
            cast(str, e["request"]),
            {
                "associations": sorted(cast(str, item) for item in associations),
                "policy": e["policy"],
                "predecessor": e["predecessor"],
                "purpose": e["purpose"],
                "settlement": None,
                "terminal": None,
            },
        )
    if k == "request_terminal":
        r = obj(obj(s["requests"]).get(cast(str, e["request"]), {}))
        if not r:
            return reject("missing")
        if r["terminal"] is not None:
            return reject("duplicate")
        r["terminal"] = {"condition": e["condition"], "failure": e["failure"]}
        return None
    if k == "settlement":
        r = obj(obj(s["requests"]).get(cast(str, e["request"]), {}))
        if not r:
            return reject("missing")
        if r["settlement"] is not None:
            return reject("duplicate")
        r["settlement"] = {"remote_stopped": e["remote_stopped"], "usage": e["usage"]}
        return None
    if k == "cleanup_association":
        return add_unique(
            obj(s["cleanup_associations"]), cast(str, e["resource"]), {x: e[x] for x in ("owner", "purpose", "targets")}
        )
    if k == "cleanup":
        return add_unique(obj(s["cleanups"]), cast(str, e["resource"]), {"disposition": e["disposition"]})
    if k == "binding_cleanup_association":
        return add_unique(
            obj(s["binding_cleanup_associations"]),
            cast(str, e["resource"]),
            {x: e[x] for x in ("owner", "purpose", "targets")},
        )
    if k == "binding_cleanup":
        return add_unique(obj(s["binding_cleanups"]), cast(str, e["resource"]), {"disposition": e["disposition"]})
    return reject("unknown_event")
