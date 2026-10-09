# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent qualification_v1 reference: authentication."""

from __future__ import annotations

from typing import cast

from tests.graph_sdk.reference._qualification_v1.model import (
    Json,
    Obj,
    arr,
    obj,
    reject,
)


def typed_endpoint_owner(d: Obj, s: Obj, activation: str, target: str, endpoint: Obj) -> tuple[str | None, str | None]:
    entries = obj(s["entries"])
    ports = obj(s["ports"])
    producers = obj(s["input_producers"])
    provenance_facts = obj(s["provenance"])
    memberships = obj(s["memberships"])
    member_entry = obj(entries.get(activation, {}))
    if not member_entry:
        return None, "missing"
    if member_entry.get("node") != endpoint.get("member") or member_entry.get("target") != target:
        return None, "foreign_owner"
    expander_activation = cast(str, member_entry.get("parent"))
    expander_entry = obj(entries.get(expander_activation, {}))
    if not expander_entry:
        return None, "missing"
    if (
        expander_entry.get("node") != endpoint.get("expander")
        or expander_entry.get("state_outcome") != endpoint.get("expansion_outcome")
        or expander_entry.get("target") != target
    ):
        return None, "foreign_owner"
    parent = expander_entry.get("parent")
    for expected_node in reversed([cast(str, value) for value in arr(endpoint.get("path"))]):
        wrapper = obj(entries.get(cast(str, parent), {}))
        if wrapper.get("node") != expected_node or wrapper.get("target") != target:
            return None, "foreign_owner"
        parent = wrapper.get("parent")
    if parent is not None:
        return None, "foreign_owner"
    item_port = obj(ports.get(f"{activation}|{endpoint['item_input']}", {}))
    input_fact = obj(producers.get(f"{activation}|{endpoint['item_input']}", {}))
    if not item_port or not input_fact:
        return None, "missing"
    producer_key = cast(str, input_fact.get("producer"))
    item_owner = obj(provenance_facts.get(producer_key, {}))
    membership_key = f"OP:{expander_activation}:{target}:{endpoint['membership_port']}"
    membership_owner = obj(provenance_facts.get(membership_key, {}))
    membership = obj(memberships.get(expander_activation, {}))
    artifact_value = obj(obj(s["artifacts"]).get(cast(str, item_port.get("artifact")), {}))
    if not item_owner or not membership_owner or not membership:
        return None, "missing"
    if artifact_value.get("target") != target or artifact_value.get("invocation") != obj(d["execution"])["invocation"]:
        return None, "foreign_owner"
    if item_owner.get("target") != target:
        return None, "foreign_owner"
    if item_owner.get("source") != "map_item":
        return None, "contradictory"
    if item_owner.get("member") not in entries or item_owner.get("expander") not in entries:
        return None, "missing"
    if (
        item_port.get("node") != endpoint.get("member")
        or item_port.get("role") != "artifact"
        or input_fact.get("node") != endpoint.get("member")
        or input_fact.get("target") != target
        or item_owner.get("artifact") != item_port.get("artifact")
        or item_owner.get("member") != activation
        or item_owner.get("expander") != expander_activation
        or item_owner.get("port") != endpoint.get("item_input")
        or arr(item_owner.get("parents")) != [membership_key]
        or membership_owner.get("activation") != expander_activation
        or membership_owner.get("node") != endpoint.get("expander")
        or membership_owner.get("port") != endpoint.get("membership_port")
        or membership_owner.get("target") != target
        or activation not in arr(membership.get("members"))
        or membership.get("target") != target
        or artifact_value.get("key") != cast(str, item_owner.get("artifact", "")).rsplit("v", 1)[0]
    ):
        return None, "contradictory"
    return producer_key, None


def authenticate(d: Obj, s: Obj) -> tuple[list[Obj], Obj | None]:
    lim = obj(d["limits"])
    raw = [cast(str, value) for value in arr(s["assessment_submissions"])]
    if len(raw) > cast(int, lim["max_submissions"]):
        return [], reject("limit_exceeded")
    ps = {(p["node"], p["outcome"], p["promise"]): p for p in map(obj, arr(d["productions"]))}
    seen_references: set[str] = set()
    seen_identities: set[tuple[Json, Json]] = set()
    verified = []
    facts = obj(s["assessment_facts"])
    for reference in raw:
        if reference in seen_references:
            return [], reject("duplicate")
        seen_references.add(reference)
        a = obj(facts.get(reference, {}))
        if not a:
            return [], reject("foreign_owner")
        identity = (a["activation"], a["promise"])
        if identity in seen_identities:
            return [], reject("duplicate")
        seen_identities.add(identity)
        p = ps.get((a["node"], a["outcome"], a["promise"]))
        if p is None:
            return [], reject("unsupported")
        if a["authenticated_factory"] != obj(d["execution"])["factory"]:
            return [], reject("foreign_owner")
        if a["evidence_port"] != p["evidence_port"] or a["subject_port"] != p["subject_port"]:
            return [], reject("contradictory")
        assessment_coverage = [cast(str, x) for x in arr(a["coverage"])]
        promise_coverage = [cast(str, x) for x in arr(p["coverage"])]
        if len(assessment_coverage) != len(set(assessment_coverage)):
            return [], reject("duplicate")
        if a["finding"] not in arr(p["findings"]) or set(assessment_coverage) - set(promise_coverage):
            return [], reject("unsupported")
        if set(assessment_coverage) != set(promise_coverage):
            return [], reject("contradictory")
        consumed = obj(a["consumed"])
        if len(consumed) > cast(int, lim["max_consumed_per_assessment"]):
            return [], reject("limit_exceeded")
        declared_consumed = set(cast(str, x) for x in arr(p["consumed_ports"]))
        if set(consumed) - declared_consumed:
            return [], reject("unsupported")
        if set(consumed) != declared_consumed:
            return [], reject("contradictory")
        target = cast(str, a["target"])
        activation = cast(str, a["activation"])
        ports = obj(s["ports"])
        arts = obj(s["artifacts"])
        entry_value = obj(obj(s["entries"]).get(activation, {}))
        terminal_value = obj(obj(s["terminals"]).get(activation, {}))
        typed_requirements = [
            requirement
            for requirement in map(obj, arr(d.get("map_item_requirements")))
            if requirement.get("target") == target
            and requirement.get("promise") == a["promise"]
            and requirement.get("meaning") == p["meaning"]
            and any(
                obj(endpoint).get("member") == p["node"]
                for endpoint in [requirement.get("subject_endpoint"), *arr(requirement.get("consumed_endpoints"))]
                if endpoint is not None
            )
        ]
        if entry_value and entry_value.get("node") not in obj(d["node_kinds"]):
            return [], reject("foreign_owner")
        if (
            not entry_value
            or entry_value.get("closed_unstarted") is True
            or entry_value.get("node_kind") != "operation"
            or entry_value.get("node") != p["node"]
            or entry_value.get("target") != target
        ):
            return [], reject("contradictory")
        if not terminal_value:
            return [], reject("missing")
        if (
            terminal_value.get("target") != target
            or terminal_value.get("structural") is not False
            or (
                terminal_value.get("category") not in ("blocked", "inconsistent")
                and terminal_value.get("attempt") is None
            )
        ):
            return [], reject("contradictory")
        if entry_value.get("state_category") != "success" or terminal_value.get("category") != "success":
            continue
        if entry_value.get("state_outcome") != p["outcome"] or terminal_value.get("outcome") != p["outcome"]:
            return [], reject("contradictory")
        ep = obj(ports.get(f"{activation}|{a['evidence_port']}", {}))
        sp = obj(ports.get(f"{activation}|{a['subject_port']}", {}))
        if not ep or not sp:
            return [], reject("missing")
        root_bound = any(
            obj(root).get("target") == target
            and obj(root).get("source_node") == p["node"]
            and obj(root).get("source_port") == a["evidence_port"]
            for root in arr(d["root_outputs"])
        )
        evidence_role = "candidate" if root_bound or a["evidence_port"] == a["subject_port"] else "evidence"
        if ep.get("target") not in (None, target) or sp.get("target") not in (None, target):
            return [], reject("foreign_owner")
        typed_subject = next(
            (
                obj(requirement["subject_endpoint"])
                for requirement in typed_requirements
                if requirement.get("subject_endpoint") is not None
            ),
            {},
        )
        subject_producer: str | None = None
        if typed_subject:
            subject_producer, endpoint_error = typed_endpoint_owner(d, s, activation, target, typed_subject)
            if endpoint_error:
                return [], reject(endpoint_error)
        if (
            ep.get("artifact") != a["evidence_artifact"]
            or ep.get("role") != evidence_role
            or ep.get("node") != p["node"]
            or ep.get("target") != target
            or sp.get("artifact") != a["subject_artifact"]
            or sp.get("role") != ("artifact" if typed_subject else "candidate")
            or sp.get("node") != p["node"]
            or sp.get("target") != target
        ):
            return [], reject("contradictory")
        consumed_roles: Obj = {}
        for consumed_port, ref in consumed.items():
            fact = obj(ports.get(f"{activation}|{consumed_port}", {}))
            if not fact:
                return [], reject("missing")
            if fact.get("artifact") != ref or fact.get("node") != p["node"] or fact.get("target") != target:
                return [], reject("contradictory")
            if fact.get("role") not in ("artifact", "candidate", "decision", "evidence"):
                return [], reject("unsupported")
            typed_consumed = next(
                (
                    obj(endpoint)
                    for requirement in typed_requirements
                    for endpoint in arr(requirement.get("consumed_endpoints"))
                    if obj(endpoint).get("item_input") == consumed_port
                ),
                {},
            )
            if typed_consumed:
                consumed_producer, endpoint_error = typed_endpoint_owner(d, s, activation, target, typed_consumed)
                if endpoint_error:
                    return [], reject(endpoint_error)
                if consumed_producer != obj(obj(s["input_producers"])[f"{activation}|{consumed_port}"])["producer"]:
                    return [], reject("contradictory")
            consumed_roles[consumed_port] = fact["role"]
        environment = obj(a["environment"])
        if set(cast(str, x) for x in arr(p["absence_queries"])) - set(obj(environment.get("absences", {}))):
            return [], reject("missing")
        if (
            set(obj(environment.get("absences", {}))) != set(cast(str, x) for x in arr(p["absence_queries"]))
            or obj(environment.get("configurations", {})) != {cast(str, p["node"]): p["configuration"]}
            or set(obj(environment.get("state", {}))) != set(cast(str, x) for x in arr(p["read_state"]))
        ):
            return [], reject("contradictory")
        for ref in [a["evidence_artifact"], a["subject_artifact"], *consumed.values()]:
            if ref not in arts:
                return [], reject("missing")
            if obj(arts[cast(str, ref)]).get("invocation") != obj(d["execution"])["invocation"]:
                return [], reject("foreign_owner")
            if obj(arts[cast(str, ref)]).get("target") != target:
                return [], reject("foreign_owner")
        verified_fact = dict(a)
        verified_fact["consumed_roles"] = consumed_roles
        verified_fact["meaning"] = p["meaning"]
        if typed_subject:
            verified_fact["subject"] = {"artifact": a["subject_artifact"], "producer": subject_producer}
        verified.append(verified_fact)
    if len(verified) > cast(int, lim["max_verified_evidence"]):
        return [], reject("limit_exceeded")
    target_ordinals = {cast(str, target): index for index, target in enumerate(arr(d["targets"]))}
    node_ordinals = {node: index for index, node in enumerate(sorted(obj(d["node_kinds"])))}
    port_ordinals = {
        (cast(str, dependency["node"]), cast(str, dependency["port"])): index
        for index, dependency in enumerate(map(obj, arr(d["output_dependencies"])))
    }
    artifact_ordinals = {artifact: index for index, artifact in enumerate(sorted(obj(s["artifacts"])))}

    def verified_key(fact: Obj) -> tuple[int, int, int, int, int]:
        target = cast(str, fact["target"])
        activation = cast(str, fact["activation"])
        node = cast(str, fact["node"])
        evidence_port = cast(str, fact["evidence_port"])
        evidence_artifact = cast(str, fact["evidence_artifact"])
        return (
            target_ordinals[target],
            cast(int, obj(obj(s["entries"])[activation])["occurrence"]),
            node_ordinals[node],
            port_ordinals[(node, evidence_port)],
            artifact_ordinals[evidence_artifact],
        )

    verified.sort(key=verified_key)
    return verified, None


def reconcile(d: Obj, s: Obj) -> tuple[dict[str, set[str]], Obj | None]:
    targets = [cast(str, x) for x in arr(d["targets"])]
    codes = {t: set() for t in targets}
    entries = obj(s["entries"])
    terms = obj(s["terminals"])
    reserv = obj(s["reservations"])
    members = obj(s["memberships"])
    if "__ROOT__" not in members or obj(members["__ROOT__"]).get("parent") is not None:
        return codes, reject("missing")
    owners: dict[str, str | None] = {}
    for membership_key, mraw in members.items():
        m = obj(mraw)
        parent = m.get("parent")
        t = m.get("target")
        root = membership_key == "__ROOT__"
        if root:
            if parent is not None or t is not None:
                return codes, reject("contradictory")
        elif parent != membership_key or t not in codes:
            return codes, reject("foreign_owner")
        if m.get("status") not in ("closed", "failed", "overflow", "open"):
            return codes, reject("invalid_value")
        if m.get("closed") is not (m.get("status") != "open"):
            return codes, reject("contradictory")
        if root:
            if m["closed"] is not True:
                for target in targets:
                    codes[target].add("incomplete_membership")
        elif parent not in entries or parent not in terms:
            codes[cast(str, t)].add("incomplete_membership")
        else:
            parent_entry = obj(entries[cast(str, parent)])
            parent_terminal = obj(terms[cast(str, parent)])
            if parent_entry.get("target") != t:
                return codes, reject("foreign_owner")
            expansion_outcomes = {
                obj(x).get("outcome")
                for x in arr(d["map_inputs"])
                if obj(x).get("expander") == parent_entry.get("node")
            }
            is_map_expander = bool(expansion_outcomes)
            is_subgraph = any(
                obj(x).get("node") == parent_entry.get("node")
                and obj(x).get("outcome") == parent_entry.get("state_outcome")
                for x in arr(d["subgraphs"])
            )
            expected_structural = parent_entry.get("node_kind") == "container"
            if (not is_map_expander and not is_subgraph) or parent_terminal.get(
                "structural"
            ) is not expected_structural:
                return codes, reject("contradictory")
            if (
                parent_terminal.get("category") != parent_entry.get("state_category")
                or parent_terminal.get("outcome") != parent_entry.get("state_outcome")
                or parent_terminal.get("target") != parent_entry.get("target")
            ):
                return codes, reject("contradictory")
            if (
                is_map_expander
                and m.get("status") == "closed"
                and parent_entry.get("state_outcome") not in expansion_outcomes
            ):
                codes[cast(str, t)].add("incomplete_membership")
            if m.get("status") in ("failed", "overflow"):
                codes[cast(str, t)].add("terminal_failure")
        if not root and m["closed"] is not True:
            codes[cast(str, t)].add("incomplete_membership")
        for child in arr(m["members"]):
            c = cast(str, child)
            if c in owners:
                return codes, reject("duplicate")
            owners[c] = None if parent is None else cast(str, parent)
            if c not in entries:
                if root:
                    return codes, reject("missing")
                codes[cast(str, t)].add("incomplete_membership")
                continue
            child_entry = obj(entries[c])
            if child_entry.get("parent") != parent:
                return codes, reject("contradictory")
            if not root and child_entry.get("target") != t:
                return codes, reject("foreign_owner")
    for activation, rraw in reserv.items():
        r = obj(rraw)
        if r.get("target") not in codes:
            return codes, reject("foreign_owner")
        selected = r["selected"] is True
        parent_key = "__ROOT__" if r.get("parent") is None else cast(str, r.get("parent"))
        parent_membership = obj(members.get(parent_key, {}))
        terminal_expansion = parent_membership.get("status") in ("failed", "overflow")
        if selected and activation not in entries and not terminal_expansion:
            codes[cast(str, r["target"])].add("incomplete_membership")
        if not selected and activation in entries:
            return codes, reject("contradictory")
        if selected and activation in entries and r["parent"] is not None and owners.get(activation) != r["parent"]:
            codes[cast(str, r["target"])].add("incomplete_membership")
    for activation, eraw in entries.items():
        e = obj(eraw)
        t = cast(str, e["target"])
        if t not in codes:
            return codes, reject("foreign_owner")
        if obj(d["node_kinds"]).get(cast(str, e.get("node"))) != e.get("node_kind"):
            return codes, reject("contradictory")
        if activation not in owners:
            return codes, reject("missing")
        if owners[activation] != e.get("parent"):
            return codes, reject("contradictory")
        if activation not in terms:
            codes[t].add("incomplete_membership")
        if activation in terms:
            fact = obj(terms[activation])
            expected_structural = e.get("node_kind") == "container"
            if fact.get("structural") is not expected_structural:
                return codes, reject("contradictory")
            if (
                fact.get("category") != e.get("state_category")
                or fact.get("outcome") != e.get("state_outcome")
                or fact.get("target") != e.get("target")
            ):
                return codes, reject("contradictory")
            if fact["category"] != "success":
                codes[t].add("terminal_failure")
    if any(activation not in entries for activation in terms):
        return codes, reject("missing")
    for join in map(obj, arr(d.get("keyed_joins"))):
        sources = [source for source in entries.values() if obj(source).get("node") == join["source"]]
        for source in map(obj, sources):
            matching = [
                candidate
                for candidate in entries.values()
                if obj(candidate).get("node") == join["join"]
                and obj(candidate).get("target") == source.get("target")
                and obj(candidate).get("parent") == source.get("parent")
            ]
            if len(matching) != 1:
                return codes, reject("missing" if not matching else "duplicate")
    return codes, None


def validity(a: Obj, s: Obj) -> str:
    rev = obj(s["revision"])
    changed = False
    missing = False
    deps = [a["evidence_artifact"], a["subject_artifact"], *obj(a["consumed"]).values()]
    for ref in deps:
        art = obj(obj(s["artifacts"])[cast(str, ref)])
        key = cast(str, art["key"])
        current = obj(rev["artifacts"])
        if key not in current:
            missing = True
        elif obj(current[key])["value"] != art["version"]:
            changed = True
    env = obj(a["environment"])
    for group in ("absences", "configurations", "state"):
        for key, value in obj(env[group]).items():
            current = obj(rev[group])
            if key not in current:
                missing = True
            elif obj(current[key])["value"] != value:
                changed = True
    return "stale" if changed else "unknown" if missing else "current"


def request_codes(d: Obj, s: Obj) -> dict[str, set[str]]:
    targets = [cast(str, x) for x in arr(d["targets"])]
    out = {t: set() for t in targets}
    requests = obj(s["requests"])
    latest: dict[str, str] = {}
    for rid, rraw in requests.items():
        r = obj(rraw)
        for t in arr(r["associations"]):
            target = cast(str, t)
            if target not in out:
                for owned in targets:
                    out[owned].add("inconsistent_attribution")
            else:
                latest[target] = rid
    allowed = {"retryable": "retry", "malformed": "correction", "permanent": "failover"}
    for rid, rraw in requests.items():
        r = obj(rraw)
        terminal = obj(r.get("terminal", {}))
        sett = obj(r.get("settlement", {}))
        assoc = [cast(str, x) for x in arr(r["associations"])]
        unresolved = (
            not terminal
            or not sett
            or sett.get("remote_stopped") is not True
            or sett.get("usage") == "unknown"
            or terminal.get("failure") in ("lost", "inconsistent")
        )
        if unresolved:
            for t in assoc:
                out[t].add("request_uncertain")
        elif terminal.get("condition") != "result" and any(latest.get(t) == rid for t in assoc):
            for t in assoc:
                out[t].add("request_accounting")
        elif terminal.get("condition") != "result":
            successor = [obj(x) for x in requests.values() if obj(x).get("predecessor") == rid]
            expected = allowed.get(cast(str, terminal.get("failure")))
            valid_successor = (
                len(successor) == 1
                and expected is not None
                and successor[0].get("purpose") == expected
                and successor[0].get("policy") == r.get("policy")
                and successor[0].get("associations") == r.get("associations")
            )
            if not valid_successor:
                for t in assoc:
                    out[t].add("request_accounting")
    return out


def cleanup_codes(d: Obj, s: Obj) -> dict[str, set[str]]:
    targets = [cast(str, x) for x in arr(d["targets"])]
    out = {t: set() for t in targets}
    ledgers = (
        (obj(s["cleanup_associations"]), obj(s["cleanups"]), False),
        (obj(s["binding_cleanup_associations"]), obj(s["binding_cleanups"]), True),
    )
    for assocs, cleanups, binding in ledgers:
        if set(assocs) != set(cleanups):
            for t in targets:
                out[t].add("inconsistent_attribution")
        for resource, craw in cleanups.items():
            c = obj(craw)
            a = obj(assocs.get(resource, {}))
            association_targets = [cast(str, target) for target in arr(a.get("targets"))]
            if (
                not a
                or (not association_targets and a.get("purpose") != "transport_only")
                or len(association_targets) != len(set(association_targets))
                or (binding and (a.get("owner") != "sdk" or a.get("purpose") != "accounting"))
            ):
                for t in targets:
                    out[t].add("inconsistent_attribution")
            elif any(cast(str, t) not in out for t in arr(a["targets"])):
                for t in targets:
                    out[t].add("inconsistent_attribution")
            elif (
                c["disposition"] not in ("closed",)
                and not (c["disposition"] == "left_open" and a["owner"] == "caller")
                and a["purpose"] != "transport_only"
            ):
                code = "cleanup_verification" if a["purpose"] == "verification" else "cleanup_accounting"
                for t in arr(a["targets"]):
                    out[cast(str, t)].add(code)
    return out
