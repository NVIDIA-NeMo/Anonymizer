# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent qualification_v1 reference: qualification."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import cast

from tests.graph_sdk.reference._qualification_v1.authentication import (
    authenticate,
    cleanup_codes,
    reconcile,
    request_codes,
    validity,
)
from tests.graph_sdk.reference._qualification_v1.declarations import (
    admit,
)
from tests.graph_sdk.reference._qualification_v1.model import (
    Obj,
    arr,
    obj,
    reject,
)
from tests.graph_sdk.reference._qualification_v1.provenance import (
    validate_provenance,
)
from tests.graph_sdk.reference._qualification_v1.records import (
    advance,
    initial,
)


def qualify(d: Obj, s: Obj) -> Obj:
    targets = [cast(str, x) for x in arr(d["targets"])]
    lim = obj(d["limits"])
    if s["execution"] != d["execution"]:
        return reject("foreign_owner")
    if s["revision_sealed"] is not True:
        return reject("missing")
    if len(obj(s["ports"])) > cast(int, lim["max_port_facts"]):
        return reject("limit_exceeded")
    if len(obj(s["artifacts"])) > cast(int, lim["max_artifacts"]) or sum(
        cast(int, obj(value)["bytes"]) for value in obj(s["artifacts"]).values()
    ) > cast(int, lim["max_artifact_bytes"]):
        return reject("limit_exceeded")
    if sum(len(arr(obj(x)["parents"])) for x in obj(s["provenance"]).values()) > cast(int, lim["max_provenance_edges"]):
        return reject("limit_exceeded")
    if sum(len(obj(obj(s["revision"])[k])) for k in ("artifacts", "configurations", "state")) > cast(
        int, lim["max_revision_entries"]
    ) or len(obj(obj(s["revision"])["absences"])) > cast(int, lim["max_absence_revisions"]):
        return reject("limit_exceeded")
    revision = obj(s["revision"])
    retained_pairs = {(cast(str, obj(value)["key"]), obj(value)["version"]) for value in obj(s["artifacts"]).values()}
    if any((key, obj(value).get("value")) not in retained_pairs for key, value in obj(revision["artifacts"]).items()):
        return reject("missing")
    admitted_absences = {
        cast(str, query)
        for production in map(obj, arr(d["productions"]))
        for query in arr(production["absence_queries"])
    }
    admitted_state = {
        cast(str, effect) for production in map(obj, arr(d["productions"])) for effect in arr(production["read_state"])
    }
    if set(obj(revision["absences"])) - admitted_absences:
        return reject("unsupported")
    if set(obj(revision["configurations"])) - set(obj(d["node_kinds"])):
        return reject("missing")
    if set(obj(revision["state"])) - admitted_state:
        return reject("unsupported")
    execution_only = d["purpose"] == "execution_only"
    if set(obj(s["input_producers"])) - set(obj(s["ports"])):
        return reject("contradictory")
    verified: list[Obj] = []
    if not execution_only:
        verified, bad = authenticate(d, s)
        if bad:
            return bad
    codes, bad = reconcile(d, s)
    if bad:
        return bad
    for source in (request_codes(d, s), cleanup_codes(d, s)):
        for t in targets:
            codes[t].update(source[t])
    finals = obj(s["finals"])
    if set(finals) - set(targets):
        return reject("foreign_owner")
    if any(obj(value).get("outcome") != "ok" for value in finals.values()):
        return reject("contradictory")
    bad = validate_provenance(d, s)
    if bad:
        return bad
    arts = obj(s["artifacts"])
    ports = obj(s["ports"])
    prov = obj(s["provenance"])
    entries = obj(s["entries"])
    memberships = obj(s["memberships"])
    valid = []
    for a in verified:
        a["validity"] = validity(a, s)
        valid.append(a)
    supporting: dict[str, list[Obj]] = {target: [] for target in targets}
    candidate_uncertain: set[str] = set()
    for t in targets:
        f = obj(finals.get(t, {}))
        if not f:
            codes[t].add("missing_candidate")
            continue
        candidate = cast(str, f["candidate"])
        p = obj(prov.get(cast(str, f["producer"]), {}))
        root_output = next((obj(x) for x in arr(d["root_outputs"]) if obj(x).get("target") == t), {})
        port_matches = [
            obj(x)
            for x in ports.values()
            if obj(x).get("activation") == f["activation"]
            and obj(x).get("artifact") == candidate
            and obj(x).get("node") == f["node"]
            and obj(x).get("role") == "candidate"
            and obj(x).get("target") == t
            and obj(x).get("port") == f["port"]
        ]
        if (
            candidate not in arts
            or obj(arts.get(candidate, {})).get("target") != t
            or obj(arts.get(candidate, {})).get("invocation") != obj(d["execution"])["invocation"]
        ):
            codes[t].add("missing_candidate")
        candidate_artifact = obj(arts.get(candidate, {}))
        candidate_key = candidate_artifact.get("key")
        candidate_version = candidate_artifact.get("version")
        candidate_revision = obj(obj(s["revision"])["artifacts"]).get(cast(str, candidate_key))
        candidate_current = (
            bool(candidate_artifact)
            and isinstance(candidate_revision, dict)
            and obj(candidate_revision).get("value") == candidate_version
        )
        if not candidate_current:
            candidate_uncertain.add(t)
            codes[t].add("missing_candidate")
        if not p:
            return reject("missing")
        direct_root = root_output.get("input_port") is not None
        if direct_root:
            expected_key = f"ROOT:{t}:{root_output['input_port']}"
            if f.get("producer") != expected_key or p.get("source") != "root_input":
                return reject("contradictory")
        elif (
            f.get("node") != root_output.get("source_node")
            or f.get("port") != root_output.get("port")
            or p.get("node") != root_output.get("source_node")
            or p.get("port") != root_output.get("source_port")
        ):
            return reject("contradictory")
        if p.get("artifact") != candidate or p.get("target") != t or (not direct_root and len(port_matches) != 1):
            return reject("contradictory")
        reqs = [obj(x) for x in arr(d["requirements"]) if obj(x)["target"] == t]
        for req in () if execution_only else reqs:
            eligible_productions = [
                p
                for p in map(obj, arr(d["productions"]))
                if p["meaning"] == req["meaning"]
                and p["outcome"] == req["outcome"]
                and p["subject_port"] == req["subject_port"]
                and set(cast(str, x) for x in arr(req["consumed_ports"])).issubset(
                    set(cast(str, x) for x in arr(p["consumed_ports"]))
                )
            ]
            eligible_promises = {cast(str, p["promise"]) for p in eligible_productions}
            input_subject_promises = {
                cast(str, p["promise"]) for p in eligible_productions if p["subject_source"] == "input"
            }

            def same_candidate_lineage(assessment_value: Obj) -> bool:
                subject = obj(arts.get(cast(str, assessment_value["subject_artifact"]), {}))
                final_artifact = obj(arts.get(candidate, {}))
                return subject.get("invocation") == final_artifact.get("invocation") and subject.get(
                    "key"
                ) == final_artifact.get("key")

            matches = [
                a
                for a in valid
                if a["target"] == t
                and (
                    a["promise"] in input_subject_promises
                    or a["subject_artifact"] == candidate
                    or same_candidate_lineage(a)
                )
                and a["promise"] in eligible_promises
                and a["meaning"] == req["meaning"]
                and a["outcome"] == req["outcome"]
                and a["subject_port"] == req["subject_port"]
                and set(cast(str, x) for x in arr(req["consumed_ports"])).issubset(set(obj(a["consumed"])))
            ]
            if not matches:
                codes[t].add("missing_assessment")
                continue
            complete = [
                a
                for a in matches
                if a["validity"] == "current"
                and a["finding"] == "satisfied"
                and set(cast(str, x) for x in arr(req["coverage"])).issubset(
                    set(cast(str, x) for x in arr(a["coverage"]))
                )
            ]
            if complete:
                for a in complete:
                    if a not in supporting[t]:
                        supporting[t].append(a)
                continue
            if any(a["validity"] == "stale" for a in matches):
                codes[t].add("stale_evidence")
            if any(a["validity"] == "unknown" or a["finding"] == "unknown" for a in matches):
                codes[t].add("assessment_unknown")
            if any(a["finding"] == "unsatisfied" for a in matches):
                codes[t].add("assessment_unsatisfied")
            if any(
                a["validity"] == "current"
                and a["finding"] == "satisfied"
                and not set(cast(str, x) for x in arr(req["coverage"])).issubset(
                    set(cast(str, x) for x in arr(a["coverage"]))
                )
                for a in matches
            ):
                codes[t].add("incomplete_coverage")

        def provenance_ancestry(key: str) -> set[str]:
            pending = [key]
            seen: set[str] = set()
            while pending:
                current = pending.pop()
                if current in seen:
                    continue
                seen.add(current)
                pending.extend(cast(str, parent) for parent in arr(obj(prov.get(current, {})).get("parents")))
            return seen

        final_ancestry = provenance_ancestry(cast(str, f.get("producer")))
        for typed_requirement in (
            requirement
            for requirement in map(obj, arr(d.get("map_item_requirements")))
            if requirement.get("target") == t
        ):
            endpoints = [
                obj(endpoint)
                for endpoint in [
                    typed_requirement.get("subject_endpoint"),
                    *arr(typed_requirement.get("consumed_endpoints")),
                ]
                if endpoint is not None
            ]
            endpoint = endpoints[0]
            if f.get("port") != typed_requirement.get("candidate_port"):
                codes[t].add("missing_candidate")
                continue
            expander_activations = [
                activation
                for activation, raw_entry in entries.items()
                if obj(raw_entry).get("node") == endpoint["expander"]
                and obj(raw_entry).get("state_outcome") == endpoint["expansion_outcome"]
                and obj(raw_entry).get("target") == t
            ]
            scoped_members: list[str] = []
            ancestry_complete = True
            for expander_activation in expander_activations:
                membership_producer = f"OP:{expander_activation}:{t}:{endpoint['membership_port']}"
                if membership_producer not in final_ancestry:
                    ancestry_complete = False
                membership = obj(memberships.get(expander_activation, {}))
                for member in (cast(str, value) for value in arr(membership.get("members"))):
                    member_entry = obj(entries.get(member, {}))
                    member_terminal = obj(obj(s["terminals"]).get(member, {}))
                    if (
                        member_entry.get("node") == endpoint["member"]
                        and member_entry.get("parent") == expander_activation
                        and member_entry.get("closed_unstarted") is False
                        and member_entry.get("state_category") == "success"
                        and member_terminal.get("category") == "success"
                        and member_terminal.get("outcome") == member_entry.get("state_outcome")
                        and member_terminal.get("target") == t
                    ):
                        scoped_members.append(member)
            if not ancestry_complete:
                codes[t].add("missing_assessment")
                continue
            for member in scoped_members:
                matches = [
                    assessment_value
                    for assessment_value in valid
                    if assessment_value["activation"] == member
                    and assessment_value["target"] == t
                    and assessment_value["promise"] == typed_requirement["promise"]
                    and assessment_value["meaning"] == typed_requirement["meaning"]
                ]
                if not matches:
                    codes[t].add("missing_assessment")
                    continue
                complete = [
                    assessment_value
                    for assessment_value in matches
                    if assessment_value["validity"] == "current"
                    and assessment_value["finding"] == "satisfied"
                    and set(cast(str, value) for value in arr(typed_requirement["coverage"])).issubset(
                        set(cast(str, value) for value in arr(assessment_value["coverage"]))
                    )
                ]
                if complete:
                    for assessment_value in complete:
                        if assessment_value not in supporting[t]:
                            supporting[t].append(assessment_value)
                    continue
                if any(assessment_value["validity"] == "stale" for assessment_value in matches):
                    codes[t].add("stale_evidence")
                if any(
                    assessment_value["validity"] == "unknown" or assessment_value["finding"] == "unknown"
                    for assessment_value in matches
                ):
                    codes[t].add("assessment_unknown")
                if any(assessment_value["finding"] == "unsatisfied" for assessment_value in matches):
                    codes[t].add("assessment_unsatisfied")
                if any(
                    assessment_value["validity"] == "current"
                    and assessment_value["finding"] == "satisfied"
                    and not set(cast(str, value) for value in arr(typed_requirement["coverage"])).issubset(
                        set(cast(str, value) for value in arr(assessment_value["coverage"]))
                    )
                    for assessment_value in matches
                ):
                    codes[t].add("incomplete_coverage")
    for _ in range(cast(int, lim["max_fixed_point_steps"])):
        before = sum(map(len, codes.values()))
        for raw in arr(d["dependencies"]):
            dep = obj(raw)
            a = cast(str, dep["prerequisite"])
            b = cast(str, dep["dependent"])
            if codes[a]:
                codes[b].add("dependency")
        for raw in arr(d["atomic"]):
            group = [cast(str, x) for x in arr(raw)]
            if len(group) > 1 and any(codes[x] for x in group):
                for x in group:
                    codes[x].add("atomic_group")
        if sum(map(len, codes.values())) == before:
            break

    def decisions(t: str) -> list[str]:
        stack = [cast(str, obj(finals[t])["producer"])]
        seen = set()
        out = set()
        for a in supporting[t]:
            for consumed_port, ref in obj(a["consumed"]).items():
                if obj(a["consumed_roles"]).get(consumed_port) == "decision":
                    out.add(cast(str, ref))
        while stack:
            key = stack.pop()
            if key in seen:
                continue
            seen.add(key)
            p = obj(prov.get(key, {}))
            if not p:
                return ["!missing"]
            if p["decision"] is True:
                ref = cast(str, p["artifact"])
                artifact_value = obj(arts.get(ref, {}))
                if (
                    artifact_value.get("target") not in (t, "shared")
                    or artifact_value.get("invocation") != obj(d["execution"])["invocation"]
                    or p.get("target") != t
                ):
                    return ["!invalid"]
                out.add(ref)
            stack.extend(cast(str, x) for x in arr(p["parents"]))
        return sorted(out)

    if not execution_only:
        possible_assessment_owners = {
            (activation, entry["target"], production["node"], production["outcome"], production["promise"])
            for activation, entry in map(lambda item: (item[0], obj(item[1])), obj(s["entries"]).items())
            for production in map(obj, arr(d["productions"]))
            if entry.get("closed_unstarted") is False
            and entry.get("node_kind") == "operation"
            and entry.get("state_category") == "success"
            and entry.get("node") == production["node"]
            and entry.get("state_outcome") == production["outcome"]
        }
        required_assessment_owners = {
            owner
            for owner in possible_assessment_owners
            for terminal_value in [obj(obj(s["terminals"]).get(cast(str, owner[0]), {}))]
            if terminal_value.get("structural") is False
            and terminal_value.get("category") == "success"
            and terminal_value.get("outcome") == owner[3]
            and terminal_value.get("target") == owner[1]
        }
        retained_assessment_owners = [
            (fact["activation"], fact["target"], fact["node"], fact["outcome"], fact["promise"])
            for fact in map(obj, obj(s["assessment_facts"]).values())
        ]
        actual_owner_set = set(retained_assessment_owners)
        if len(retained_assessment_owners) != len(actual_owner_set):
            return reject("duplicate")
        if any(cast(str, owner[0]) not in obj(s["entries"]) for owner in actual_owner_set):
            return reject("missing")
        if required_assessment_owners - actual_owner_set:
            return reject("missing")
        if actual_owner_set - possible_assessment_owners:
            return reject("unsupported")
    rows = []
    released = []
    union = set()
    for t in targets:
        f = obj(finals.get(t, {}))
        decs = decisions(t) if f and not execution_only else []
        if "!missing" in decs:
            return reject("missing")
        if "!invalid" in decs:
            return reject("foreign_owner")
        ok = not codes[t] and bool(f) and d["purpose"] == "protection"
        if ok:
            union.update(decs)
            released.append(
                {
                    "candidate": f["candidate"],
                    "evidence": sorted(cast(str, a["evidence_artifact"]) for a in supporting[t]),
                    "required_decisions": decs,
                    "target": t,
                }
            )
        q = (
            "not_assessed"
            if execution_only
            else "met"
            if ok
            else "unknown"
            if t in candidate_uncertain
            or codes[t] & {"assessment_unknown", "request_uncertain", "inconsistent_attribution"}
            else "unmet"
        )
        rows.append(
            {
                "artifact_available": bool(f)
                and cast(str, f.get("candidate")) in arts
                and (execution_only or t not in candidate_uncertain),
                "candidate": None if execution_only else f.get("candidate"),
                "completion": "pending" if "incomplete_membership" in codes[t] else "closed",
                "protection_available": ok,
                "qualification": q,
                "required_decisions": decs if ok else [],
                "target": t,
                "verified": sorted(cast(str, a["evidence_artifact"]) for a in supporting[t]),
                "withholding": sorted(
                    ({"request_accounting"} if "request_uncertain" in codes[t] else set())
                    | (codes[t] - {"request_uncertain"})
                    | ({"execution_only"} if execution_only else set())
                ),
            }
        )
    if len(union) > cast(int, lim["max_required_decisions"]):
        return reject("limit_exceeded")
    projected_memberships = deepcopy(obj(s["memberships"]))
    for membership_key, raw_membership in projected_memberships.items():
        membership = obj(raw_membership)
        if membership_key == "__ROOT__":
            membership["expansion_outcome"] = None
            continue
        parent_entry = obj(obj(s["entries"]).get(membership_key, {}))
        is_map = any(obj(map_input).get("expander") == parent_entry.get("node") for map_input in arr(d["map_inputs"]))
        if is_map and membership.get("status") not in ("failed", "overflow"):
            membership["expansion_outcome"] = parent_entry.get("state_outcome")
    record = {
        "artifacts": sorted(arts),
        "evidence": sorted(cast(str, a["evidence_artifact"]) for a in valid),
        "execution": s["execution"],
        "memberships": projected_memberships,
        "targets": rows,
        "terminals": s["terminals"],
    }
    return {
        "qualified": released,
        "record": record,
        "required_decisions": sorted(union),
        "status": "accepted",
        "targets": rows,
        "verified": valid,
    }


def reduce(d: Obj, events: Sequence[Obj]) -> Obj:
    a = admit(d)
    if a["status"] != "accepted":
        return a
    s = initial(d)
    for e in events:
        bad = advance(s, e)
        if bad:
            return bad
    return qualify(d, s)
