# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent effects_v1 reference: reducer."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import cast

from tests.graph_sdk.reference._effects_v1.builders import (
    _materialization_category,
)
from tests.graph_sdk.reference._effects_v1.materialization import (
    _materialization_declaration,
    _materialize,
    _natural,
    _publish_operation,
    _requires_materialization,
    _source_followup_available,
)
from tests.graph_sdk.reference._effects_v1.model import (
    FAILURE_CLASSES,
    RUNTIME_CONDITIONS,
    Json,
    Object,
    _array,
    _object,
    _strings,
)
from tests.graph_sdk.reference._effects_v1.request_facts import (
    _apply_embedded_settlement,
    _initial,
    _record_binding_success,
    _record_late_terminal_conflict,
    _record_request_failure,
    _reject,
    _remove,
    _request_fact,
    _terminal,
    _unique,
    _valid_settlement,
    _valid_usage,
)


def _advance(state: Object, declaration: Object, event: Object) -> Object | None:
    kind = cast(str, event.get("kind"))
    reservations = _object(state["reservations"])
    dispatched = _strings(state["dispatched"])
    bindings = _object(state["bindings"])
    attempts = _object(state["attempts"])
    if state.get("binding_finished") is True:
        initial_associations = {
            cast(str, spec["association"])
            for spec in (_object(raw) for raw in _array(declaration.get("materializations", [])))
            if spec.get("path") == "initial"
        }
        request = cast(str, event.get("request", ""))
        request_associations = set(_strings(_object(state["request_associations"]).get(request, [])))
        reserved_associations = set(_strings(reservations.get(request, [])))
        event_associations = set(_strings(event.get("associations", [])))
        event_association = event.get("association")
        mutates_binding = (
            kind == "bind_policy"
            and event_association in initial_associations
            or kind == "reserve"
            and bool(event_associations & initial_associations)
            or kind == "dispatch"
            and bool(request_associations & initial_associations or reserved_associations & initial_associations)
            or kind in {"result", "failure", "cancel", "stop", "lost", "settlement"}
            and bool(request_associations & initial_associations or reserved_associations & initial_associations)
            or kind in {"source_result", "source_failure", "materialize_result"}
            and bool(
                event_association in initial_associations
                or request_associations & initial_associations
                or reserved_associations & initial_associations
            )
        )
        if mutates_binding:
            return _reject("contradictory")
    if kind == "bind_policy":
        association = cast(str, event["association"])
        if association in bindings:
            return _reject("duplicate_binding")
        bindings[association] = event["policies"]
    elif kind == "reserve":
        request, policy = cast(str, event["request"]), cast(str, event["policy"])
        associations = _strings(event["associations"])
        if request in reservations or request in dispatched:
            return _reject("duplicate_request")
        if policy not in _object(state["policies"]):
            return _reject("unknown_policy")
        if any(policy not in _strings(bindings.get(item, [])) for item in associations):
            return _reject("cross_policy")
        maximum = cast(int, _object(_object(state["policies"])[policy])["max_attempts"])
        if event.get("purpose") in ("retry", "correction", "failover"):
            replay = _object(_object(state["policies"])[policy])["replay"]
            for item in associations:
                request_id = cast(str, _object(state["association_requests"]).get(item))
                facts = _object(state["request_facts"])
                if request_id not in facts:
                    return _reject("missing_predecessor")
                predecessor = _object(facts[request_id])
                if _object(state["request_policies"])[request_id] != policy:
                    return _reject("predecessor_policy")
                failure = predecessor.get("failure")
                purpose = event.get("purpose")
                if purpose == "retry":
                    if failure not in ("rejected_before_acceptance", "retryable", "transport_unknown"):
                        return _reject("invalid_retry")
                    permitted = failure == "rejected_before_acceptance" and replay in (
                        "before_acceptance",
                        "idempotent",
                    )
                    permitted = permitted or failure in ("retryable", "transport_unknown") and replay == "idempotent"
                elif purpose == "correction":
                    if failure != "malformed_response":
                        return _reject("invalid_correction")
                    permitted = failure == "malformed_response" and replay == "idempotent"
                else:
                    policy_value = _object(_object(state["policies"])[policy])
                    if failure not in _strings(policy_value.get("failover_failures", [])):
                        return _reject("invalid_failover")
                    permitted = replay == "idempotent" or (
                        failure == "rejected_before_acceptance" and replay == "before_acceptance"
                    )
                if not permitted:
                    return _reject("replay_forbidden")
        eligible: list[str] = []
        for item in associations:
            if cast(int, attempts.get(item, 0)) >= maximum:
                _object(state["denials"])[item] = "request_limit_stopped"
            else:
                eligible.append(item)
        limit = declaration.get("hard_limit")
        if (
            eligible
            and limit is not None
            and cast(int, state["dispatched_count"]) + len(reservations) >= cast(int, limit)
        ):
            for item in eligible:
                _object(state["denials"])[item] = "budget_stopped"
            eligible = []
        if eligible:
            reservations[request] = eligible
            _object(state["reservation_policies"])[request] = policy
    elif kind == "dispatch":
        request = cast(str, event["request"])
        if request not in reservations:
            return _reject("missing_reservation")
        associations = _strings(reservations.pop(request))
        _object(state["request_associations"])[request] = associations
        policy = cast(str, _object(state["reservation_policies"]).pop(request))
        _object(state["request_policies"])[request] = policy
        dispatched.append(request)
        state["dispatched"] = sorted(set(dispatched))
        state["dispatched_count"] = cast(int, state["dispatched_count"]) + 1
        for item in associations:
            attempts[item] = cast(int, attempts.get(item, 0)) + 1
            _object(state["association_requests"])[item] = request
            if _object(state["tasks"]).get(item) == "running":
                _object(state["task_requests"])[item] = request
        for key in ("local_in_flight", "remote_outstanding"):
            values = _strings(state[key])
            _unique(cast(list[Json], values), request)
            state[key] = values
    elif kind == "result":
        request = cast(str, event["request"])
        if request not in dispatched:
            return _reject("missing")
        expected = set(_strings(_object(state["request_associations"])[request]))
        if any(_requires_materialization(declaration, "adaptive", association) for association in expected):
            return _reject("materialization_required")
        returned = _strings(event["returned"])
        outcomes_value = event.get("outcomes")
        if not isinstance(outcomes_value, dict) or any(not isinstance(value, str) for value in outcomes_value.values()):
            return _reject("invalid_result")
        outcomes = cast(Object, outcomes_value)
        seen: set[str] = set()
        defects: list[str] = []
        for item in returned:
            if item in seen:
                defects.append("duplicate_keyed_result")
            elif item not in expected:
                defects.append("foreign_keyed_result" if item.startswith("X") else "extra_keyed_result")
            seen.add(item)
        if expected - seen:
            defects.append("missing_keyed_result")
        if not defects and set(outcomes) != expected:
            defects.append("missing_keyed_result" if expected - set(outcomes) else "extra_keyed_result")
        for defect in defects:
            _unique(_array(state["defects"]), defect)
        terminal = "inconsistent" if defects else "success"
        first_terminal = request not in _object(state["terminals"])
        _terminal(state, request, terminal)
        if first_terminal:
            if defects:
                _request_fact(state, request, {"condition": "request_inconsistent"})
            else:
                _request_fact(state, request, {"condition": "result", "outcomes": outcomes})
        _remove(state, "local_in_flight", request)
        if first_terminal:
            _remove(state, "remote_outstanding", request)
    elif kind == "failure":
        request = cast(str, event["request"])
        if request not in dispatched:
            return _reject("not_dispatched")
        _record_request_failure(state, request, cast(str, event["failure"]))
    elif kind == "cancel":
        request = cast(str, event["request"])
        if request in _object(state["terminals"]):
            pass
        elif request in reservations:
            reservations.pop(request)
            _object(state["reservation_policies"]).pop(request)
            _terminal(state, request, "cancelled")
        elif request in dispatched:
            values = _strings(state["cancel_requested"])
            _unique(cast(list[Json], values), request)
            state["cancel_requested"] = values
        else:
            return _reject("unknown_request")
    elif kind == "scope_cancel":
        for request in tuple(reservations):
            reservations.pop(request)
            _object(state["reservation_policies"]).pop(request)
            _terminal(state, request, "cancelled")
        state["cancel_requested"] = sorted(set(_strings(state["cancel_requested"])) | set(dispatched))
    elif kind == "stop":
        request = cast(str, event["request"])
        if "usage" not in event or not _valid_usage(event["usage"]):
            return _reject("invalid_usage")
        if request not in _strings(state["cancel_requested"]):
            return _reject("cancel_not_requested")
        _terminal(state, request, "cancelled")
        _request_fact(state, request, {"condition": "cancel_after_dispatch"})
        _remove(state, "local_in_flight", request)
        _remove(state, "remote_outstanding", request)
    elif kind == "lost":
        request = cast(str, event["request"])
        if request not in dispatched:
            return _reject("not_dispatched")
        _terminal(state, request, "lost")
        _request_fact(state, request, {"condition": "lost"})
        _remove(state, "local_in_flight", request)
    elif kind == "settlement":
        request = cast(str, event["request"])
        settlements = _object(state["settlements"])
        if request not in dispatched:
            return _reject("not_dispatched")
        disposition = event.get("disposition")
        remote_stopped = event.get("remote_stopped")
        if disposition not in ("completed", "rejected", "stopped", "unknown"):
            return _reject("invalid_settlement")
        if not _valid_usage(event.get("usage")):
            return _reject("invalid_usage")
        if remote_stopped is not None and not isinstance(remote_stopped, bool):
            return _reject("invalid_settlement")
        if (disposition == "unknown") != (remote_stopped is not True):
            return _reject("invalid_settlement")
        value: Object = {key: event[key] for key in ("disposition", "usage", "remote_stopped")}
        if request in settlements and settlements[request] != value:
            _unique(_array(state["defects"]), "conflicting_settlement")
        else:
            settlements[request] = value
            if event["remote_stopped"] is True:
                _remove(state, "remote_outstanding", request)
    elif kind == "source_item_constructor":
        if any(not isinstance(_object(item).get("value"), str) for item in _array(event["items"])):
            return _reject("invalid_type")
    elif kind == "source_failure_constructor":
        # Required Python keywords and typed value invariants precede response acceptance.
        if "failure" not in event or "settlement" not in event:
            return {"status": "rejected", "exception": "TypeError"}
        if event.get("disposition") == "omitted_optional" and event.get("failure") != "permanent":
            return _reject("contradictory")
    elif kind == "binding_result":
        # AssociationResult construction precedes any request reducer event.
        if (
            event.get("outcome") != "retrieved"
            or event.get("outputs") != []
            or event.get("consumed_context_ports") != []
        ):
            return _reject("contradictory")
    elif kind == "source_result":
        request = cast(str, event.get("request"))
        if request not in dispatched or request not in _object(state["request_associations"]):
            return _reject("unsolicited_source")
        expected = _strings(_object(state["request_associations"])[request])
        if len(expected) != 1:
            return _reject("binding_request_shape")
        identity, source = expected[0], cast(str, event["source"])
        if _requires_materialization(declaration, "initial", identity):
            return _reject("materialization_required")
        if not _valid_settlement(event.get("settlement"), optional=False):
            return _reject("invalid_settlement")
        was_terminal = request in _object(state["terminals"])
        if _object(state["binding_declarations"]).get(identity) != source:
            _record_request_failure(state, request, "malformed_response")
            _apply_embedded_settlement(state, request, event["settlement"])
            if not was_terminal and not _source_followup_available(
                state, declaration, identity, request, "malformed_response"
            ):
                _object(state["binding_sources"])[identity] = "failed"
                requirements = _object(declaration.get("binding_requirements", {}))
                state["binding_terminal"] = "partial" if requirements.get(identity) == "optional" else "failed"
            return None
        if (
            event.get("outcome") != "retrieved"
            or event.get("outputs") != []
            or event.get("consumed_context_ports") != []
            or not isinstance(event.get("items"), list)
        ):
            _record_request_failure(state, request, "malformed_response")
            _apply_embedded_settlement(state, request, event["settlement"])
            if not was_terminal and not _source_followup_available(
                state, declaration, identity, request, "malformed_response"
            ):
                defects = state.setdefault("binding_defects", [])
                _unique(_array(defects), "malformed_source_result")
                requirements = _object(declaration.get("binding_requirements", {}))
                state["binding_terminal"] = "partial" if requirements.get(identity) == "optional" else "failed"
            return None
        items = [_object(item) for item in _array(event["items"]) if isinstance(item, dict)]
        returned = [cast(str, item.get("association")) for item in items]
        defects: list[str] = []
        if len(items) != len(_array(event["items"])) or not returned:
            defects.append("missing_keyed_result")
        if any(item != identity for item in returned):
            defects.append("foreign_keyed_result")
        item_keys = [(cast(str, item.get("association")), item.get("key"), item.get("version")) for item in items]
        if len(item_keys) != len(set(item_keys)):
            defects.append("duplicate_keyed_result")
        if any(
            not _natural(item.get("key"))
            or not _natural(item.get("version"), positive=True)
            or not isinstance(item.get("text"), str)
            for item in items
        ):
            defects.append("malformed_source_item")
        if defects:
            _record_request_failure(state, request, "malformed_response")
            _apply_embedded_settlement(state, request, event["settlement"])
            if not was_terminal and not _source_followup_available(
                state, declaration, identity, request, "malformed_response"
            ):
                _object(state["binding_sources"])[identity] = "failed"
                requirements = _object(declaration.get("binding_requirements", {}))
                state["binding_terminal"] = "partial" if requirements.get(identity) == "optional" else "failed"
            return None
        if request in _object(state["terminals"]):
            _record_binding_success(state, request, identity)
            _apply_embedded_settlement(state, request, event["settlement"])
            return None
        limits = _object(declaration.get("binding_limits", {}))
        byte_count = sum(len(cast(str, item["text"]).encode()) for item in items)
        _record_binding_success(state, request, identity)
        _apply_embedded_settlement(state, request, event["settlement"])
        if len(items) > cast(int, limits.get("max_items", len(items))) or byte_count > cast(
            int, limits.get("max_bytes", byte_count)
        ):
            _object(state["binding_sources"])[identity] = "oversize"
            requirements = _object(declaration.get("binding_requirements", {}))
            state["binding_terminal"] = "partial" if requirements.get(identity, "required") == "optional" else "failed"
            return None
        for item in items:
            artifact: Object = {
                "identity": f"{identity}:{item['key']}:{item['version']}",
                "source": source,
                "text": item["text"],
            }
            if artifact in _array(state["artifacts"]):
                _unique(_array(state["defects"]), "duplicate_keyed_result")
            else:
                _array(state["artifacts"]).append(artifact)
        _object(state["binding_sources"])[identity] = "bound"
    elif kind == "source_failure":
        request = cast(str, event.get("request"))
        if request not in dispatched or request not in _object(state["request_associations"]):
            return _reject("missing")
        associations = _strings(_object(state["request_associations"])[request])
        if len(associations) != 1:
            return _reject("binding_request_shape")
        identity = associations[0]
        if event.get("association") != identity:
            return _reject("foreign_association")
        retrieval_sources = _object(declaration.get("retrieval_sources", {}))
        adaptive = identity in retrieval_sources
        source = retrieval_sources.get(identity) if adaptive else _object(state["binding_declarations"]).get(identity)
        if event.get("source") != source:
            return _reject("foreign_source")
        failure = event.get("failure")
        if failure not in FAILURE_CLASSES:
            return _reject("invalid_failure")
        if "settlement" not in event or not _valid_settlement(event.get("settlement"), optional=True):
            return _reject("invalid_settlement")
        disposition = event.get("disposition", "failed")
        requirements = _object(declaration.get("binding_requirements", {}))
        valid_omission = (
            not adaptive
            and disposition == "omitted_optional"
            and requirements.get(identity) == "optional"
            and failure == "permanent"
        )
        if disposition not in ("failed", "omitted_optional") or (
            disposition == "omitted_optional" and not valid_omission
        ):
            failure = "malformed_response"
            disposition = "failed"
        was_terminal = request in _object(state["terminals"])
        _record_request_failure(state, request, cast(str, failure))
        _apply_embedded_settlement(state, request, event.get("settlement"))
        if was_terminal or _source_followup_available(state, declaration, identity, request, cast(str, failure)):
            return None
        if adaptive:
            _object(state["tasks"])[identity] = _materialization_category(
                declaration, "failure", failure=cast(str, failure)
            )
        else:
            _object(state["binding_sources"])[identity] = disposition
            state["binding_terminal"] = "partial" if requirements.get(identity) == "optional" else "failed"
    elif kind == "root_input":
        key = f"RootInputKey:{event.get('target')}:{event.get('port')}"
        specs = [_object(raw) for raw in _array(declaration.get("materializations", []))]
        if not any(key in _strings(spec.get("selector_inputs", [])) for spec in specs):
            return _reject("invalid_provenance")
        if event.get("artifact_type") not in _strings(declaration.get("root_input_types", [])) or not isinstance(
            event.get("value"), str
        ):
            return _reject("invalid_type")
        materialization = _object(state.get("materialization", {})) or {
            "artifact_bytes": 0,
            "artifact_count": 0,
            "ports": {},
            "provenance": {},
            "provenance_edges": 0,
        }
        if key in _object(materialization["provenance"]):
            return _reject("duplicate")
        limits = _object(declaration["materialization_limits"])
        byte_count = len(cast(str, event["value"]).encode())
        if cast(int, materialization["artifact_count"]) + 1 > cast(int, limits["max_artifacts"]):
            return _reject("artifact_count_exceeded")
        if cast(int, materialization["artifact_bytes"]) + byte_count > cast(int, limits["max_artifact_bytes"]):
            return _reject("artifact_bytes_exceeded")
        _object(materialization["provenance"])[key] = []
        materialization["artifact_count"] = cast(int, materialization["artifact_count"]) + 1
        materialization["artifact_bytes"] = cast(int, materialization["artifact_bytes"]) + byte_count
        state["materialization"] = materialization
        _array(state["artifacts"]).append(
            {"identity": key, "artifact_type": event["artifact_type"], "value": event["value"]}
        )
    elif kind == "root_artifact":
        if set(event) not in (
            {"artifact_type", "kind", "port", "target", "value"},
            {"artifact_type", "kind", "node", "port", "target", "value"},
        ) or not isinstance(event.get("value"), str):
            return _reject("invalid_type")
        admitted_root = {
            "artifact_type": event.get("artifact_type"),
            "port": event.get("port"),
            "target": event.get("target"),
        }
        if admitted_root not in [_object(raw) for raw in _array(declaration.get("root_artifacts", []))]:
            return _reject("foreign_owner")
        allocation = f"K{state['allocator_next']}"
        state["allocator_next"] = cast(int, state["allocator_next"]) + 1
        identity = f"ArtifactRef:I0:{allocation}:1"
        materialization = _object(state.get("materialization", {})) or {
            "artifact_bytes": 0,
            "artifact_count": 0,
            "ports": {},
            "provenance": {},
            "provenance_edges": 0,
        }
        producer = f"RootInputKey:{event['target']}:{event['port']}"
        _object(materialization["provenance"])[producer] = {"artifact": identity, "parents": []}
        if event.get("node") is not None:
            port_key = f"root:{event['target']}:{event['node']}:{event['port']}"
            _object(materialization["ports"])[port_key] = {
                "artifact_type": event["artifact_type"],
                "key": identity,
                "value": event["value"],
            }
            _object(materialization.setdefault("input_parents", {}))[port_key] = producer
        materialization["artifact_count"] = cast(int, materialization["artifact_count"]) + 1
        materialization["artifact_bytes"] = cast(int, materialization["artifact_bytes"]) + len(
            cast(str, event["value"]).encode()
        )
        state["materialization"] = materialization
        _array(state["artifacts"]).append(
            {"artifact_type": event["artifact_type"], "identity": identity, "source": producer, "text": event["value"]}
        )
    elif kind == "operation_start":
        if set(event) != {"activation", "attempt", "binding_declaration", "kind", "node", "target"}:
            return _reject("invalid_value")
        admitted = [_object(raw) for raw in _array(declaration.get("operation_occurrences", []))]
        owner = {key: event[key] for key in ("activation", "attempt", "binding_declaration", "node", "target")}
        if owner not in admitted:
            return _reject("foreign_owner")
        initial_specs = [
            spec
            for spec in (_object(raw) for raw in _array(declaration.get("materializations", [])))
            if spec.get("path") == "initial"
        ]
        latest_specs = [
            spec
            for spec in initial_specs
            if spec.get("version_selection") == "latest" and spec.get("declaration") == owner["binding_declaration"]
        ]
        if state.get("binding_finished") is not True or any(
            _object(state["binding_sources"]).get(cast(str, spec["association"])) != "bound" for spec in latest_specs
        ):
            return _reject("missing")
        cleanup_associations = _object(state["binding_cleanup_associations"])
        cleanups = _object(state["binding_cleanup"])
        expected_cleanup = {cast(str, spec["association"]): cast(str, spec["target"]) for spec in initial_specs}
        actual_cleanup: dict[str, str] = {}
        for resource, raw in cleanup_associations.items():
            value = _object(raw)
            for association in _strings(value.get("associations", [value.get("association")])):
                actual_cleanup[association] = resource
        if set(actual_cleanup) != set(expected_cleanup) or set(cleanups) != set(actual_cleanup.values()):
            return _reject("missing")
        for association, target in expected_cleanup.items():
            resource = actual_cleanup[association]
            cleanup_owner = _object(cleanup_associations[resource]).get("owner")
            cleanup_targets = _strings(
                _object(cleanup_associations[resource]).get(
                    "targets", [_object(cleanup_associations[resource]).get("target")]
                )
            )
            if target not in cleanup_targets or cleanup_owner not in (
                "sdk",
                "caller",
            ):
                return _reject("foreign_owner")
            disposition = cleanups[resource]
            if cleanup_owner == "sdk" and disposition not in ("closed", "close_failed", "close_unknown"):
                return _reject("contradictory")
            if cleanup_owner == "caller" and disposition != "left_open":
                return _reject("missing")
        key = f"{event['activation']}:{event['attempt']}"
        occurrences = _object(state["operation_occurrences"])
        if key in occurrences:
            return _reject("duplicate")
        occurrences[key] = {**owner, "published_ports": [], "terminal": None}
    elif kind == "operation_blocked":
        if set(event) != {"activation", "attempt", "binding_declaration", "kind", "node", "reason", "target"}:
            return _reject("invalid_value")
        admitted = [_object(raw) for raw in _array(declaration.get("operation_occurrences", []))]
        owner = {key: event[key] for key in ("activation", "attempt", "binding_declaration", "node", "target")}
        if owner not in admitted or event.get("reason") != "omitted_optional":
            return _reject("foreign_owner")
        key = f"{event['activation']}:{event['attempt']}"
        if key in _object(state["operation_occurrences"]):
            return _reject("duplicate")
        _object(state["operation_occurrences"])[key] = {
            **owner,
            "published_ports": [],
            "terminal": {"category": "blocked", "reason": "omitted_optional"},
        }
    elif kind == "operation_publish":
        candidate = deepcopy(state)
        rejected = _publish_operation(candidate, declaration, event)
        if rejected is None:
            state.clear()
            state.update(candidate)
        elif rejected.get("code") in (
            "artifact_count_exceeded",
            "artifact_bytes_exceeded",
            "provenance_limit_exceeded",
        ):
            occurrence = _object(_object(state["operation_occurrences"])[f"{event['activation']}:{event['attempt']}"])
            occurrence["terminal"] = {"category": "blocked", "reason": "artifact_limit_exhausted"}
            _array(state["publication_failures"]).append(
                {"activation": event["activation"], "attempt": event["attempt"], "reason": "artifact_limit_exhausted"}
            )
        else:
            return rejected
    elif kind == "root_operation_publish":
        if set(event) != {"activation", "attempt", "kind", "node", "outcome", "output_port", "target", "value"}:
            return _reject("invalid_value")
        publications = [_object(raw) for raw in _array(declaration.get("root_publications", []))]
        owner = {key: event[key] for key in ("activation", "node", "outcome", "output_port", "target")}
        matches = [item for item in publications if all(item.get(key) == value for key, value in owner.items())]
        if len(matches) != 1 or not isinstance(event.get("value"), str):
            return _reject("foreign_owner")
        publication = matches[0]
        occurrence = _object(
            _object(state["operation_occurrences"]).get(f"{event['activation']}:{event['attempt']}", {})
        )
        if not occurrence or occurrence.get("terminal") is not None:
            return _reject("missing")
        parent = f"RootInputKey:{event['target']}:{publication['input_port']}"
        materialization = _object(state.get("materialization", {}))
        if not any(
            _object(raw).get("artifact") and key == parent
            for key, raw in _object(materialization["provenance"]).items()
        ):
            return _reject("missing")
        allocation = f"K{state['allocator_next']}"
        output_ref = f"ArtifactRef:I0:{allocation}:1"
        producer = (
            f"OperationOutputKey:{event['activation']}:{event['target']}:"
            f"{event['node']}:{event['outcome']}:{event['output_port']}"
        )
        byte_count = len(cast(str, event["value"]).encode())
        limits = _object(declaration["materialization_limits"])
        if cast(int, materialization["artifact_count"]) + 1 > cast(int, limits["max_artifacts"]):
            return _reject("artifact_count_exceeded")
        if cast(int, materialization["artifact_bytes"]) + byte_count > cast(int, limits["max_artifact_bytes"]):
            return _reject("artifact_bytes_exceeded")
        if cast(int, materialization["provenance_edges"]) + 1 > cast(int, limits["max_provenance_edges"]):
            return _reject("provenance_limit_exceeded")
        state["allocator_next"] = cast(int, state["allocator_next"]) + 1
        _object(materialization["provenance"])[producer] = {"artifact": output_ref, "parents": [parent]}
        materialization["artifact_count"] = cast(int, materialization["artifact_count"]) + 1
        materialization["artifact_bytes"] = cast(int, materialization["artifact_bytes"]) + byte_count
        materialization["provenance_edges"] = cast(int, materialization["provenance_edges"]) + 1
        _array(state["artifacts"]).append(
            {
                "artifact_type": publication["artifact_type"],
                "identity": output_ref,
                "source": f"operation:{event['node']}",
                "text": event["value"],
            }
        )
        occurrence["published_ports"] = [event["output_port"]]
        occurrence["terminal"] = {"category": "success", "outcome": event["outcome"]}
        if publication.get("final") is True:
            state["final_outputs"] = [output_ref]
    elif kind == "materialize_result":
        request = cast(str, event.get("request"))
        if request not in dispatched or request not in _object(state["request_associations"]):
            return _reject("missing")
        spec = _materialization_declaration(declaration, event)
        if spec is None:
            return _reject("missing_materialization")
        expected = _strings(_object(state["request_associations"])[request])
        association = expected[0]
        if not _valid_settlement(event.get("settlement"), optional=False):
            return _reject("invalid_settlement")
        adaptive = spec["path"] == "adaptive"
        outcome = cast(str, event["reported_outcome"]) if adaptive else "retrieved"
        was_terminal = request in _object(state["terminals"])
        raw_items = _array(event["items"])
        pairs = [(_object(item).get("key"), _object(item).get("version")) for item in raw_items]
        malformed = (
            not raw_items
            or len(pairs) != len(set(pairs))
            or expected != [event.get("association")]
            or (spec.get("version_selection") == "latest" and len({pair[0] for pair in pairs}) != 1)
        )
        if malformed:
            _record_request_failure(state, request, "malformed_response")
            if was_terminal and spec.get("version_selection") == "latest":
                _record_late_terminal_conflict(
                    state,
                    request,
                    association=association,
                    failure="malformed_response",
                )
            _apply_embedded_settlement(state, request, event["settlement"])
            if not was_terminal and not _source_followup_available(
                state, declaration, association, request, "malformed_response"
            ):
                if adaptive:
                    _object(state["tasks"]).setdefault(
                        cast(str, event["activation"]),
                        _materialization_category(declaration, "failure", failure="malformed_response"),
                    )
                else:
                    _object(state["binding_sources"])[association] = "failed"
                    state["binding_terminal"] = "failed"
            elif was_terminal and spec.get("version_selection") == "latest":
                original = cast(str, _object(state["terminals"])[request])
                _object(state["binding_sources"])[association] = original
                state["binding_terminal"] = "lost" if original == "lost" else "failed"
            return None
        if was_terminal:
            _record_binding_success(state, request, association, outcome)
            _apply_embedded_settlement(state, request, event["settlement"])
            if spec.get("version_selection") == "latest":
                _record_late_terminal_conflict(state, request, association=association, failure=None)
                original = cast(str, _object(state["terminals"])[request])
                if original == "success":
                    original = "failed"
                _object(state["binding_sources"])[association] = original
                state["binding_terminal"] = "lost" if original == "lost" else "failed"
            return None
        candidate = deepcopy(state)
        rejected = _materialize(candidate, declaration, event)
        if rejected is None:
            state.clear()
            state.update(candidate)
        if "binding_artifacts" in state and (
            rejected is None
            or rejected.get("code")
            in ("artifact_count_exceeded", "artifact_bytes_exceeded", "provenance_limit_exceeded")
        ):
            _array(state["binding_artifacts"]).extend(
                {
                    "association": association,
                    "key": item.get("key"),
                    "source": spec.get("source"),
                    "text": item.get("value"),
                    "version": item.get("version"),
                }
                for item in (_object(raw) for raw in raw_items)
            )
        oversize = rejected is not None and rejected.get("code") in (
            "collection_limit_exceeded",
            "single_cardinality",
            "collection_cardinality",
            "materialization_bytes_exceeded",
        )
        publication_failure = rejected is not None and rejected.get("code") in (
            "artifact_count_exceeded",
            "artifact_bytes_exceeded",
            "provenance_limit_exceeded",
        )
        if rejected is not None and not oversize and not publication_failure:
            return rejected
        _record_binding_success(state, request, association, outcome)
        _apply_embedded_settlement(state, request, event["settlement"])
        if publication_failure:
            _array(state["publication_failures"]).append(
                {"association": association, "code": rejected["code"], "request": request}
            )
            _object(state["binding_sources"])[association] = "bound"
            state["binding_terminal"] = "success"
            return None
        if adaptive:
            condition = "artifact_limit_exhausted" if oversize else "result"
            _object(state["tasks"]).setdefault(
                cast(str, event["activation"]),
                _materialization_category(declaration, condition, reported_outcome=None if oversize else outcome),
            )
        else:
            _object(state["binding_sources"])[association] = "oversize" if oversize else "bound"
            if oversize:
                state["binding_terminal"] = "failed"
            elif all(
                _object(state["binding_sources"]).get(identity) == "bound"
                for identity in _object(state["binding_declarations"])
            ):
                state["binding_terminal"] = "success"
    elif kind == "binding_finish":
        if state.get("binding_finished") is True:
            return _reject("duplicate")
        requirements = _object(declaration.get("binding_requirements", {}))
        if "binding_finished" not in state:
            required = [
                cast(str, _object(raw)["association"])
                for raw in _array(declaration.get("materializations", []))
                if isinstance(raw, dict)
                and raw.get("path") == "initial"
                and requirements.get(cast(str, raw.get("association")), "required") != "optional"
            ]
            if any(_object(state["binding_sources"]).get(association) != "bound" for association in required):
                return _reject("missing_materialization")
            if state["binding_terminal"] is None:
                state["binding_terminal"] = "success"
            return None
        initial = [
            cast(str, spec["association"])
            for spec in (_object(raw) for raw in _array(declaration.get("materializations", [])))
            if spec.get("path") == "initial"
        ]
        required = {association for association in initial if requirements.get(association, "required") != "optional"}
        optional = set(initial) - required
        sources = _object(state["binding_sources"])
        if any(sources.get(association) != "bound" for association in required) or any(
            sources.get(association) not in ("bound", "omitted_optional") for association in optional
        ):
            return _reject("missing_materialization")
        expected_terminal = (
            "partial" if any(sources.get(association) == "omitted_optional" for association in optional) else "success"
        )
        if state["binding_terminal"] not in (None, expected_terminal):
            return _reject("missing_materialization")
        state["binding_terminal"] = expected_terminal
        if "binding_finished" in state:
            state["binding_finished"] = True
            if _array(declaration.get("materializations", [])):
                state.setdefault(
                    "materialization",
                    {
                        "artifact_bytes": 0,
                        "artifact_count": 0,
                        "ports": {},
                        "provenance": {},
                        "provenance_edges": 0,
                    },
                )
    elif kind == "resource":
        resource = cast(str, event["resource"])
        resources = _object(state["resources"])
        if resource not in resources:
            state["resource_count"] = cast(int, state["resource_count"]) + 1
        resources[resource] = {"owner": event["owner"], "safe_detachment": event["safe_detachment"]}
    elif kind == "binding_cleanup_association":
        single = {"association", "kind", "owner", "resource", "target"}
        shared = {"associations", "kind", "owner", "purpose", "resource", "targets"}
        if set(event) not in (single, shared):
            return _reject("invalid_value")
        resource = cast(str, event["resource"])
        associations = _object(state["binding_cleanup_associations"])
        if resource in associations:
            return _reject("duplicate")
        associations[resource] = {key: value for key, value in event.items() if key not in {"kind", "resource"}}
    elif kind == "binding_cleanup":
        if set(event) != {"disposition", "kind", "resource"}:
            return _reject("invalid_value")
        resource = cast(str, event["resource"])
        association = _object(_object(state["binding_cleanup_associations"]).get(resource, {}))
        if not association:
            return _reject("missing")
        if _array(state["local_in_flight"]) or _array(state["remote_outstanding"]):
            return _reject("contradictory")
        if association.get("owner") == "sdk" and event.get("disposition") not in (
            "closed",
            "close_failed",
            "close_unknown",
        ):
            return _reject("contradictory")
        _object(state["binding_cleanup"])[resource] = event["disposition"]
    elif kind == "close_resource":
        resource = cast(str, event["resource"])
        value = _object(_object(state["resources"])[resource])
        cleanup = _object(state["cleanup"])
        if resource in cleanup:
            return _reject("duplicate_cleanup")
        if value["owner"] == "caller":
            cleanup[resource] = "left_open"
        elif _array(state["local_in_flight"]):
            pass
        elif _array(state["remote_outstanding"]) and value["safe_detachment"] != "independent_after_dispatch":
            pass
        else:
            cleanup[resource] = event.get("disposition", "closed")
    elif kind == "bridge_start":
        task = cast(str, event["task"])
        if task in _object(state["tasks"]):
            return _reject("duplicate_task")
        _object(state["tasks"])[task] = "running"
        request = _object(state["association_requests"]).get(task)
        if request is not None:
            _object(state["task_requests"])[task] = request
    elif kind == "bridge_close_unstarted":
        category = cast(str, event["category"])
        if category not in ("blocked", "inconsistent"):
            return _reject("invalid_category")
        _object(state["closed_unstarted"])[cast(str, event["activation"])] = category
    elif kind == "bridge_condition":
        task = cast(str, event["task"])
        if _object(state["tasks"]).get(task) != "running":
            return _reject("task_not_running")
        if set(event) - {"condition", "failure", "kind", "reported_outcome", "task"}:
            return _reject("runtime_mapping")
        request = _object(state["task_requests"]).get(task)
        if request is None:
            request = _object(state["association_requests"]).get(task)
            if request is not None:
                _object(state["task_requests"])[task] = request
        if request is not None:
            fact = _object(_object(state["request_facts"]).get(cast(str, request), {}))
            if fact.get("condition") != event.get("condition"):
                return _reject("request_causality")
            if fact.get("condition") == "failure" and fact.get("failure") != event.get("failure"):
                return _reject("request_causality")
            if fact.get("condition") == "result":
                outcomes = _object(fact.get("outcomes", {}))
                if outcomes.get(task) != event.get("reported_outcome"):
                    return _reject("request_causality")
        mappings = [
            _object(value)
            for value in _array(declaration.get("runtime_mappings", []))
            if _object(value).get("condition") == event.get("condition")
            and _object(value).get("failure") == event.get("failure")
            and _object(value).get("reported_outcome") == event.get("reported_outcome")
        ]
        if len(mappings) != 1:
            return _reject("runtime_mapping")
        mapping = mappings[0]
        condition = mapping.get("condition")
        outcome = mapping.get("outcome")
        category = mapping.get("category")
        categories = _object(declaration.get("outcome_categories", {}))
        if condition not in RUNTIME_CONDITIONS:
            return _reject("runtime_mapping")
        if outcome is None and category == "success":
            return _reject("runtime_mapping")
        if outcome is not None and categories.get(cast(str, outcome)) != category:
            return _reject("runtime_mapping")
        _object(state["tasks"])[task] = f"pending:{mapping['outcome']}:{mapping['category']}"
    elif kind == "bridge_emit":
        task = cast(str, event["task"])
        expected = f"pending:{event['outcome']}:{event['category']}"
        if _object(state["tasks"]).get(task) != expected:
            return _reject("wrong_bridge_emit")
        _object(state["tasks"])[task] = event["category"]
    elif kind == "decision_open":
        wait = cast(str, event["wait"])
        decisions = _object(state["decisions"])
        if wait in decisions:
            return _reject("duplicate_wait")
        active = sum(1 for value in decisions.values() if _object(value).get("closed") is not True)
        if active >= cast(int, declaration.get("max_pending", active + 1)):
            return _reject("pending_limit")
        decisions[wait] = {key: event[key] for key in ("task", "workflow", "artifact", "allowed")}
    elif kind == "decision_submit":
        wait = cast(str, event["wait"])
        decisions = _object(state["decisions"])
        if event.get("invocation", "I0") != "I0":
            return _reject("foreign_owner")
        if wait not in decisions:
            return _reject("missing")
        current = _object(decisions[wait])
        if current.get("closed") is True:
            return _reject("duplicate")
        if event["workflow"] != current["workflow"] or event["artifact"] != current["artifact"]:
            return _reject("foreign_owner")
        if event["decision"] not in _strings(current["allowed"]):
            return _reject("unsupported")
        current["closed"] = True
        _object(state["tasks"])[cast(str, current["task"])] = "success"
    elif kind == "decision_deadline":
        current = _object(_object(state["decisions"])[cast(str, event["wait"])])
        current["closed"] = True
        _object(state["tasks"])[cast(str, current["task"])] = "failure"
    else:
        return _reject("unknown_event")
    return None


def reduce_trace(declaration: Object, events: Sequence[Object]) -> Object:
    state = _initial(declaration)
    for event in events:
        rejected = _advance(state, declaration, event)
        if rejected is not None:
            return rejected
    return {"state": state, "status": "accepted"}
