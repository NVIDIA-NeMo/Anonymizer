# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request reducer event translation and normalized receipt comparisons."""

from __future__ import annotations

from typing import Any, cast

from anonymizer.engine.graph_sdk.requests import (
    AcceptFailure,
    AcceptResult,
    AssociationResult,
    Dispatch,
    ExactUsage,
    ExternalSettlement,
    MarkLost,
    ObserveSettlement,
    PhysicalRequestId,
    PhysicalRequestPolicy,
    RequestCancel,
    RequestPolicyBinding,
    RequestState,
    Reserve,
    ScopeCancel,
    StopAcknowledged,
    UnknownUsage,
    advance_requests,
    bind_request_policies,
)


class _Closable:
    def __init__(self, disposition: str) -> None:
        self.disposition = disposition

    async def close(self) -> None:
        if self.disposition == "close_failed":
            raise RuntimeError("close failed")


def _policy(value: dict[str, Any]) -> PhysicalRequestPolicy:
    return PhysicalRequestPolicy(
        visibility="dispatch_and_settlement",
        pre_dispatch_control="executor",
        retry_owner=value["retry_owner"],
        replay=value["replay"],
        max_attempts=cast(int, value["max_attempts"]),
    )


def _apply(
    state: RequestState,
    event: dict[str, Any],
    tasks: dict[str, Any],
    requests: dict[str, PhysicalRequestId],
    policies: dict[str, PhysicalRequestPolicy],
) -> RequestState:
    kind = event["kind"]
    if kind == "bind_policy":
        return bind_request_policies(
            state=state,
            binding=RequestPolicyBinding.create(
                association=tasks[event["association"]],
                policies=frozenset(policies[item] for item in event["policies"]),
            ),
        )
    if kind == "reserve":
        return advance_requests(
            state=state,
            event=Reserve(
                request=requests[event["request"]],
                purpose=event["purpose"],
                associations=frozenset(tasks[item] for item in event["associations"]),
                policy=policies[event["policy"]],
            ),
        )
    if kind == "dispatch":
        value = Dispatch(request=requests[event["request"]])
    elif kind == "result":
        value = AcceptResult(
            request=requests[event["request"]],
            results=tuple(
                AssociationResult(
                    association=tasks[item],
                    outcome=event["outcomes"].get(item, "ok"),
                    outputs=(),
                    consumed_context_ports=frozenset(),
                )
                for item in event["returned"]
            ),
        )
    elif kind == "failure":
        value = AcceptFailure(request=requests[event["request"]], failure=event["failure"])
    elif kind == "cancel":
        value = RequestCancel(request=requests[event["request"]])
    elif kind == "scope_cancel":
        value = ScopeCancel()
    elif kind == "stop":
        usage = _usage(event["usage"]) if "usage" in event else object()
        value = StopAcknowledged(request=requests[event["request"]], usage=cast(Any, usage))
    elif kind == "lost":
        value = MarkLost(request=requests[event["request"]])
    elif kind == "settlement":
        value = ObserveSettlement(
            settlement=ExternalSettlement(
                request=requests[event["request"]],
                disposition=event["disposition"],
                usage=_usage(event["usage"]),
                remote_stopped=event["remote_stopped"],
            )
        )
    elif kind == "source_failure":
        state = advance_requests(
            state=state,
            event=AcceptFailure(request=requests[event["request"]], failure=event["failure"]),
        )
        settlement = cast(dict[str, Any], event["settlement"])
        return advance_requests(
            state=state,
            event=ObserveSettlement(
                settlement=ExternalSettlement(
                    request=requests[event["request"]],
                    disposition=settlement["disposition"],
                    usage=_usage(settlement["usage"]),
                    remote_stopped=settlement["remote_stopped"],
                )
            ),
        )
    elif kind == "source_result":
        returned = cast(list[dict[str, Any]], event["items"])
        state = advance_requests(
            state=state,
            event=AcceptResult(
                request=requests[event["request"]],
                results=(
                    AssociationResult(
                        association=tasks[returned[0]["association"]],
                        outcome=event["outcome"],
                        outputs=(),
                        consumed_context_ports=frozenset(),
                    ),
                ),
            ),
        )
        settlement = cast(dict[str, Any], event["settlement"])
        return advance_requests(
            state=state,
            event=ObserveSettlement(
                settlement=ExternalSettlement(
                    request=requests[event["request"]],
                    disposition=settlement["disposition"],
                    usage=_usage(settlement["usage"]),
                    remote_stopped=settlement["remote_stopped"],
                )
            ),
        )
    else:
        raise AssertionError(kind)
    return advance_requests(state=state, event=value)


def _usage(value: object):
    if value == "unknown":
        return UnknownUsage()
    usage = cast(dict[str, int], value)
    return ExactUsage(input_units=usage["input"], output_units=usage["output"])


def _normalize(
    state: Any,
    tasks: dict[str, Any],
    requests: dict[str, PhysicalRequestId],
    policies: dict[str, PhysicalRequestPolicy],
) -> dict[str, object]:
    task_names: dict[object, str] = {value: key for key, value in tasks.items()}
    request_names = {value: key for key, value in requests.items()}
    attempts = {
        name: sum(association in item.associations for item in state.dispatches)
        for association, name in task_names.items()
        if any(association in item.associations for item in state.dispatches)
    }
    request_failures = {
        request_names[item.request]: item.failure for item in state.terminals if item.failure is not None
    }
    association_terminals: dict[str, object] = {}
    for terminal in state.terminals:
        dispatch = next((item for item in state.dispatches if item.request == terminal.request), None)
        if dispatch is not None and terminal.failure is not None:
            association_terminals.update(
                {
                    task_names[item]: {
                        "request": request_names[terminal.request],
                        "failure": terminal.failure,
                        "policy": next(key for key, policy in policies.items() if policy is dispatch.policy),
                    }
                    for item in dispatch.associations
                }
            )
    bindings = {
        task_names[item.association]: sorted(
            key for key, policy in policies.items() if any(policy is retained for retained in item.policies)
        )
        for item in state.bindings
    }
    dispatches = {item.request: item for item in state.dispatches}
    dispatched_ids = frozenset(dispatches)
    settlements = {
        request_names[item.request]: {
            "disposition": item.disposition,
            "usage": (
                "unknown"
                if isinstance(item.usage, UnknownUsage)
                else {"input": item.usage.input_units, "output": item.usage.output_units}
            ),
            "remote_stopped": item.remote_stopped,
        }
        for item in state.settlements
    }
    request_facts: dict[str, object] = {}
    for terminal in state.terminals:
        fact: dict[str, object] = {
            "condition": "request_inconsistent" if terminal.category == "inconsistent" else terminal.category
        }
        if terminal.category == "success":
            fact = {
                "condition": "result",
                "outcomes": {task_names[item.association]: item.outcome for item in terminal.results},
            }
        elif terminal.category == "failure" and terminal.failure is not None:
            fact = {"condition": "failure", "failure": terminal.failure}
        elif terminal.category == "cancelled" and terminal.request in dispatched_ids:
            fact = {"condition": "cancel_after_dispatch"}
        if terminal.request in dispatched_ids:
            request_facts[request_names[terminal.request]] = fact
    return {
        "bindings": bindings,
        "dispatched": [request_names[item.request] for item in state.dispatches],
        "dispatched_count": len(state.dispatches),
        "denials": {task_names[item]: denial.category for denial in state.denials for item in denial.associations},
        "terminals": {request_names[item.request]: item.category for item in state.terminals},
        "defects": [item.code for item in state.defects],
        "local_in_flight": sorted(request_names[item] for item in state.local_in_flight),
        "remote_outstanding": sorted(request_names[item] for item in state.remote_outstanding),
        "attempts": attempts,
        "request_failures": request_failures,
        "association_terminals": association_terminals,
        "reservations": {
            request_names[item.request]: sorted(task_names[value] for value in item.associations)
            for item in getattr(state, "reserved", ())
        },
        "request_associations": {
            request_names[request]: sorted(task_names[item] for item in reservation.associations)
            for request, reservation in dispatches.items()
        },
        "request_policies": {
            request_names[request]: next(key for key, policy in policies.items() if policy is reservation.policy)
            for request, reservation in dispatches.items()
        },
        "reservation_policies": {
            request_names[item.request]: next(key for key, policy in policies.items() if policy is item.policy)
            for item in getattr(state, "reserved", ())
        },
        "settlements": settlements,
        "request_facts": request_facts,
        "cancel_requested": sorted(request_names[item] for item in state.cancel_requested),
        "association_requests": {
            task_names[association]: request_names[item.request]
            for item in state.dispatches
            for association in item.associations
        },
    }
