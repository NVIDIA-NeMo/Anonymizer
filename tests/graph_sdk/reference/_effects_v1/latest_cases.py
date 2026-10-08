# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent effects_v1 reference: latest cases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import cast

from tests.graph_sdk.reference._effects_v1.builders import (
    _bind,
    _dispatch,
    _materialization_decl,
    _materialization_event,
    _materialization_spec,
    _operation_start,
    _publication_event,
    _reserve,
    _source_failure,
)
from tests.graph_sdk.reference._effects_v1.case import (
    _case,
)
from tests.graph_sdk.reference._effects_v1.model import (
    Object,
    _array,
)


def _latest_selection_specs() -> list[Object]:
    """Exercise latest selection through the complete binding/request owner chain."""
    cases: list[Object] = []
    latest = _materialization_spec("initial", "single", declaration="D0", version_selection="latest")

    def declaration(
        *specs: Object,
        limits: Mapping[str, int] | None = None,
        max_requests: int = 3,
        optional: bool = False,
    ) -> Object:
        value = _materialization_decl(*specs, max_requests=max_requests, limit_changes=limits)
        if optional:
            value["binding_requirements"] = {cast(str, spec["association"]): "optional" for spec in specs}
        return value

    def cleanup(owner: str = "sdk", disposition: str = "closed", *, association: str = "D0") -> list[Object]:
        return [
            {
                "association": association,
                "kind": "binding_cleanup_association",
                "owner": owner,
                "resource": f"Q:{association}",
                "target": "T0",
            },
            {"disposition": disposition, "kind": "binding_cleanup", "resource": f"Q:{association}"},
        ]

    def trace(items: Sequence[Object], *, spec: Object = latest, request: str = "R0") -> list[Object]:
        association = cast(str, spec["association"])
        return [
            _bind(association, "P0"),
            _reserve(request, [association], purpose="initial_binding"),
            _dispatch(request),
            _materialization_event(spec, items, request=request),
            {"kind": "binding_finish"},
            *cleanup(association=association),
            _operation_start(spec),
            _publication_event(spec),
        ]

    one: list[Object] = [{"key": 0, "value": "one", "version": 1}]
    two: list[Object] = [
        {"key": 0, "value": "old", "version": 1},
        {"key": 0, "value": "new", "version": 2},
    ]
    gap: list[Object] = [
        {"key": 0, "value": "old", "version": 1},
        {"key": 0, "value": "new", "version": 3},
    ]
    for name, items in (
        ("one_version", one),
        ("two_versions", two),
        ("reordered_versions", tuple(reversed(two))),
        ("version_gap", gap),
    ):
        cases.append(_case("materialization", f"latest_{name}", declaration(latest), trace(items)))

    duplicate: list[Object] = [two[0], dict(two[0])]
    multikey: list[Object] = [two[0], {"key": 1, "value": "other", "version": 2}]
    for name, items in (("duplicate_pair", duplicate), ("multiple_keys", multikey)):
        cases.append(
            _case(
                "materialization",
                f"latest_{name}",
                declaration(latest, max_requests=1),
                [*trace(items)[:4], *cleanup()],
            )
        )
    items_one_over: list[Object] = [{"key": 0, "value": str(index), "version": index + 1} for index in range(4)]
    items_exact: list[Object] = items_one_over[:3]
    cases.append(
        _case(
            "materialization",
            "latest_items_exact",
            declaration(latest),
            trace(items_exact),
        )
    )
    cases.append(
        _case(
            "materialization",
            "latest_items_one_over",
            declaration(latest),
            [*trace(items_one_over)[:4], *cleanup()],
        )
    )
    bytes_one_over: list[Object] = [{"key": 0, "value": "x" * 13, "version": 1}]
    bytes_exact: list[Object] = [{"key": 0, "value": "x" * 12, "version": 1}]
    cases.append(
        _case(
            "materialization",
            "latest_bytes_exact",
            declaration(latest),
            trace(bytes_exact),
        )
    )
    cases.append(
        _case(
            "materialization",
            "latest_bytes_one_over",
            declaration(latest),
            [*trace(bytes_one_over)[:4], *cleanup()],
        )
    )

    for terminal in ("cancelled", "lost"):
        prefix = trace(two)[:3]
        terminal_events: list[Object] = (
            [
                {"kind": "cancel", "request": "R0"},
                {"kind": "stop", "request": "R0", "usage": {"input": 0, "output": 0}},
            ]
            if terminal == "cancelled"
            else [{"kind": "cancel", "request": "R0"}, {"kind": "lost", "request": "R0"}]
        )
        for shape, items in (("valid", two), ("multikey", multikey)):
            cases.append(
                _case(
                    "materialization",
                    f"latest_{terminal}_late_{shape}",
                    declaration(latest),
                    [*prefix, *terminal_events, _materialization_event(latest, items), *cleanup()],
                )
            )

    retry_prefix: list[Object] = [
        _bind("D0", "P0"),
        _reserve("R0", ["D0"], purpose="initial_binding"),
        _dispatch("R0"),
        _source_failure("D0", "S0", request="R0", failure="retryable"),
        _reserve("R1", ["D0"], purpose="retry"),
        _dispatch("R1"),
        _materialization_event(latest, two, request="R1"),
        {"kind": "binding_finish"},
        *cleanup(),
        _operation_start(latest),
        _publication_event(latest),
    ]
    cases.append(_case("materialization", "latest_retry", declaration(latest), retry_prefix))
    correction_prefix: list[Object] = [
        _bind("D0", "P0"),
        _reserve("R0", ["D0"], purpose="initial_binding"),
        _dispatch("R0"),
        _materialization_event(latest, multikey),
        _reserve("R1", ["D0"], purpose="correction"),
        _dispatch("R1"),
        _materialization_event(latest, two, request="R1"),
        {"kind": "binding_finish"},
        *cleanup(),
        _operation_start(latest),
        _publication_event(latest),
    ]
    cases.append(_case("materialization", "latest_correction", declaration(latest), correction_prefix))
    cases.append(
        _case(
            "materialization",
            "latest_optional_omission",
            declaration(latest, optional=True),
            [
                _bind("D0", "P0"),
                _reserve("R0", ["D0"], purpose="initial_binding"),
                _dispatch("R0"),
                _source_failure("D0", "S0", request="R0", disposition="omitted_optional"),
                {"kind": "binding_finish"},
                *cleanup(),
            ],
        )
    )
    caller_cleanup = trace(one)
    next(event for event in caller_cleanup if event["kind"] == "binding_cleanup_association")["owner"] = "caller"
    next(event for event in caller_cleanup if event["kind"] == "binding_cleanup")["disposition"] = "left_open"
    cases.append(_case("materialization", "latest_caller_cleanup", declaration(latest), caller_cleanup))
    for disposition in ("close_failed", "close_unknown"):
        events = trace(one)
        next(event for event in events if event["kind"] == "binding_cleanup")["disposition"] = disposition
        cases.append(
            _case(
                "materialization",
                f"latest_sdk_cleanup_{disposition}",
                declaration(latest),
                events,
            )
        )
    optional = _materialization_spec(
        "initial",
        "single",
        declaration="D1",
        node="N1",
        port="optional_context",
        version_selection="exact_one",
    )
    mixed_declaration = declaration(latest, optional)
    mixed_declaration["binding_requirements"] = {"D0": "required", "D1": "optional"}
    _array(mixed_declaration["operation_occurrences"]).append(
        {
            "activation": "OP:D1",
            "attempt": "TASK:OP:D1",
            "binding_declaration": "D1",
            "node": "N1",
            "target": "T0",
        }
    )
    shared_cleanup: list[Object] = [
        {
            "associations": ["D0", "D1"],
            "kind": "binding_cleanup_association",
            "owner": "sdk",
            "purpose": "binding",
            "resource": "Q:S0",
            "targets": ["T0"],
        },
        {"disposition": "closed", "kind": "binding_cleanup", "resource": "Q:S0"},
    ]
    mixed_events: list[Object] = [
        _bind("D0", "P0"),
        _bind("D1", "P0"),
        _reserve("R0", ["D0"], purpose="initial_binding"),
        _dispatch("R0"),
        _materialization_event(latest, one),
        _reserve("R1", ["D1"], purpose="initial_binding"),
        _dispatch("R1"),
        _source_failure("D1", "S0", request="R1", disposition="omitted_optional"),
        {"kind": "binding_finish"},
        *shared_cleanup,
        _operation_start(latest),
        _publication_event(latest),
        {
            "activation": "OP:D1",
            "attempt": "TASK:OP:D1",
            "binding_declaration": "D1",
            "kind": "operation_blocked",
            "node": "N1",
            "reason": "omitted_optional",
            "target": "T0",
        },
    ]
    cases.append(
        _case(
            "materialization",
            "latest_bound_with_optional_omission",
            mixed_declaration,
            mixed_events,
        )
    )
    unresolved = [
        deepcopy(event)
        for event in mixed_events
        if not (event.get("request") == "R1" or event.get("kind") == "reserve" and event.get("associations") == ["D1"])
    ]
    cases.append(
        _case(
            "materialization",
            "latest_unresolved_optional_at_finish",
            mixed_declaration,
            unresolved,
            comparison_scope="neutral_only",
            witness_obligation=(
                "RunningBinding.wait remains pending while D1 retrieval is unresolved and cleanup has not run; "
                "after a permanent omitted_optional response it returns one immutable partial result"
            ),
        )
    )
    late_failure = deepcopy(mixed_events)
    finish_index = next(i for i, event in enumerate(late_failure) if event["kind"] == "binding_finish")
    late_failure.insert(
        finish_index + 1,
        deepcopy(next(event for event in late_failure if event["kind"] == "source_failure")),
    )
    cases.append(
        _case(
            "materialization",
            "latest_post_finish_source_failure",
            mixed_declaration,
            late_failure,
            comparison_scope="neutral_only",
            witness_obligation=(
                "Repeated RunningBinding.wait returns the same completed result and does not request or retain "
                "a second provider failure after sealing"
            ),
        )
    )
    late_result = deepcopy(trace(one))
    finish_index = next(i for i, event in enumerate(late_result) if event["kind"] == "binding_finish")
    late_result.insert(
        finish_index + 1,
        deepcopy(next(event for event in late_result if event["kind"] == "materialize_result")),
    )
    cases.append(
        _case(
            "materialization",
            "latest_post_finish_materialization",
            declaration(latest),
            late_result,
            comparison_scope="neutral_only",
            witness_obligation=(
                "Repeated RunningBinding.wait returns the same completed result and does not request, publish, "
                "or retain a second materialization after sealing"
            ),
        )
    )
    for name, maximum in (("exact", 1), ("one_over", 0)):
        preflight_declaration = declaration(latest, limits={"max_provenance_edges": maximum})
        preflight_declaration["preflight_uses_completed_binding"] = True
        cases.append(
            _case(
                "materialization",
                f"latest_provenance_edges_{name}",
                preflight_declaration,
                trace(one),
                "execution_preflight",
            )
        )

    rollback_events: list[Object] = [
        {"artifact_type": "text", "kind": "root_artifact", "port": "left", "target": "T0", "value": "L"},
        {
            "artifact_type": "text",
            "kind": "root_artifact",
            "node": "N1",
            "port": "right",
            "target": "T0",
            "value": "R",
        },
        _bind("D0", "P0"),
        _reserve("R0", ["D0"], purpose="initial_binding"),
        _dispatch("R0"),
        _materialization_event(latest, two),
        {"kind": "binding_finish"},
        *cleanup(association="D0"),
        _operation_start(latest),
        _publication_event(latest, value="x" * 9),
        {
            "activation": "OP:N1",
            "attempt": "TASK:OP:N1",
            "binding_declaration": None,
            "kind": "operation_start",
            "node": "N1",
            "target": "T0",
        },
        {
            "activation": "OP:N1",
            "attempt": "TASK:OP:N1",
            "kind": "root_operation_publish",
            "node": "N1",
            "outcome": "ok",
            "output_port": "result",
            "target": "T0",
            "value": "ok",
        },
    ]
    rollback_decl = declaration(latest, limits={"max_artifact_bytes": 16, "max_artifacts": 5})
    _array(rollback_decl["operation_occurrences"]).append(
        {
            "activation": "OP:N1",
            "attempt": "TASK:OP:N1",
            "binding_declaration": None,
            "node": "N1",
            "target": "T0",
        }
    )
    rollback_decl["root_publications"] = [
        {
            "activation": "OP:N1",
            "artifact_type": "text",
            "final": True,
            "input_port": "right",
            "node": "N1",
            "outcome": "ok",
            "output_port": "result",
            "target": "T0",
        }
    ]
    cases.append(
        _case(
            "materialization",
            "latest_publication_rollback_then_reuse",
            rollback_decl,
            rollback_events,
        )
    )
    return cases
