# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace

import pytest

from anonymizer.engine.execution.phase8_ndd_backend import _Phase8Operation
from anonymizer.engine.execution.phase8_runtime import _GroupInconsistent
from anonymizer.engine.execution.phase8_service import (
    _admit_obligations,
    _backend_group_operation,
    _context_request,
    _new_wire,
    _Phase8AcceptedMention,
    _Phase8ContextProjection,
    _Phase8GroupInput,
    _Phase8InvocationInconsistent,
    _Phase8WireRegistry,
    _valid_group_input,
    _zero_route_admitted,
)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "foreign", "wrong_owner"])
def test_phase8_analysis_rejects_invalid_mention_coverage_or_ownership(kind: str) -> None:
    first, second = object(), object()
    registry = _Phase8WireRegistry()
    member_tokens = (registry.new(), registry.new())
    mention_tokens = (registry.new(),)
    mention_owners = {mention_tokens[0]: first}
    mention_identities = {mention_tokens[0]: object()}
    member_owners = {member_tokens[0]: first, member_tokens[1]: second}
    source_mentions: list[str] = [mention_tokens[0]]
    source_members: list[str] = [member_tokens[0]]
    if kind == "missing":
        source_mentions = []
    elif kind == "duplicate":
        source_mentions *= 2
    elif kind == "foreign":
        source_mentions = [registry.new()]
    elif kind == "wrong_owner":
        source_members = [member_tokens[1]]
    payload = {
        "privacy_obligations": [
            {
                "statement": "protect",
                "kind": "direct",
                "sensitivity": "high",
                "source_member_tokens": source_members,
                "source_mention_tokens": source_mentions,
            }
        ],
        "utility_obligations": [],
    }

    if kind == "foreign":
        with pytest.raises(_Phase8InvocationInconsistent):
            _admit_obligations(
                payload,
                member_tokens,
                mention_tokens,
                mention_owners,
                mention_identities,
                member_owners,
                registry,
            )
    else:
        assert (
            _admit_obligations(
                payload,
                member_tokens,
                mention_tokens,
                mention_owners,
                mention_identities,
                member_owners,
                registry,
            )
            is None
        )


@pytest.mark.parametrize("kind", ["missing", "duplicate", "foreign"])
def test_phase8_analysis_rejects_invalid_consumed_context_binding_evidence(kind: str) -> None:
    member = object()

    class Backend:
        def run_operation(self, operation: _Phase8Operation, request: dict[str, object]) -> object:
            assert operation is _Phase8Operation.ANALYZE
            members = request["members"]
            contexts = request["context_bindings"]
            assert isinstance(members, list) and isinstance(contexts, list)
            token = contexts[0]["binding_token"]
            consumed = [] if kind == "missing" else [token, token] if kind == "duplicate" else ["foreign"]
            return _Response(
                operation,
                {
                    "analyzed_member_tokens": [members[0]["member_token"]],
                    "consumed_context_binding_tokens": consumed,
                    "privacy_obligations": [],
                    "utility_obligations": [],
                },
            )

    group_input = _Phase8GroupInput(
        {member: "original"},
        {member: False},
        context_projections=(_Phase8ContextProjection(member, object(), 0, "context"),),
    )
    outcome = _backend_group_operation(Backend(), group_input)((member,), {member: "original"})

    assert isinstance(outcome.terminal, _GroupInconsistent)
    assert outcome.revisions is None


def test_phase8_context_binding_preserves_owner_ordinal_and_shared_content_multiplicity() -> None:
    first, second = object(), object()
    members = (first, second)
    group_input = _Phase8GroupInput(
        {first: "one", second: "two"},
        {first: True, second: True},
        context_projections=(
            _Phase8ContextProjection(first, object(), 0, "shared"),
            _Phase8ContextProjection(second, object(), 0, "shared"),
        ),
    )
    registry = _Phase8WireRegistry()
    wire = _new_wire(members, group_input, registry)

    request = _context_request(wire, members, group_input)

    assert len(request) == 2
    assert request[0]["text"] == request[1]["text"] == "shared"
    assert request[0]["binding_token"] != request[1]["binding_token"]
    assert request[0]["owner_member_token"] != request[1]["owner_member_token"]
    assert [item["ordinal"] for item in request] == [0, 0]


@pytest.mark.parametrize("mutation", ["foreign_owner", "duplicate_ordinal", "ordinal_type"])
def test_phase8_context_owner_or_ordinal_mutation_rejects_before_dispatch(mutation: str) -> None:
    member = object()
    first = _Phase8ContextProjection(member, object(), 0, "context")
    contexts = (first,)
    if mutation == "foreign_owner":
        contexts = (replace(first, owner=object()),)
    elif mutation == "duplicate_ordinal":
        contexts = (first, _Phase8ContextProjection(member, object(), 0, "other"))
    elif mutation == "ordinal_type":
        contexts = (replace(first, ordinal=True),)
    group_input = _Phase8GroupInput(
        {member: "original"},
        {member: True},
        context_projections=contexts,
    )

    assert not _valid_group_input((member,), {member: "baseline"}, group_input)


@pytest.mark.parametrize("guard", ["provenance", "mentions", "applied", "identity"])
def test_phase8_zero_route_requires_each_frozen_guard_independently(guard: str) -> None:
    member = object()
    baselines = {member: "original"}
    group_input: _Phase8GroupInput | None = _Phase8GroupInput({member: "original"}, {member: False})
    if guard == "provenance":
        group_input = None
    elif guard == "mentions":
        assert group_input is not None
        group_input = replace(
            group_input,
            accepted_mentions=(_Phase8AcceptedMention(member, object(), 0, 1, "o", "name", "span"),),
        )
    elif guard == "applied":
        assert group_input is not None
        group_input.phase7_applied[member] = True
    elif guard == "identity":
        baselines[member] = "different"

    assert not _zero_route_admitted((member,), baselines, group_input)


def test_phase8_zero_route_accepts_only_the_exact_all_true_guard_set() -> None:
    member = object()
    baselines = {member: "original"}
    group_input = _Phase8GroupInput({member: "original"}, {member: False})

    assert _zero_route_admitted((member,), baselines, group_input)


class _Response:
    def __init__(self, operation: _Phase8Operation, payload: dict[str, object]) -> None:
        self.operation = operation
        self.payload = payload
        self.failed = False
