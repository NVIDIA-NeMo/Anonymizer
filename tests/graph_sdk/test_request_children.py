# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owner-admitted child chains; these tests do not claim runnable delegation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectRejected
from anonymizer.engine.graph_sdk._request_driver import _RequestAuthority
from anonymizer.engine.graph_sdk.requests import (
    AcceptChildEvidence,
    AcceptFailure,
    AcceptResult,
    AssociationResult,
    ChildEvidence,
    ChildRoute,
    Dispatch,
    ExternalSettlement,
    InvocationRequestScope,
    ObserveSettlement,
    PhysicalRequestId,
    PhysicalRequestPolicy,
    RequestCancel,
    RequestChain,
    RequestChainId,
    RequestChild,
    RequestPolicyBinding,
    Reserve,
    SemanticAssociation,
    UnknownUsage,
    bind_request_chain,
    initialize_requests,
    request_receipt,
)
from anonymizer.graph._values import ActivationKey, InvocationId, PlanId, TaskAttemptId
from tests.graph_sdk.reference.request_children.reference import cases, expected


class ChainFixture:
    def __init__(self, budget: int = 3, parse_attempts: int = 2) -> None:
        self.scope = InvocationRequestScope(invocation=InvocationId.new(plan=PlanId.new()))
        self.policy = PhysicalRequestPolicy(
            visibility="dispatch_and_settlement",
            pre_dispatch_control="executor",
            retry_owner="implementation",
            replay="idempotent",
            max_attempts=6,
        )
        self.association = SemanticAssociation(
            task=TaskAttemptId.new(
                activation=ActivationKey(
                    invocation=self.scope.invocation,
                    occurrence=0,
                    parent=None,
                    iteration=None,
                )
            )
        )
        self.authority = _RequestAuthority(
            initialize_requests(
                scope=self.scope,
                hard_limit=budget,
                policies=frozenset({self.policy}),
            )
        )
        self.authority.bind(
            RequestPolicyBinding.create(
                association=self.association,
                policies=frozenset({self.policy}),
            )
        )
        self.chain = RequestChain(
            chain=RequestChainId.new(scope=self.scope),
            associations=frozenset({self.association}),
            policy=self.policy,
            role="generation",
            followups=frozenset({"correction", "conversation_restart", "failover", "retry"}),
            routes=(
                ChildRoute(name="primary", parse_attempts=parse_attempts),
                ChildRoute(name="fallback", parse_attempts=1),
            ),
        )
        self.authority.bind_chain(self.chain)
        self.last: PhysicalRequestId | None = None
        self.evidence: ChildEvidence | None = None

    def reserve(
        self,
        source: Literal["initial", "correction", "conversation_restart", "failover"] = "initial",
        *,
        predecessor: PhysicalRequestId | None = None,
        route: str | None = None,
        evidence: ChildEvidence | None = None,
    ) -> PhysicalRequestId:
        request = PhysicalRequestId.new(scope=self.scope)
        purpose = "correction" if source == "conversation_restart" else source
        self.authority.apply(
            Reserve(
                request=request,
                purpose=purpose,
                associations=self.chain.associations,
                policy=self.policy,
                child=RequestChild(
                    chain=self.chain.chain,
                    role=self.chain.role,
                    route=route or ("fallback" if source == "failover" else "primary"),
                    source_kind=source,
                    predecessor=predecessor,
                    evidence=evidence,
                ),
            )
        )
        return request

    def settle(self, request: PhysicalRequestId, *, malformed: bool = True) -> None:
        self.authority.apply(Dispatch(request=request))
        self.authority.apply(
            ObserveSettlement(
                settlement=ExternalSettlement(
                    request=request,
                    disposition="completed",
                    usage=UnknownUsage(),
                    remote_stopped=True,
                )
            )
        )
        if malformed:
            self.authority.apply(AcceptFailure(request=request, failure="malformed_response"))
        else:
            self.authority.apply(
                AcceptResult(
                    request=request,
                    results=(
                        AssociationResult(
                            association=self.association,
                            outcome="ok",
                            outputs=(),
                            consumed_context_ports=frozenset(),
                        ),
                    ),
                )
            )
        self.last = request

    def proof(self, kind: Literal["parser_rejected", "generation_exhausted"]) -> ChildEvidence:
        evidence = ChildEvidence(
            chain=self.chain.chain, request=self.last or PhysicalRequestId.new(scope=self.scope), kind=kind
        )
        self.authority.apply(AcceptChildEvidence(evidence=evidence))
        self.evidence = evidence
        return evidence

    def step(self, letter: str) -> None:
        if letter in "PE":
            self.proof("parser_rejected" if letter == "P" else "generation_exhausted")
            return
        sources: dict[str, Literal["initial", "correction", "conversation_restart", "failover"]] = {
            "I": "initial",
            "C": "correction",
            "R": "conversation_restart",
            "V": "failover",
        }
        source = sources[letter]
        request = self.reserve(
            source, predecessor=None if letter == "I" else self.last, evidence=None if letter == "I" else self.evidence
        )
        self.settle(request, malformed=letter != "V")


def test_reference_manifest_is_frozen() -> None:
    manifest = json.loads((Path(__file__).parent / "reference/request_children/manifest.json").read_text())
    assert manifest["corpus_version"] == 2
    assert len(cases()) == manifest["case_count"] == 35
    assert sum(expected(trace)[0] for trace in cases()) == manifest["valid_case_count"] == 11
    assert all(expected(trace)[0] or expected(trace[:-1])[0] for trace in cases())
    assert hashlib.sha256(json.dumps(cases(), separators=(",", ":")).encode()).hexdigest() == manifest["digest"]


@pytest.mark.parametrize("trace", cases())
def test_bounded_child_language(trace: str) -> None:
    fixture = ChainFixture()
    accepted = True
    executed = ""
    before = fixture.authority.state
    try:
        for letter in trace:
            before = fixture.authority.state
            executed += letter
            fixture.step(letter)
    except EffectRejected:
        accepted = False
        assert fixture.authority.state == before
    verdict, _ = expected(trace)
    assert accepted == verdict
    assert executed == trace
    prefix = trace if accepted else trace[:-1]
    _, dispatched = expected(prefix)
    receipt = request_receipt(fixture.authority.state)
    assert receipt.dispatched_count == len(receipt.terminals) == len(receipt.settlements) == dispatched
    assert [(row.category, row.failure) for row in receipt.terminals] == [
        ("success", None) if letter == "V" else ("failure", "malformed_response")
        for letter in prefix
        if letter in "ICRV"
    ]
    assert len(receipt.child_evidence) == prefix.count("P") + prefix.count("E")
    assert not receipt.defects and not fixture.authority.state.reserved
    assert receipt.chains == (fixture.chain,)


@pytest.mark.parametrize(
    ("trace", "budget", "parse_attempts", "accepted", "dispatched"),
    [
        ("IPEV", 3, 1, True, 2),
        ("IPCPEV", 3, 2, True, 3),
        ("IPRPEV", 3, 2, True, 3),
        ("IPCPEV", 2, 2, False, 2),
        ("IPCPEV", 4, 2, True, 3),
        ("IPEV", 3, 2, False, 1),
        ("IPCPEV", 3, 3, False, 2),
    ],
)
def test_child_language_bound_controls(
    trace: str, budget: int, parse_attempts: int, accepted: bool, dispatched: int
) -> None:
    fixture = ChainFixture(budget=budget, parse_attempts=parse_attempts)
    completed = True
    try:
        for letter in trace:
            fixture.step(letter)
    except EffectRejected:
        completed = False
    assert completed is accepted
    receipt = request_receipt(fixture.authority.state)
    assert receipt.dispatched_count == len(receipt.terminals) == len(receipt.settlements) == dispatched


def test_interleaved_chains_use_explicit_predecessor() -> None:
    fixture = ChainFixture(budget=6)
    first_chain = fixture.chain
    other = replace(first_chain, chain=RequestChainId.new(scope=fixture.scope))
    fixture.authority.bind_chain(other)
    fixture.step("I")
    first = fixture.last
    proof = fixture.proof("parser_rejected")
    fixture.chain = other
    fixture.step("I")
    foreign_predecessor = fixture.last
    fixture.chain = first_chain
    with pytest.raises(EffectRejected):
        fixture.reserve("correction", predecessor=foreign_predecessor, evidence=proof)
    request = fixture.reserve("correction", predecessor=first, evidence=proof)
    fixture.settle(request, malformed=False)
    assert len(fixture.authority.state.dispatched) == 3


def test_forged_proof_missing_settlement_and_early_exhaustion() -> None:
    fixture = ChainFixture()
    request = fixture.reserve()
    fixture.authority.apply(Dispatch(request=request))
    fixture.last = request
    with pytest.raises(EffectRejected):
        fixture.proof("parser_rejected")
    fixture.authority.apply(AcceptFailure(request=request, failure="malformed_response"))
    with pytest.raises(EffectRejected):
        fixture.proof("parser_rejected")
    fixture.authority.apply(
        ObserveSettlement(
            settlement=ExternalSettlement(
                request=request,
                disposition="completed",
                usage=UnknownUsage(),
                remote_stopped=True,
            )
        )
    )
    proof = ChildEvidence(chain=fixture.chain.chain, request=request, kind="parser_rejected")
    with pytest.raises(EffectRejected):
        fixture.reserve("correction", predecessor=request, evidence=proof)
    fixture.proof("parser_rejected")
    with pytest.raises(EffectRejected):
        fixture.proof("generation_exhausted")


def test_scope_role_policy_and_declared_fallback_are_enforced() -> None:
    fixture = ChainFixture(parse_attempts=1)
    for letter in "IPE":
        fixture.step(letter)
    for route in ("primary", "undeclared"):
        with pytest.raises(EffectRejected):
            fixture.reserve("failover", predecessor=fixture.last, evidence=fixture.evidence, route=route)
    foreign = ChainFixture()
    with pytest.raises(EffectRejected):
        bind_request_chain(state=fixture.authority.state, chain=foreign.chain)
    with pytest.raises(EffectRejected):
        ChildEvidence(
            chain=fixture.chain.chain, request=PhysicalRequestId.new(scope=foreign.scope), kind="parser_rejected"
        )
    with pytest.raises(EffectRejected):
        bind_request_chain(state=fixture.authority.state, chain=fixture.chain)
    with pytest.raises(EffectRejected):
        replace(fixture.chain, routes=(ChildRoute(name="primary", parse_attempts=7),))
    with pytest.raises(EffectRejected):
        fixture.authority.apply(
            Reserve(
                request=PhysicalRequestId.new(scope=fixture.scope),
                purpose="failover",
                associations=fixture.chain.associations,
                policy=fixture.policy,
                child=RequestChild(
                    chain=fixture.chain.chain,
                    role="readiness",
                    route="fallback",
                    source_kind="failover",
                    predecessor=fixture.last,
                    evidence=fixture.evidence,
                ),
            )
        )


def test_budget_and_transportless_cancel_do_not_invent_dispatches() -> None:
    fixture = ChainFixture(budget=1)
    unused = fixture.reserve()
    fixture.authority.apply(RequestCancel(request=unused))
    assert request_receipt(fixture.authority.state).dispatched_count == 0
    fixture.step("I")
    proof = fixture.proof("parser_rejected")
    request = fixture.reserve("correction", predecessor=fixture.last, evidence=proof)
    assert not any(item.request == request for item in fixture.authority.state.reserved)
    assert fixture.authority.state.denials[-1].category == "budget_stopped"
    assert request_receipt(fixture.authority.state).dispatched_count == 1


def test_terminal_is_immutable_and_evidence_cannot_counterfeit_failure() -> None:
    fixture = ChainFixture()
    request = fixture.reserve()
    fixture.settle(request, malformed=False)
    terminal = fixture.authority.state.terminals[0]
    fixture.authority.apply(AcceptFailure(request=request, failure="malformed_response"))
    assert fixture.authority.state.terminals == (terminal,)
    assert fixture.authority.state.defects[0].code == "conflicting_terminal"
    with pytest.raises(EffectRejected):
        fixture.proof("parser_rejected")


def test_chain_admission_is_bounded_and_precedes_effects() -> None:
    fixture = ChainFixture()
    for _ in range(fixture.policy.max_attempts - 1):
        fixture.authority.bind_chain(replace(fixture.chain, chain=RequestChainId.new(scope=fixture.scope)))
    with pytest.raises(EffectRejected):
        fixture.authority.bind_chain(replace(fixture.chain, chain=RequestChainId.new(scope=fixture.scope)))
    fixture.step("I")
    with pytest.raises(EffectRejected):
        fixture.authority.bind_chain(replace(fixture.chain, chain=RequestChainId.new(scope=fixture.scope)))
    # Omitting child provenance cannot escape the contract after its root failed.
    with pytest.raises(EffectRejected):
        fixture.authority.apply(
            Reserve(
                request=PhysicalRequestId.new(scope=fixture.scope),
                purpose="initial",
                associations=fixture.chain.associations,
                policy=fixture.policy,
            )
        )


def test_undeclared_source_and_cancelled_request_id_are_rejected() -> None:
    fixture = ChainFixture()
    fixture.authority.state = replace(fixture.authority.state, chains=(replace(fixture.chain, followups=frozenset()),))
    unused = fixture.reserve()
    fixture.authority.apply(RequestCancel(request=unused))
    with pytest.raises(EffectRejected):
        fixture.authority.apply(
            Reserve(
                request=unused,
                purpose="initial",
                associations=fixture.chain.associations,
                policy=fixture.policy,
                child=RequestChild(
                    chain=fixture.chain.chain, role="generation", route="primary", source_kind="initial"
                ),
            )
        )
    fixture.step("I")
    proof = fixture.proof("parser_rejected")
    with pytest.raises(EffectRejected):
        fixture.reserve("correction", predecessor=fixture.last, evidence=proof)


def test_child_retry_keeps_existing_failure_and_replay_rules() -> None:
    fixture = ChainFixture()
    request = fixture.reserve()
    fixture.authority.apply(Dispatch(request=request))
    fixture.authority.apply(AcceptFailure(request=request, failure="retryable"))
    retry = PhysicalRequestId.new(scope=fixture.scope)
    fixture.authority.apply(
        Reserve(
            request=retry,
            purpose="retry",
            associations=fixture.chain.associations,
            policy=fixture.policy,
            child=RequestChild(
                chain=fixture.chain.chain, role="generation", route="primary", source_kind="retry", predecessor=request
            ),
        )
    )
    fixture.authority.apply(Dispatch(request=retry))
    assert len(fixture.authority.state.dispatched) == 2


def test_pending_children_share_the_association_attempt_bound() -> None:
    fixture = ChainFixture(budget=100)
    chains = [fixture.chain]
    for _ in range(5):
        chain = replace(fixture.chain, chain=RequestChainId.new(scope=fixture.scope))
        fixture.authority.bind_chain(chain)
        chains.append(chain)
    for chain in chains[:2]:
        fixture.chain = chain
        for letter in "IPC":
            fixture.step(letter)
    assert len(fixture.authority.state.dispatched) == 4
    for chain in chains[2:]:
        fixture.chain = chain
        fixture.reserve()
    assert len(fixture.authority.state.reserved) == 2
    assert len(fixture.authority.state.denials) == 2
    assert all(item.category == "request_limit_stopped" for item in fixture.authority.state.denials)


@pytest.mark.parametrize(
    "ordering",
    ["before_reserve", "before_dispatch", "earlier_support", "earlier_before_reserve", "earlier_before_dispatch"],
)
@pytest.mark.parametrize("conflict", ["terminal", "settlement"])
def test_child_proof_rechecks_current_support(ordering: str, conflict: str) -> None:
    earlier = ordering.startswith("earlier")
    fixture = ChainFixture(parse_attempts=2 if earlier else 1)
    for letter in "IP":
        fixture.step(letter)
    predecessor = fixture.last
    assert predecessor is not None
    first_terminal = fixture.authority.state.terminals[0]
    successor = None
    if earlier:
        for letter in "CP":
            fixture.step(letter)
    if ordering != "earlier_support":
        fixture.proof("generation_exhausted")
        if ordering.endswith("before_dispatch"):
            successor = fixture.reserve("failover", predecessor=fixture.last, evidence=fixture.evidence)
    if conflict == "terminal":
        fixture.authority.apply(AcceptFailure(request=predecessor, failure="permanent"))
    else:
        fixture.authority.apply(
            ObserveSettlement(
                settlement=ExternalSettlement(
                    request=predecessor,
                    disposition="unknown",
                    usage=UnknownUsage(),
                    remote_stopped=None,
                )
            )
        )
    before = fixture.authority.state
    with pytest.raises(EffectRejected, match="contradictory"):
        if ordering.endswith("before_reserve"):
            fixture.reserve("failover", predecessor=fixture.last, evidence=fixture.evidence)
        elif ordering.endswith("before_dispatch"):
            assert successor is not None
            fixture.authority.apply(Dispatch(request=successor))
        else:
            fixture.proof("generation_exhausted")
    assert fixture.authority.state is before
    assert fixture.authority.state.terminals[0] == first_terminal
    assert len(fixture.authority.state.dispatched) == (2 if earlier else 1)
    if successor is not None:
        fixture.authority.apply(RequestCancel(request=successor))
        assert not fixture.authority.state.reserved
        assert successor not in fixture.authority.state.dispatched


def test_child_dispatch_accepts_consistent_reobserved_support() -> None:
    fixture = ChainFixture(parse_attempts=1)
    for letter in "IPE":
        fixture.step(letter)
    predecessor = fixture.last
    assert predecessor is not None
    proof = fixture.evidence
    successor = fixture.reserve("failover", predecessor=predecessor, evidence=proof)
    fixture.authority.apply(AcceptFailure(request=predecessor, failure="malformed_response"))
    fixture.authority.apply(
        ObserveSettlement(
            settlement=ExternalSettlement(
                request=predecessor,
                disposition="completed",
                usage=UnknownUsage(),
                remote_stopped=True,
            )
        )
    )
    fixture.settle(successor, malformed=False)
    receipt = request_receipt(fixture.authority.state)
    assert receipt.dispatched_count == 2 and not receipt.defects
    assert [item.category for item in receipt.terminals] == ["failure", "success"]
    assert proof in receipt.child_evidence


def test_ordinary_followup_keeps_first_terminal_semantics() -> None:
    fixture = ChainFixture()
    fixture.authority = _RequestAuthority(
        initialize_requests(
            scope=fixture.scope,
            hard_limit=3,
            policies=frozenset({fixture.policy}),
        )
    )
    fixture.authority.bind(
        RequestPolicyBinding.create(
            association=fixture.association,
            policies=frozenset({fixture.policy}),
        )
    )
    first = PhysicalRequestId.new(scope=fixture.scope)
    fixture.authority.apply(
        Reserve(request=first, purpose="initial", associations=fixture.chain.associations, policy=fixture.policy)
    )
    fixture.settle(first)
    fixture.authority.apply(AcceptFailure(request=first, failure="permanent"))
    successor = PhysicalRequestId.new(scope=fixture.scope)
    fixture.authority.apply(
        Reserve(request=successor, purpose="correction", associations=fixture.chain.associations, policy=fixture.policy)
    )
    fixture.authority.apply(Dispatch(request=successor))
    assert len(fixture.authority.state.dispatched) == 2
    assert fixture.authority.state.terminals[0].failure == "malformed_response"
    assert fixture.authority.state.defects[0].code == "conflicting_terminal"
