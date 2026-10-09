# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk.continuity import StateNamespace, commit_qualified, create_continuity_state
from anonymizer.engine.graph_sdk.extensions import bind_projection_source, project_protected
from tests.graph_sdk.continuity_fixtures import limits, prepare_case
from tests.graph_sdk.continuity_graph_adapter import project
from tests.graph_sdk.reference.continuity.reference import greatest


def test_create_reuse_supersede_and_immutable_receipt() -> None:
    async def run() -> None:
        state = create_continuity_state(namespace=StateNamespace(name="history"), limits=limits())
        graph, admitted = await prepare_case("independent", state)
        first = commit_qualified(admitted=admitted, execution=graph.result, qualification=graph.qualified)
        assert first.status == "committed" and first.resulting_revision == 1
        assert {ref.key.slot for ref in first.retained} == {"x", "y"}
        graph2, admitted2 = await prepare_case("independent", state, changes={"x": "reuse", "y": "reuse"})
        second = commit_qualified(admitted=admitted2, execution=graph2.result, qualification=graph2.qualified)
        assert second.status == "no_change" and second.resulting_revision == 1
        graph3, admitted3 = await prepare_case(
            "independent", state, changes={"x": "supersede", "y": "reuse"}, values={"x": "new"}
        )
        third = commit_qualified(admitted=admitted3, execution=graph3.result, qualification=graph3.qualified)
        assert third.status == "committed" and third.resulting_revision == 2
        assert {ref.key.slot: ref.revision for ref in third.retained} == {"x": 2, "y": 1}
        assert first.resulting_revision == 1 and {ref.revision for ref in first.retained} == {1}
        projection = project_protected(
            source=bind_projection_source(execution=graph.result, qualification=graph.qualified, continuity=first)
        )
        assert {item.content for item in projection.items} == {"A", "B"}

    asyncio.run(run())


@pytest.mark.parametrize("case", ["independent", "sole_b", "sole_b_without_y", "consumes_y", "atomic", "alternating"])
def test_actual_graph_conflict_closure_matches_frozen_reference(case: str) -> None:
    async def run() -> None:
        state = create_continuity_state(namespace=StateNamespace(name="history"), limits=limits())
        initial, initial_admitted = await prepare_case(case, state)
        commit_qualified(admitted=initial_admitted, execution=initial.result, qualification=initial.qualified)
        graph, admitted = await prepare_case(case, state, changes={"x": "reuse", "z": "reuse", "y": "create"})
        before = state.view(limits=limits())
        receipt = commit_qualified(admitted=admitted, execution=graph.result, qualification=graph.qualified)
        names = {target: name for name, target in graph.targets}
        assert frozenset(names[t] for t in receipt.releasable) == greatest(project(graph), frozenset({"y"}))
        assert state.view(limits=limits()) == before
        assert {conflict.key.slot for conflict in receipt.conflicts} == {"y"}
        projection = project_protected(
            source=bind_projection_source(execution=graph.result, qualification=graph.qualified, continuity=receipt)
        )
        assert {item.target for item in projection.items} == set(receipt.releasable)

    asyncio.run(run())


def test_stale_whole_history_and_foreign_qualification() -> None:
    async def run() -> None:
        state = create_continuity_state(namespace=StateNamespace(name="history"), limits=limits())
        one, admission_one = await prepare_case("independent", state)
        two, admission_two = await prepare_case("independent", state)
        commit_qualified(admitted=admission_one, execution=one.result, qualification=one.qualified)
        receipt = commit_qualified(admitted=admission_two, execution=two.result, qualification=two.qualified)
        assert receipt.status == "conflicted" and not receipt.releasable and len(receipt.withheld) == 2
        assert state.view(limits=limits()).revision == 1
        with pytest.raises(EffectRejected) as failure:
            commit_qualified(admitted=admission_two, execution=two.result, qualification=one.qualified)
        assert failure.value.code is EffectCode.FOREIGN_OWNER

    asyncio.run(run())


def test_commit_step_exact_and_one_less_before_mutation() -> None:
    async def run() -> None:
        for bound, success in [(2, True), (1, False)]:
            state = create_continuity_state(namespace=StateNamespace(name="history"), limits=limits())
            graph, admitted = await prepare_case("independent", state, bounds=limits(max_commit_steps=bound))
            before = state.view(limits=limits())
            if success:
                assert (
                    commit_qualified(admitted=admitted, execution=graph.result, qualification=graph.qualified).status
                    == "committed"
                )
            else:
                with pytest.raises(EffectRejected) as failure:
                    commit_qualified(admitted=admitted, execution=graph.result, qualification=graph.qualified)
                assert failure.value.code is EffectCode.LIMIT_EXCEEDED
                assert state.view(limits=limits()) == before

    asyncio.run(run())


def test_capacity_rejects_whole_commit_not_fitting_subset() -> None:
    async def run() -> None:
        for bounds in [limits(max_decisions=1), limits(max_decisions_per_group=1), limits(max_snapshot_bytes=150)]:
            state = create_continuity_state(namespace=StateNamespace(name="history"), limits=bounds)
            # Admission decision count is an earlier bound for max_decisions.
            before = state.view(limits=bounds)
            with pytest.raises(EffectRejected) as failure:
                graph, admitted = await prepare_case("independent", state, bounds=bounds)
                commit_qualified(admitted=admitted, execution=graph.result, qualification=graph.qualified)
            assert failure.value.code is EffectCode.LIMIT_EXCEEDED
            assert state.view(limits=bounds) == before

    asyncio.run(run())


def test_direct_singletons_explicit_groups_and_fresh_lifetimes() -> None:
    from anonymizer.engine.graph_sdk.continuity import (
        ContinuityScope,
        StateGroupRef,
        TargetStateGroup,
        begin_continuity,
    )
    from tests.graph_sdk.test_preparation import _data

    data = _data(2)
    first = begin_continuity(data=data, scope=None, limits=limits())
    second = begin_continuity(data=data, scope=None, limits=limits())
    assert first.base.identity != second.base.identity
    assert len({row.group for row in first.groups}) == 2
    assert first.base.namespace == StateNamespace(name="per-call")
    group = StateGroupRef(name="explicit")
    rows = tuple(TargetStateGroup(target=t, group=group) for t in data.targets)
    explicit = begin_continuity(
        data=data, scope=ContinuityScope(lifetime="per_call", state=None, groups=rows), limits=limits()
    )
    assert len({row.group for row in explicit.groups}) == 1 and all(len(peers) == 1 for peers in data.atomic)
    state = create_continuity_state(namespace=StateNamespace(name="shared"), limits=limits())
    one = begin_continuity(
        data=data, scope=ContinuityScope(lifetime="across_calls", state=state, groups=rows), limits=limits()
    )
    two = begin_continuity(
        data=data, scope=ContinuityScope(lifetime="across_calls", state=state, groups=rows), limits=limits()
    )
    assert one.base.identity is two.base.identity is state.identity
    for scope, code in [
        (ContinuityScope(lifetime="across_calls", state=None, groups=rows), EffectCode.MISSING),
        (ContinuityScope(lifetime="per_call", state=state, groups=rows), EffectCode.CONTRADICTORY),
        (ContinuityScope(lifetime="per_call", state=None, groups=rows[:1]), EffectCode.MISSING),
        (ContinuityScope(lifetime="per_call", state=None, groups=(rows[0], rows[0])), EffectCode.DUPLICATE),
    ]:
        with pytest.raises(EffectRejected) as failure:
            begin_continuity(data=data, scope=scope, limits=limits())
        assert failure.value.code is code


def test_declaration_and_current_support_are_required_after_restore(tmp_path) -> None:
    from anonymizer.engine.graph_sdk.continuity import (
        admit_continuity,
        capture_snapshot,
        encode_snapshot,
        load_snapshot,
        restore_snapshot,
    )
    from anonymizer.engine.graph_sdk.qualification import qualify

    async def run() -> None:
        state = create_continuity_state(namespace=StateNamespace(name="history"), limits=limits())
        graph, admitted = await prepare_case("independent", state)
        commit_qualified(admitted=admitted, execution=graph.result, qualification=graph.qualified)
        path = tmp_path / "snapshot.json"
        path.write_bytes(
            encode_snapshot(
                snapshot=capture_snapshot(state=state, expected_revision=1, limits=limits()), limits=limits()
            )
        )
        restored = restore_snapshot(snapshot=load_snapshot(path=path, limits=limits()), limits=limits())
        graph2, admitted2 = await prepare_case("independent", restored, changes={"x": "reuse", "y": "reuse"})
        unsupported = qualify(
            admitted=graph2.qualified._admitted, result=graph2.result, current=graph2.current, submissions=()
        )
        assert not unsupported.qualified
        receipt = commit_qualified(admitted=admitted2, execution=graph2.result, qualification=unsupported)
        assert receipt.status == "no_change" and not receipt.releasable and not receipt.retained
        assert restored.view(limits=limits()).revision == 1
        missing = admit_continuity(
            prepared=admitted2.prepared, session=admitted2.session, declarations=(), limits=limits()
        )
        with pytest.raises(EffectRejected) as failure:
            commit_qualified(admitted=missing, execution=graph2.result, qualification=graph2.qualified)
        assert failure.value.code is EffectCode.MISSING
        accepted = commit_qualified(admitted=admitted2, execution=graph2.result, qualification=graph2.qualified)
        assert accepted.status == "no_change" and len(accepted.releasable) == 2

    asyncio.run(run())


def test_failed_producer_and_unrelated_provisional_decision() -> None:
    from dataclasses import replace

    from anonymizer.engine.graph_sdk.continuity import admit_continuity
    from anonymizer.engine.graph_sdk.evidence import AssessmentSubmission
    from anonymizer.engine.graph_sdk.qualification import qualify

    async def run() -> None:
        state = create_continuity_state(namespace=StateNamespace(name="history"), limits=limits())
        graph, admitted = await prepare_case("sole_b", state)
        targets = dict(graph.targets)
        declarations = tuple(replace(d, target=targets["A"]) for d in admitted.declarations)
        accepted = admit_continuity(
            prepared=admitted.prepared, session=admitted.session, declarations=declarations, limits=limits()
        )
        only_b = qualify(
            admitted=graph.qualified._admitted,
            result=graph.result,
            current=graph.current,
            submissions=tuple(
                AssessmentSubmission(fact=fact) for fact in graph.result.assessments if fact.outcome == "B"
            ),
        )
        assert {out.target for out in only_b.qualified} == {targets["B"]}
        receipt = commit_qualified(admitted=accepted, execution=graph.result, qualification=only_b)
        assert receipt.releasable == (targets["B"],) and {r.key.slot for r in receipt.retained} == {"x", "y"}
        # Independent case: omit B support, so y remains only an unrelated declaration/physical decision.
        independent = create_continuity_state(namespace=StateNamespace(name="independent"), limits=limits())
        graph2, admitted2 = await prepare_case("independent", independent)
        only_a = qualify(
            admitted=graph2.qualified._admitted,
            result=graph2.result,
            current=graph2.current,
            submissions=tuple(
                AssessmentSubmission(fact=fact) for fact in graph2.result.assessments if fact.outcome == "A"
            ),
        )
        second = commit_qualified(admitted=admitted2, execution=graph2.result, qualification=only_a)
        assert {r.key.slot for r in second.retained} == {"x"}
        assert {entry.reference.key.slot for entry in independent.view(limits=limits()).decisions} == {"x"}

    asyncio.run(run())


def test_alternating_commit_exact_work_bound_preserves_state_on_exhaustion() -> None:
    async def run() -> None:
        for maximum, accepted in [(27, True), (26, False)]:
            state = create_continuity_state(namespace=StateNamespace(name="history"), limits=limits())
            initial, initial_admitted = await prepare_case("alternating", state)
            commit_qualified(admitted=initial_admitted, execution=initial.result, qualification=initial.qualified)
            graph, admitted = await prepare_case(
                "alternating",
                state,
                changes={"x": "reuse", "y": "create", "z": "reuse"},
                bounds=limits(max_commit_steps=maximum),
            )
            before = state.view(limits=limits())
            if accepted:
                receipt = commit_qualified(admitted=admitted, execution=graph.result, qualification=graph.qualified)
                assert len(receipt.releasable) == 2 and receipt.status == "partial"
            else:
                with pytest.raises(EffectRejected) as failure:
                    commit_qualified(admitted=admitted, execution=graph.result, qualification=graph.qualified)
                assert failure.value.code is EffectCode.LIMIT_EXCEEDED
            assert state.view(limits=limits()) == before

    asyncio.run(run())


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("fault", ["duplicate", "missing"])
def test_admission_mixed_faults_follow_global_phase_order(reverse: bool, fault: str) -> None:
    from dataclasses import replace

    from anonymizer.engine.graph_sdk.continuity import admit_continuity

    async def run() -> None:
        state = create_continuity_state(namespace=StateNamespace(name="history"), limits=limits())
        _, admitted = await prepare_case("sole_b", state)
        first, second = admitted.declarations
        if fault == "duplicate":
            declarations = (replace(first, outcome="absent"), replace(second, supports=second.supports * 2))
            expected = EffectCode.DUPLICATE
        else:
            declarations = (replace(first, change="invalid"), replace(second, outcome="absent"))
            expected = EffectCode.MISSING
        before = state.view(limits=limits())
        with pytest.raises(EffectRejected) as failure:
            admit_continuity(
                prepared=admitted.prepared,
                session=admitted.session,
                declarations=tuple(reversed(declarations)) if reverse else declarations,
                limits=limits(),
            )
        assert failure.value.code is expected
        assert state.view(limits=limits()) == before

    asyncio.run(run())


def test_initial_support_requires_qualified_binding_location() -> None:
    from anonymizer.engine.graph_sdk.evidence import AssessmentSubmission
    from anonymizer.engine.graph_sdk.qualification import qualify

    async def run() -> None:
        state = create_continuity_state(namespace=StateNamespace(name="history"), limits=limits())
        graph, admitted = await prepare_case("sole_b", state)
        only_a = qualify(
            admitted=graph.qualified._admitted,
            result=graph.result,
            current=graph.current,
            submissions=tuple(AssessmentSubmission(fact=f) for f in graph.result.assessments if f.outcome == "A"),
        )
        assert len(only_a.qualified) == 1
        before = state.view(limits=limits())
        with pytest.raises(EffectRejected) as failure:
            commit_qualified(admitted=admitted, execution=graph.result, qualification=only_a)
        assert failure.value.code is EffectCode.MISSING
        assert state.view(limits=limits()) == before

    asyncio.run(run())
