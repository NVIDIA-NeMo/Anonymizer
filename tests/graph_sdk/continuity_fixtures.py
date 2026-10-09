# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real graph preparation before continuity admission and commit."""

from __future__ import annotations

from dataclasses import replace

from anonymizer.engine.graph_sdk._continuity_values import DecisionChange
from anonymizer.engine.graph_sdk.continuity import (
    AdmittedContinuity,
    ContinuityDecisionDecl,
    ContinuityLimits,
    ContinuityScope,
    ContinuitySession,
    ContinuityState,
    ContinuitySupportDecl,
    StateDecisionKey,
    StateGroupRef,
    StateSubjectRef,
    StateSupportRef,
    TargetStateGroup,
    admit_continuity,
    begin_continuity,
)
from anonymizer.engine.graph_sdk.data import ValidatedDataGraph
from tests.graph_sdk.continuity_graph_fixture import GraphWitness, execute_graph


def limits(**changes: int) -> ContinuityLimits:
    return replace(
        ContinuityLimits(
            max_groups=20,
            max_targets=20,
            max_decisions=20,
            max_decisions_per_group=20,
            max_supports_per_decision=20,
            max_total_supports=100,
            max_name_bytes=100,
            max_slot_bytes=100,
            max_value_depth=8,
            max_value_atoms=100,
            max_value_bytes=1000,
            max_commit_steps=1000,
            max_snapshot_bytes=10000,
            max_snapshot_depth=10,
        ),
        **changes,
    )


async def prepare_case(
    case: str,
    state: ContinuityState,
    *,
    changes: dict[str, DecisionChange] | None = None,
    bounds: ContinuityLimits | None = None,
    values: dict[str, str] | None = None,
) -> tuple[GraphWitness, AdmittedContinuity]:
    selected = limits() if bounds is None else bounds
    sessions: list[ContinuitySession] = []
    group = StateGroupRef(name="explicit-group")

    def before(data: ValidatedDataGraph) -> int:
        session = begin_continuity(
            data=data,
            scope=ContinuityScope(
                lifetime="across_calls",
                state=state,
                groups=tuple(TargetStateGroup(target=t, group=group) for t in data.targets),
            ),
            limits=selected,
        )
        sessions.append(session)
        return session.base.revision + 1

    witness = await execute_graph(case, namespace=state.namespace.name, before_prepare=before, decision_values=values)
    session = sessions[0]
    prior = {entry.reference.key: entry.reference for entry in session.base.decisions}
    declarations = []
    for name, node in witness.decision_nodes:
        key = StateDecisionKey(
            group=group, subject=StateSubjectRef(kind="person", name="caller-subject", version=1), slot=name
        )
        change = "create" if changes is None else changes.get(name, "create")
        support_target = dict(witness.case.support_targets)[name]
        target = next(port.target for port in witness.result.ports if port.node == node and port.port == "approved")
        declarations.append(
            ContinuityDecisionDecl(
                target=target,
                node=node,
                outcome="ok",
                port="approved",
                key=key,
                change=change,
                prior=None if change == "create" else prior.get(key),
                supports=(
                    ContinuitySupportDecl(
                        reference=StateSupportRef(kind="assessment", name=f"stable-{name}", version=1),
                        node=witness.assessor,
                        outcome=support_target,
                        promise=f"support_{support_target}",
                        required_coverage=witness.coverage,
                    ),
                ),
            )
        )
    admitted = admit_continuity(
        prepared=witness.execution.context.prepared, session=session, declarations=tuple(declarations), limits=selected
    )
    return witness, admitted
