# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared traversal of admitted workflow bodies for engine admission."""

from __future__ import annotations

from anonymizer.graph.workflow import AdmittedActivationWorkflow, AdmittedWorkflow, SubgraphNode


def reachable_workflows(workflow: AdmittedActivationWorkflow) -> tuple[AdmittedWorkflow, ...]:
    """Visit each reachable body once, preserving shared-subgraph identity."""
    reachable: list[AdmittedWorkflow] = []
    pending = [workflow.workflow]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        reachable.append(current)
        pending.extend(node.body for node in current.nodes if isinstance(node, SubgraphNode))
    return tuple(reachable)
