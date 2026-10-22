# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Root closure and selected unstarted member witnesses."""

from __future__ import annotations

import asyncio

import pytest

from anonymizer.engine.graph_sdk.qualification import qualify
from tests.graph_sdk.evidence_fixtures import _execute_assessment
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.graph_sdk.test_versioned_map_and_membership_witnesses import _execute_root_protected_map, _root_inputs


def test_open_root_membership_withholds_completed_candidate() -> None:
    execution, result = asyncio.run(_execute_assessment())
    admitted, current, submissions = _inputs(execution, result)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(baseline.qualified) == 1
    (root,) = result.record.memberships
    assert root.parent is None and root.closed and len(root.members) == 1


def test_selected_blocked_member_is_closed_without_started_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    execution, result, _ = _execute_root_protected_map(monkeypatch, count=1, blocked_member=True)
    admitted, current, submissions = _root_inputs(execution, result)
    (expansion,) = result.states[0].expansions
    (member,) = tuple(entry for entry in result.states[0].entries if entry.activation in expansion.members)
    assert member.status == "blocked"
    assert any(member.activation in row.members and row.closed for row in result.record.memberships)
    (terminal,) = tuple(row for row in result.record.terminals if row.activation == member.activation)
    assert terminal.category == "blocked" and terminal.attempt is None
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert not output.qualified and len(output.verified) == 1
    assert output.record.statuses[0].completion == "closed"
    assert output.record.statuses[0].qualification == "unmet"
    assert output.targets[0].withholding == frozenset({"terminal_failure"})
