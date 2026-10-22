# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An actual staged-assessment rollback preserves a qualified peer."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from anonymizer.engine.graph_sdk.preparation import prepare
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.graph import ProtectionRequirement
from tests.graph_sdk import effects_map_execution, effects_map_fixtures
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.graph_sdk.test_assessment_publication_rollback import _assert_rollback, _execute
from tests.graph_sdk.workflow_construction import admit_static_workflow


def test_rejected_staged_assessment_preserves_actual_qualified_peer(monkeypatch: pytest.MonkeyPatch) -> None:
    def static(**kwargs: Any) -> Any:
        kwargs["protection"] = (
            ProtectionRequirement(
                outcome="ok",
                meaning="map membership accepted",
                subject_port="members",
                consumed_ports=frozenset({"default"}),
                coverage=frozenset(),
            ),
        )
        return admit_static_workflow(**kwargs)

    def preparation(**kwargs: Any) -> Any:
        kwargs["configuration"] = replace(
            kwargs["configuration"], purpose="protection", required_protection_outcomes=frozenset({"ok"})
        )
        return prepare(**kwargs)

    monkeypatch.setattr(effects_map_fixtures, "admit_static_workflow", static)
    monkeypatch.setattr(effects_map_execution, "prepare", preparation)
    fixture, result, callbacks, discard = _execute(monkeypatch)
    _assert_rollback(fixture, result, callbacks, discard)
    execution = result._execution
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    (accepted,) = output.qualified
    (verified,) = output.verified
    assert verified.reference.artifact == discard.prior[0].evidence_artifact
    assert accepted.evidence == frozenset({verified.reference})
    assert accepted.candidate == result.final_outputs[0].candidate
    assert {row.target for row in output.targets if not row.withholding} == {accepted.target}
    statuses = {row.target: row for row in output.record.statuses}
    assert statuses[accepted.target].qualification == "met" and statuses[accepted.target].completion == "closed"
    assert all(row.qualification == "unmet" for target, row in statuses.items() if target != accepted.target)
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == output
