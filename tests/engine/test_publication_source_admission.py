# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Admission binds a delegated contract's publication sources to producers it owns."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import httpx
import pandas as pd
import pytest

from anonymizer import AnonymizerConfig, AnonymizerInput, Substitute
from anonymizer.config.models import ReplaceModelSelection
from anonymizer.engine.graph_sdk.executor import admit_execution_plan
from anonymizer.graph import EffectCode, EffectRejected
from tests.engine import test_detection_source_frame as detection_tests
from tests.engine.test_detection_graph_publication import _modify_dd_responses

scripted_detection = detection_tests.scripted_detection


@pytest.fixture
def substitute_plan(scripted_detection: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
    """The admitted plan of a real public Substitute run (its owner publishes the map producers)."""

    def response(request: httpx.Request, answer: httpx.Response) -> httpx.Response:
        if "Generate synthetic replacements" not in json.dumps(json.loads(request.content)["messages"]):
            return answer
        payload = answer.json()
        payload["choices"][0]["message"]["content"] = (
            '```json\n{"replacements":[{"original":"Alice","label":"first_name","synthetic":"Maya"}]}\n```'
        )
        return httpx.Response(200, json=payload)

    _modify_dd_responses(scripted_detection, monkeypatch, response)
    path = tmp_path / "source.csv"
    pd.DataFrame({"text": ["Alice travels."]}).to_csv(path, index=False)
    facade = detection_tests.public_anonymizer(scripted_detection)
    facade._selected_models.replace = ReplaceModelSelection(replacement_generator="singleton")
    result = facade.run(
        config=AnonymizerConfig(replace=Substitute(), emit_telemetry=False), data=AnonymizerInput(source=str(path))
    )
    return result._execution._execution


def _readmit(plan: Any, policies: tuple[Any, ...]) -> Any:
    return admit_execution_plan(
        context=plan.context,
        capabilities=plan.capabilities,
        policies=policies,
        assessment_productions=plan.assessment_productions,
        assessment_limits=plan.assessment_limits,
        ingress_limits=plan.ingress_limits,
    )


@pytest.mark.parametrize("source", ["not_submitted_by_owner", "unproduced_port"])
def test_publication_sources_outside_the_owners_producers_reject_at_admission(
    substitute_plan: Any, source: str
) -> None:
    owner = next(
        policy for policy in substitute_plan.policies if policy.delegated and policy.delegated.publication_sources
    )
    contract = owner.delegated
    (node, ports), *_ = contract.publication_sources
    if source == "not_submitted_by_owner":
        # A real node of this plan whose submissions the owner does not own: the owner itself.
        published = ((owner.node, frozenset(ports)),)
    else:
        published = ((node, frozenset({*ports, "never_produced"})),)
    others = tuple(policy for policy in substitute_plan.policies if policy is not owner)
    _readmit(substitute_plan, (owner, *others))
    changed = replace(owner, delegated=replace(contract, publication_sources=published))
    with pytest.raises(EffectRejected) as error:
        _readmit(substitute_plan, (changed, *others))
    assert error.value.code is EffectCode.FOREIGN_OWNER
