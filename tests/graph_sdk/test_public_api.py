# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public-only entry points, examples, import isolation and legacy compatibility."""

from __future__ import annotations

import asyncio
import inspect
import json
import runpy
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from anonymizer import graph as g

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/graph_sdk/local_workflows.py"


def test_public_examples_and_controls() -> None:
    completed = subprocess.run([sys.executable, str(EXAMPLE)], check=True, capture_output=True, text=True)
    results = json.loads(completed.stdout)
    assert results["examples"][0]["qualification"] == "not_assessed"
    assert results["examples"][1]["qualification"] == "met"
    assert [item["callback_counts"] for item in results["examples"]] == [[1, 0], [1, 1]]
    assert [item["protected_items"] for item in results["examples"]] == [0, 1]
    assert all(not item["projection_reexecuted"] for item in results["examples"])
    assert results["controls"]["negative_callbacks"] == 0
    assert results["controls"]["nested_loop_rejected"]
    assert results["context"]["remote_context_calls"] == 2
    assert results["sync"]["sync_callbacks"] == 1


@pytest.mark.parametrize(
    "component",
    [
        "anonymizer.engine.graph_sdk._workflow_admission",
        "anonymizer.engine.graph_sdk._activation_reducer",
    ],
)
def test_pure_imports_do_not_load_provider_or_dataframe_stack(component: str) -> None:
    code = (
        f"import sys; import {component}; assert 'pandas' not in sys.modules; assert 'data_designer' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_finite_public_exports_and_factory_signatures() -> None:
    assert len(g.__all__) == len(set(g.__all__))
    assert all(not name.startswith("_") for name in g.__all__)
    for name in g.__all__:
        assert getattr(g, name) is getattr(g, name)
        assert name in dir(g)
    assert "submissions" not in inspect.signature(g.qualify).parameters
    assert "submissions" not in inspect.signature(g.observed_qualify).parameters
    assert "AssessmentSubmission" not in g.__all__
    assert "advance_requests" not in g.__all__
    assert inspect.signature(g.InitialContextDecl).parameters["version_selection"].default == "exact_one"
    assert "candidate_port" in inspect.signature(g.ProtectionRequirement).parameters
    assert g.MapItemPort and g.MapItemSubjectRef and g.VerifiedEvidence and g.RemoteContext
    with pytest.raises(AttributeError):
        getattr(g, "not_a_graph_export")


def test_legacy_root_exports_are_the_same_objects() -> None:
    import anonymizer
    from anonymizer.config.anonymizer_config import AnonymizerConfig, Detect, Rewrite
    from anonymizer.config.replace_strategies import Annotate, Hash, Redact, Substitute
    from anonymizer.interface.anonymizer import Anonymizer

    expected = {
        "Anonymizer": Anonymizer,
        "AnonymizerConfig": AnonymizerConfig,
        "Detect": Detect,
        "Rewrite": Rewrite,
        "Annotate": Annotate,
        "Hash": Hash,
        "Redact": Redact,
        "Substitute": Substitute,
    }
    for name in anonymizer.__all__:
        assert getattr(anonymizer, name) is getattr(anonymizer, name)
        assert name in dir(anonymizer)
    for name, value in expected.items():
        assert getattr(anonymizer, name) is value
    assert type(anonymizer.DEFAULT_ENTITY_LABELS) is tuple
    assert anonymizer.graph is g
    with pytest.raises(AttributeError):
        getattr(anonymizer, "not_an_export")


def test_saved_result_selective_requalification_never_reexecutes() -> None:
    example = runpy.run_path(str(EXAMPLE))
    admitted, services, transform, verifier = example["build"](verified=True)
    result = g.execute_sync(admitted=admitted, capabilities=admitted.capabilities, services=services)
    qualification, current = example["qualification_inputs"](admitted, result)
    qualified = g.qualify(admitted=qualification, result=result, current=current)
    assert len(qualified.verified) == len(result.assessments) == 1
    final = result.final_outputs[0].candidate.artifact
    reduced = g.evidence_revision_view(
        admitted=qualification,
        result=result,
        artifacts=tuple(artifact for artifact, _ in result.artifacts if artifact != final),
        configurations=tuple((item.node, item.environment.configuration) for item in result.assessments),
    )
    withheld = g.qualify(admitted=qualification, result=result, current=reduced)
    assert not withheld.qualified
    assert not g.project_protected(source=g.bind_projection_source(execution=result, qualification=withheld)).items
    restored = g.qualify(admitted=qualification, result=result, current=current)
    assert restored == qualified and (transform.calls, verifier.calls) == (1, 1)
    session = g.begin_observations(
        limits=g.ObservationLimits(
            max_pairs=2,
            max_events=4,
            max_nodes=200,
            max_depth=12,
            max_encoded_bytes=8192,
            max_duration_ns=1_000_000_000,
            max_count=100,
        ),
        clock=example["Clock"](),
        sink=None,
    )
    assert isinstance(session, g.ObservationSession)
    observed = g.observed_qualify(observations=session, admitted=qualification, result=result, current=current)
    assert observed == qualified
    receipt = session.close()
    assert isinstance(receipt, g.ObservationReceipt)
    assert len(receipt.pairs) == 1 and receipt.pairs[0].terminal.outcome == "success"
    assert (transform.calls, verifier.calls) == (1, 1)


def test_default_qualification_consumes_all_retained_assessments_and_bounds_count() -> None:
    from tests.graph_sdk.evidence_fixtures import _execute_assessment, _qualification_limits
    from tests.graph_sdk.qualification_fixtures import _inputs

    execution, result = asyncio.run(_execute_assessment(partial_assessment=True))
    admitted, current, _ = _inputs(execution, result)
    qualified = g.qualify(admitted=admitted, result=result, current=current)
    assert len(qualified.verified) == len(result.assessments) == 2
    bounded = g.admit_qualification(
        execution=execution,
        productions=execution.assessment_productions,
        limits=replace(_qualification_limits(), max_submissions=1),
    )
    with pytest.raises(g.EffectRejected) as caught:
        g.qualify(admitted=bounded, result=result, current=current)
    assert caught.value.code == g.EffectCode.LIMIT_EXCEEDED


def test_public_observed_qualification_keeps_invalid_input_pair() -> None:
    class Clock:
        def now_ns(self) -> int:
            return 1

    session = g.begin_observations(
        limits=g.ObservationLimits(
            max_pairs=2,
            max_events=4,
            max_nodes=200,
            max_depth=12,
            max_encoded_bytes=8192,
            max_duration_ns=100,
            max_count=100,
        ),
        clock=Clock(),
        sink=None,
    )
    assert isinstance(session, g.ObservationSession)
    with pytest.raises(g.EffectRejected) as caught:
        g.observed_qualify(
            observations=session, admitted=cast(Any, None), result=cast(Any, None), current=cast(Any, None)
        )
    assert caught.value.code == g.EffectCode.INVALID_TYPE
    receipt = session.close()
    assert isinstance(receipt, g.ObservationReceipt)
    assert len(receipt.pairs) == 1 and receipt.pairs[0].start.boundary == "qualification"
    assert receipt.pairs[0].terminal.outcome == "failure"


def test_bundled_skill_template_imports_and_configuration_still_work(tmp_path: Path) -> None:
    import ast
    import re

    skill = EXAMPLE.parents[2] / "skills/anonymizer/SKILL.md"
    template = next(
        block for block in re.findall(r"```python\n(.*?)```", skill.read_text(), re.S) if "def build_config()" in block
    )
    source = tmp_path / "records.csv"
    source.write_text("text\nAlice\n")
    template = template.replace("TODO: path to .csv / .parquet / .jsonl", str(source)).replace(
        "TODO: name of the text column", "text"
    )
    tree = ast.parse(template)
    selected: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "anonymizer"
        or isinstance(node, ast.FunctionDef)
        and node.name == "build_config"
    ]
    scope: dict[str, Any] = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(skill), "exec"), scope)
    data, config = scope["build_config"]()
    assert data.data_summary and config.detect.gliner_threshold == 0.3
    assert config.rewrite.risk_tolerance == "low"
    assert config.rewrite.max_repair_iterations == 3


@pytest.mark.parametrize("selected", ["a", "b", "both"])
def test_public_two_target_current_artifact_selection(selected: str) -> None:
    example = runpy.run_path(str(EXAMPLE))
    plan, services, transform, verifier = example["build"](verified=True, texts=("Alice A", "Alice B"))
    result = g.execute_sync(admitted=plan, capabilities=plan.capabilities, services=services)
    admitted, all_current = example["qualification_inputs"](plan, result)
    candidates = result.final_outputs
    expected = {
        item.target
        for item in (candidates[:1] if selected == "a" else candidates[1:] if selected == "b" else candidates)
    }
    artifacts = {item.artifact for item in result.ports if item.target in expected}
    current = g.evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=tuple(ref for ref, _ in result.artifacts if ref in artifacts),
        configurations=all_current.configurations,
    )
    qualified = g.qualify(admitted=admitted, result=result, current=current)
    assert len(qualified.verified) == len(result.assessments) == 2
    assert {item.target for item in qualified.qualified} == expected
    assert {item.candidate for item in qualified.qualified} == {
        item.candidate for item in candidates if item.target in expected
    }
    source = g.bind_projection_source(execution=result, qualification=qualified)
    first = g.project_protected(source=source)
    assert {item.target for item in first.items} == expected
    assert g.project_protected(source=source) == first
    assert g.qualify(admitted=admitted, result=result, current=current) == qualified
    assert (transform.calls, verifier.calls) == (2, 2)


@pytest.mark.parametrize("invalid", ["admitted", "result", "current"])
@pytest.mark.parametrize("observed", [False, True])
def test_public_qualification_validates_each_outer_argument(invalid: str, observed: bool) -> None:
    example = runpy.run_path(str(EXAMPLE))
    plan, services, transform, verifier = example["build"](verified=True)
    result = g.execute_sync(admitted=plan, capabilities=plan.capabilities, services=services)
    admitted, current = example["qualification_inputs"](plan, result)
    arguments: dict[str, Any] = dict(admitted=admitted, result=result, current=current)
    arguments[invalid] = None
    session = g.begin_observations(
        limits=g.ObservationLimits(
            max_pairs=2,
            max_events=4,
            max_nodes=200,
            max_depth=12,
            max_encoded_bytes=8192,
            max_duration_ns=1_000_000_000,
            max_count=100,
        ),
        clock=example["Clock"](),
        sink=None,
    )
    assert isinstance(session, g.ObservationSession)
    with pytest.raises(g.EffectRejected) as caught:
        if observed:
            g.observed_qualify(observations=session, **arguments)
        else:
            g.qualify(**arguments)
    assert caught.value.code == g.EffectCode.INVALID_TYPE
    receipt = session.close()
    assert isinstance(receipt, g.ObservationReceipt)
    assert len(receipt.pairs) == int(observed)
    if observed:
        assert receipt.pairs[0].terminal.reasons == frozenset({"invalid_input"})
    assert (transform.calls, verifier.calls) == (1, 1)


def test_public_diagnosis_after_prepare_rejection() -> None:
    example = runpy.run_path(str(EXAMPLE))
    session = g.begin_observations(
        limits=g.ObservationLimits(
            max_pairs=2,
            max_events=4,
            max_nodes=200,
            max_depth=12,
            max_encoded_bytes=8192,
            max_duration_ns=1_000_000_000,
            max_count=100,
        ),
        clock=example["Clock"](),
        sink=None,
    )
    assert isinstance(session, g.ObservationSession)
    with pytest.raises(g.PreparationRejected):
        cast(Any, g.observed_prepare)(
            observations=session,
            data=None,
            workflow=None,
            activation_limits=None,
            bound_inputs=(),
            configuration=None,
            selections=(),
            capabilities=(),
            limits=None,
        )
    receipt = session.close()
    assert isinstance(receipt, g.ObservationReceipt)
    diagnosis = g.diagnose_observations(
        observations=receipt,
        limits=g.InspectionLimits(
            max_pairs=2,
            max_events=4,
            max_nodes=200,
            max_depth=12,
            max_encoded_bytes=8192,
            max_count=1_000_000_000,
            max_reason_counts=17,
        ),
    )
    assert diagnosis.terminal_unknown and not diagnosis.observation_failed
    assert any(item.reason == "invalid_input" for item in diagnosis.reasons)
