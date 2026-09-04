# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import hashlib
import importlib
import json
from dataclasses import fields, replace
from pathlib import Path

import pytest

from tests.engine.execution.phase8_reference_model import (
    MAX_DATUMS,
    MAX_GROUPS,
    MAX_REPAIRS,
    ReferenceCase,
    ReferenceGroup,
    canonical_corpus_bytes,
    case_by_name,
    finite_reference_cases,
    reduce_reference,
    reference_manifest,
)


def test_phase8_reference_model_has_no_production_or_runtime_imports() -> None:
    source = Path(__file__).with_name("phase8_reference_model.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}

    assert not any(
        module == forbidden or module.startswith(f"{forbidden}.")
        for module in modules
        for forbidden in ("anonymizer", "pandas", "data_designer", "datadesigner")
    )
    assert tuple(field.name for field in fields(ReferenceCase)) == (
        "name",
        "targets",
        "groups",
        "atomic_groups",
        "dependencies",
        "max_repairs",
        "strict",
        "mention_evidence",
        "context_evidence",
        "consumed_binding_evidence",
        "capability",
        "retention",
        "prompt",
        "model_route",
        "failure_evidence",
        "pre_cleanup",
        "post_cleanup",
        "workframe_bytes",
    )


def test_phase8_reference_manifest_freezes_counts_serialization_and_digest() -> None:
    frozen = json.loads(Path(__file__).with_name("phase8_reference_manifest.json").read_text(encoding="utf-8"))
    generated = reference_manifest()

    assert generated == frozen
    assert generated["reference_model_version"] == "phase8-grouped-rewrite-reference-model/v1"
    assert generated["generator_version"] == "phase8-finite-envelope/v1"
    assert generated["case_count"] == 167
    assert generated["canonical_trace_count"] == 167
    assert generated["structural_envelope_case_count"] == 26
    assert generated["dependency_envelope_case_count"] == 74
    assert generated["repair_envelope_case_count"] == 9
    assert generated["directed_case_count"] == 58
    assert generated["corpus_sha256"] == hashlib.sha256(canonical_corpus_bytes()).hexdigest()
    assert not canonical_corpus_bytes().endswith(b"\n")


def test_corpus_serialization_is_invariant_to_case_order(monkeypatch: pytest.MonkeyPatch) -> None:
    model = importlib.import_module("tests.engine.execution.phase8_reference_model")
    baseline = canonical_corpus_bytes()
    cases = finite_reference_cases()

    monkeypatch.setattr(model, "finite_reference_cases", lambda: tuple(reversed(cases)))

    assert canonical_corpus_bytes() == baseline


def test_finite_envelope_covers_every_bounded_shape_dependency_and_repair_axis() -> None:
    cases = finite_reference_cases()
    structural = [case for case in cases if case.name.startswith("envelope-shape-")]
    dependencies = [case for case in cases if case.name.startswith("envelope-dag-")]
    repairs = [case for case in cases if case.name.startswith("envelope-repair-")]

    assert {len(case.targets) for case in structural} == set(range(1, MAX_DATUMS + 1))
    assert {len(case.groups) for case in structural} == set(range(1, MAX_GROUPS + 1))
    assert {len(case.targets) for case in dependencies} == {2, 3, 4}
    assert {case.max_repairs for case in repairs} == set(range(MAX_REPAIRS))
    assert all(reduce_reference(case).admission == "admitted" for case in structural + dependencies + repairs)


def test_group_results_are_keyed_complete_or_fail_closed() -> None:
    valid = reduce_reference(case_by_name("valid-group"))
    partial = reduce_reference(case_by_name("partial-result"))
    extra = reduce_reference(case_by_name("extra-result"))
    subset_repair = reduce_reference(case_by_name("subset-repair"))

    assert valid.released == ("a", "b")
    assert partial.group_states == ("inconsistent",) and partial.released == ()
    assert extra.group_states == ("inconsistent",) and extra.released == ()
    assert subset_repair.group_states == ("inconsistent",) and subset_repair.released == ()


def test_repair_requires_initial_evaluation_complete_rounds_and_re_evaluation() -> None:
    assert reduce_reference(case_by_name("repair-pass")).group_states == ("succeeded",)
    assert reduce_reference(case_by_name("repair-exhausted")).group_states == ("failed",)
    assert reduce_reference(case_by_name("skipped-evaluation")).group_states == ("inconsistent",)
    assert reduce_reference(case_by_name("directed-three-repairs-pass")).released == ("a",)
    assert reduce_reference(case_by_name("fourth-repair-rejected")).admission == "rejected"


def test_failure_isolation_phase4_closure_and_global_embargo_are_distinct() -> None:
    local = reduce_reference(case_by_name("local-failed-disconnected"))
    atomic = reduce_reference(case_by_name("atomic-sibling-withheld"))
    dependent = reduce_reference(case_by_name("dependency-closure"))
    global_fault = reduce_reference(case_by_name("global-inconsistent-stops"))

    assert local.released == ("b",)
    assert atomic.released == ()
    assert dependent.released == ()
    assert global_fault.group_states == ("inconsistent", "blocked")
    assert global_fault.released == ()


@pytest.mark.parametrize("name", ["cancel-stops", "loss-stops", "global-inconsistent-stops"])
def test_invocation_global_terminals_stop_later_groups(name: str) -> None:
    result = reduce_reference(case_by_name(name))

    assert result.group_states[1:] == ("blocked",)
    assert result.released == ()


def test_first_terminal_is_absorbing_and_failure_precedence_is_frozen() -> None:
    absorbed = reduce_reference(case_by_name("late-success-absorbed"))
    failed_and_inconsistent = ReferenceCase(
        "failed-and-inconsistent",
        ("a", "b"),
        (
            ReferenceGroup(("a",), ("failed",)),
            ReferenceGroup(("b",), ("inconsistent",)),
        ),
        (("a",), ("b",)),
    )
    blocked_and_failed = replace(
        failed_and_inconsistent,
        name="blocked-and-failed",
        groups=(ReferenceGroup(("a",), ("blocked",)), ReferenceGroup(("b",), ("failed",))),
    )

    assert absorbed.group_states == ("failed",)
    assert reduce_reference(failed_and_inconsistent).aggregate == "inconsistent"
    assert reduce_reference(blocked_and_failed).aggregate == "failed"


@pytest.mark.parametrize(
    "name",
    [
        "strict-false",
        "mention-missing",
        "mention-duplicate",
        "mention-foreign",
        "mention-wrong_owner",
        "context-owner_swap",
        "context-ordinal_swap",
        "context-flattened",
        "consumed-binding-missing",
        "consumed-binding-duplicate",
        "consumed-binding-foreign",
        "capability-missing",
        "capability-drift",
        "retention-unknown",
        "retention-enabled",
        "prompt-drift",
        "model-config-drift",
        "wrong-model-role",
        "fallback-model-route",
    ],
)
def test_invalid_authority_or_reconciliation_evidence_never_releases(name: str) -> None:
    assert reduce_reference(case_by_name(name)).released == ()


def test_cleanup_failure_and_unconfirmed_evidence_have_distinct_precedence() -> None:
    assert reduce_reference(case_by_name("cleanup-pre-failed")).invocation == "failed"
    assert reduce_reference(case_by_name("cleanup-post-failed")).invocation == "failed"
    assert reduce_reference(case_by_name("cleanup-pre-missing")).invocation == "inconsistent"
    assert reduce_reference(case_by_name("cleanup-post-contradictory")).invocation == "inconsistent"


def test_opaque_renaming_and_group_presentation_preserve_normalized_semantics() -> None:
    baseline = case_by_name("local-failed-disconnected")
    renamed = ReferenceCase(
        "renamed",
        ("x", "y"),
        (ReferenceGroup(("x",), ("failed",)), ReferenceGroup(("y",))),
        (("x",), ("y",)),
    )
    reordered = replace(baseline, name="reordered", groups=tuple(reversed(baseline.groups)))

    assert reduce_reference(renamed).group_states == reduce_reference(baseline).group_states
    assert reduce_reference(renamed).released == ("y",)
    assert reduce_reference(reordered) == replace(reduce_reference(baseline), reason=None)


def test_withholding_is_monotone_when_an_atomic_group_or_dependency_is_added() -> None:
    local = reduce_reference(case_by_name("local-failed-disconnected"))
    atomic = reduce_reference(case_by_name("atomic-sibling-withheld"))
    dependent = reduce_reference(case_by_name("dependency-closure"))

    assert set(atomic.released) <= set(local.released)
    assert set(dependent.released) <= set(local.released)
