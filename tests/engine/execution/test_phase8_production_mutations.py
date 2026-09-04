# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_Mutation = tuple[str, str, tuple[tuple[str, str], ...], str]
_MUTATIONS: tuple[_Mutation, ...] = (
    (
        "positional-instead-of-keyed-presentation",
        "phase8_service.py",
        (
            (
                "released = tuple((member, candidates[member]) for member, _baseline in phase7_released if member in qualified)",
                "released = tuple((member, candidates[member]) for group in groups for member in group if member in qualified)",
            ),
        ),
        "tests/engine/execution/test_phase8_reference_conformance.py::test_production_runtime_and_phase4_match_every_stable_reference_trace[envelope-shape-3-3-0]",
    ),
    (
        "accept-partial-keys",
        "phase8_validation.py",
        (("and _has_exact_keys(expected, tuple(revisions))", "and set(revisions).issubset(set(expected))"),),
        "tests/engine/execution/test_phase8_grouped_rewrite.py::test_phase8_runtime_never_adopts_a_partial_group_repair",
    ),
    (
        "infer-missing-group-members",
        "phase8_admission.py",
        (("if seen != target_set:", "if False:"),),
        "tests/engine/execution/test_phase8_grouped_rewrite.py::test_phase8_admission_requires_one_flat_exact_target_partition",
    ),
    (
        "mixed-group-zero-route-split",
        "phase8_service.py",
        (
            (
                "and all(group_input.phase7_applied[member] is False for member in members)",
                "and any(group_input.phase7_applied[member] is False for member in members)",
            ),
        ),
        "tests/engine/execution/test_phase8_grouped_rewrite.py::test_phase8_zero_route_rejects_a_mixed_applied_and_no_entity_group",
    ),
    (
        "accept-subset-repair",
        "phase8_runtime.py",
        (
            (
                """        if not _validate_complete_revisions(members, revisions):
            fault = _Phase8OperationFault(_Phase8FaultKind.INCONSISTENT, _Phase8Reason.INCOMPLETE_GROUP)
            return _faulted(ledger, repair_stage, fault, round_number + 1)
""",
                """        if False and not _validate_complete_revisions(members, revisions):
            fault = _Phase8OperationFault(_Phase8FaultKind.INCONSISTENT, _Phase8Reason.INCOMPLETE_GROUP)
            return _faulted(ledger, repair_stage, fault, round_number + 1)
""",
            ),
        ),
        "tests/engine/execution/test_phase8_grouped_rewrite.py::test_phase8_runtime_never_adopts_a_partial_group_repair",
    ),
    (
        "skip-re-evaluation",
        "phase8_runtime.py",
        (("for round_number in range(max_repairs + 1):", "for round_number in range(1):"),),
        "tests/engine/execution/test_phase8_repair_runtime.py::test_every_repair_is_followed_by_complete_group_evaluation",
    ),
    (
        "repair-limit-off-by-one",
        "phase8_admission.py",
        (("not 0 <= max_repairs <= limit", "not 0 <= max_repairs < limit"),),
        "tests/engine/execution/test_phase8_operation_ledger.py::test_phase8_operation_plan_freezes_exact_stage_vectors",
    ),
    (
        "late-terminal-resurrection",
        "phase8_runtime.py",
        (
            (
                """    def succeed(self, stage: _Phase8Stage) -> bool:
        if stage not in self.plan.stages or stage in self._terminals:
            return False
""",
                """    def succeed(self, stage: _Phase8Stage) -> bool:
        if stage not in self.plan.stages:
            return False
""",
            ),
            (
                """    def _close(self, stage: _Phase8Stage, terminal: _Phase8Terminal) -> bool:
        if stage in self._terminals:
            return False
""",
                """    def _close(self, stage: _Phase8Stage, terminal: _Phase8Terminal) -> bool:
        if False and stage in self._terminals:
            return False
""",
            ),
        ),
        "tests/engine/execution/test_phase8_operation_ledger.py::test_terminal_is_absorbing_and_failure_blocks_descendants",
    ),
    (
        "trust-public-failed-record-id",
        "phase8_ndd_backend.py",
        (
            ("and len(result.failed_row_evidence) == 1", "and len(result.failed_row_evidence) in {0, 1}"),
            (
                "and result.failed_row_evidence[0].row_token == token",
                "and (not result.failed_row_evidence or result.failed_records[0].record_id == token)",
            ),
            (
                "and result.failed_row_evidence[0].record == result.failed_records[0]",
                "and (not result.failed_row_evidence or result.failed_row_evidence[0].record == result.failed_records[0])",
            ),
        ),
        "tests/engine/execution/test_phase8_grouped_rewrite.py::test_phase8_public_failed_record_id_without_private_binding_is_invocation_inconsistent",
    ),
    (
        "invert-failure-precedence",
        "phase8_runtime.py",
        (
            (
                "return max(non_success, key=_group_precedence) if non_success else _GroupSucceeded()",
                "return min(non_success, key=_group_precedence) if non_success else _GroupSucceeded()",
            ),
        ),
        "tests/engine/execution/test_phase8_operation_ledger.py::test_group_aggregate_uses_frozen_failure_precedence",
    ),
    (
        "release-before-cleanup-and-phase4",
        "phase8_service.py",
        (
            ("if pre.status is _Phase8CleanupStatus.FAILED:", "if False:"),
            ("elif pre.status is not _Phase8CleanupStatus.VERIFIED:", "elif False:"),
            (
                "released_cells: dict[_DatumId, _SealedCandidateCell] = {}",
                "released_cells: dict[_DatumId, _SealedCandidateCell] = dict(cells)",
            ),
            (
                "embargo = phase4_embargo or pre.status is not _Phase8CleanupStatus.VERIFIED",
                "embargo = False",
            ),
        ),
        "tests/engine/execution/test_phase8_release.py::test_phase8_pre_reduction_cleanup_failure_or_unconfirmed_evidence_embargoes_release",
    ),
    (
        "retain-private-candidate-content",
        "phase8_service.py",
        (("completed.revisions.clear()", "pass"),),
        "tests/engine/execution/test_phase8_grouped_rewrite.py::test_phase8_group_operation_cleanup_discards_candidate_evidence_and_token_authority",
    ),
    (
        "skip-dispatch-capability-snapshot",
        "phase8_ndd_backend.py",
        (("or self.phase8_capability(self._invocation) != self._compiled_capability", "or False"),),
        "tests/engine/execution/test_phase8_capability.py::test_phase8_ndd_backend_detects_prompt_or_model_drift_before_adapter_call",
    ),
    (
        "accept-retention-or-capability-drift",
        "phase8_service.py",
        (
            (
                "and _snapshot_phase8_capability(self.backend, self.invocation) == self.expected",
                "and True",
            ),
        ),
        "tests/engine/execution/test_phase8_capability.py::test_phase8_capability_drift_fails_before_dispatch_and_makes_close_unconfirmed",
    ),
    (
        "route-analyze-to-wrong-model-role",
        "phase8_ndd_backend.py",
        ((('"analyze": "disposition_analyzer"'), ('"analyze": "rewriter"')),),
        "tests/engine/execution/test_phase8_capability.py::test_phase8_operation_routes_use_exact_roles_without_alias_fallback",
    ),
    (
        "dispatch-oversized-workframe",
        "phase8_ndd_backend.py",
        (
            (
                'if len(encoded.encode()) > limits.get("max_workframe_utf8_bytes_per_operation", 0):',
                'if len(encoded.encode()) < limits.get("max_workframe_utf8_bytes_per_operation", 0):',
            ),
        ),
        "tests/engine/execution/test_phase8_capability.py::test_oversized_phase8_workframe_fails_locally_without_adapter_call",
    ),
    (
        "call-datadesigner-outside-adapter-boundary",
        "phase8_ndd_backend.py",
        (("self._adapter.run_workflow(", "self._adapter.preview("),),
        "tests/engine/execution/test_phase8_capability.py::test_phase8_ndd_dispatch_calls_only_adapter_run_workflow",
    ),
)


@pytest.mark.parametrize("mutation", _MUTATIONS, ids=lambda mutation: mutation[0])
def test_each_phase8_production_seam_mutant_is_killed(mutation: _Mutation, tmp_path: Path) -> None:
    name, relative_path, replacements, witness = mutation
    repository = Path(__file__).parents[3]
    mutant_source = tmp_path / "src"
    shutil.copytree(repository / "src" / "anonymizer", mutant_source / "anonymizer")
    target = mutant_source / "anonymizer" / "engine" / "execution" / relative_path
    source = target.read_text(encoding="utf-8")
    for original, replacement in replacements:
        assert source.count(original) == 1, f"production mutation seam drifted for {name}"
        source = source.replace(original, replacement)
    target.write_text(source, encoding="utf-8")
    environment = os.environ.copy()
    for key in tuple(environment):
        if key.startswith("COV_CORE_"):
            environment.pop(key)
    environment["PYTHONPATH"] = os.pathsep.join((str(mutant_source), environment.get("PYTHONPATH", ""))).rstrip(
        os.pathsep
    )

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", witness, "-q"],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 1, (
        f"required Phase 8 production mutation survived or did not reach an assertion: {name}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


def test_production_mutation_inventory_is_complete_and_unique() -> None:
    names = tuple(name for name, _path, _replacements, _witness in _MUTATIONS)

    assert len(names) == len(set(names)) == 17
