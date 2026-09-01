# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

import pytest

from tests.engine.execution.phase7_reference_model import case_by_name, reduce_reference

_Mutation = tuple[str, tuple[tuple[str, str], ...], str]
_MUTATIONS: tuple[_Mutation, ...] = (
    (
        "representative-datum-accounting",
        (("if not set(group).issubset(eligible):", "if group and group[0] not in eligible:"),),
        "atomic-group-member-failure",
    ),
    (
        "label-remapping",
        (('"first_name": "person_given_name",', '"first_name": "person_family_name",'),),
        "future-slots-2",
    ),
    (
        "text-identity",
        (
            (
                "structural_key = (scope_index, mention.cluster, role)",
                "structural_key = (scope_index, mention.source, role)",
            ),
        ),
        "equal-text-distinct-clusters",
    ),
    (
        "position-identity",
        (
            (
                "structural_key = (scope_index, mention.cluster, role)",
                "structural_key = (scope_index, mention.datum, mention.start, role)",
            ),
        ),
        "shared-slot-reuse",
    ),
    (
        "partial-acceptance",
        (("if set(keys) != expected:", "if set(keys) - expected:"),),
        "partial-candidate",
    ),
    (
        "distinct-slot-aliasing",
        (("if canonical[left] == canonical[right]:", "if False and canonical[left] == canonical[right]:"),),
        "owner-distinct_slots_same_canonical_value",
    ),
    (
        "cascading-application",
        (
            ("for mention, slot_key in reversed(ordered):", "for mention, slot_key in ordered:"),
            (
                """        if output[mention.start : mention.end] != mention.source:
            return None
        output = output[: mention.start] + assignment_by_slot[slot_key] + output[mention.end :]
""",
                """        output = output.replace(mention.source, assignment_by_slot[slot_key])
""",
            ),
        ),
        "anchored-non-cascading-application",
    ),
    (
        "late-resurrection",
        (
            (
                'if state != "reserved" or not dispatched[scope_index] or event.attempt != attempts[scope_index]:',
                'if state not in {"reserved", "cancelled", "lost"} or not dispatched[scope_index] or event.attempt != attempts[scope_index]:',
            ),
        ),
        "late-candidate-after-loss",
    ),
    (
        "skipped-cleanup",
        (
            (
                'global_inconsistent = global_inconsistent or cleanup != "verified"',
                "global_inconsistent = global_inconsistent or False",
            ),
            (
                'and cleanup == "verified"',
                'and cleanup in {"verified", "failed", "unconfirmed", "contradictory"}',
            ),
        ),
        "cleanup-failure",
    ),
)


@pytest.mark.parametrize(
    ("name", "replacements", "witness"), _MUTATIONS, ids=lambda value: value if isinstance(value, str) else None
)
def test_frozen_phase7_corpus_kills_every_required_mutation(
    name: str,
    replacements: tuple[tuple[str, str], ...],
    witness: str,
    tmp_path: Path,
) -> None:
    baseline = asdict(reduce_reference(case_by_name(witness)))
    mutant = _load_mutant(name, replacements, tmp_path)

    try:
        observed = asdict(mutant.reduce_reference(mutant.case_by_name(witness)))
    except (AssertionError, KeyError, TypeError, ValueError):
        return

    assert observed != baseline, f"required Phase 7 mutation survived: {name}"


def _load_mutant(name: str, replacements: tuple[tuple[str, str], ...], tmp_path: Path) -> ModuleType:
    source_path = Path(__file__).with_name("phase7_reference_model.py")
    source = source_path.read_text(encoding="utf-8")
    for original, replacement in replacements:
        assert source.count(original) == 1, f"mutation seam drifted for {name}"
        source = source.replace(original, replacement)
    mutant_path = tmp_path / f"phase7_reference_model_{name.replace('-', '_')}.py"
    mutant_path.write_text(source, encoding="utf-8")
    module_name = f"tests.engine.execution._phase7_mutant_{name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, mutant_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
