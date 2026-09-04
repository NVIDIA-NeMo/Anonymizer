# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

import pytest

from tests.engine.execution.phase8_reference_model import case_by_name, reduce_reference

_Mutation = tuple[str, tuple[tuple[str, str], ...], str]
_MUTATIONS: tuple[_Mutation, ...] = (
    (
        "positional-group-order",
        (
            (
                "tuple(sorted(case.groups, key=lambda group: min(case.targets.index(key) for key in group.members)))",
                "tuple(case.groups)",
            ),
        ),
        "declared-order-terminal-normalized",
    ),
    (
        "partial-key-acceptance",
        (
            (
                "if set(result_keys) != set(group.members) or len(result_keys) != len(group.members):",
                "if set(result_keys) - set(group.members):",
            ),
        ),
        "partial-result",
    ),
    (
        "inferred-group-coverage",
        (
            (
                "if len(declared) != len(set(declared)) or set(declared) != set(case.targets):",
                "if len(declared) != len(set(declared)):",
            ),
        ),
        "coverage-gap",
    ),
    (
        "row-fallback-splits-group",
        (("eligible.update(group.members)", "eligible.add(group.members[0])"),),
        "valid-group",
    ),
    (
        "subset-repair-acceptance",
        (
            (
                "if any(set(keys) != set(group.members) or len(keys) != len(group.members) for keys in group.repair_keys):",
                "if any(set(keys) - set(group.members) for keys in group.repair_keys):",
            ),
        ),
        "subset-repair",
    ),
    (
        "skipped-initial-evaluation",
        (
            (
                'if not group.evaluations:\n        return "inconsistent"',
                'if not group.evaluations:\n        return "succeeded"',
            ),
        ),
        "skipped-evaluation",
    ),
    (
        "repair-limit-off-by-one",
        (("or not 0 <= case.max_repairs <= MAX_REPAIRS", "or not 0 <= case.max_repairs < MAX_REPAIRS"),),
        "directed-three-repairs-pass",
    ),
    (
        "late-result-resurrection",
        (("first = group.terminal_events[0]", "first = group.terminal_events[-1]"),),
        "late-success-absorbed",
    ),
    (
        "failure-precedence-inversion",
        (
            (
                "max(non_success, key=_PRECEDENCE.__getitem__)",
                "min(non_success, key=_PRECEDENCE.__getitem__)",
            ),
        ),
        "precedence-failed-inconsistent",
    ),
    (
        "release-before-cleanup",
        (("cleanup = _cleanup_terminal(case.pre_cleanup, case.post_cleanup)", "cleanup = None"),),
        "cleanup-pre-failed",
    ),
    (
        "strict-false-acceptance",
        (("if not case.strict:", "if False and not case.strict:"),),
        "strict-false",
    ),
    (
        "mention-coverage-bypass",
        (('if case.mention_evidence != "exact":', 'if False and case.mention_evidence != "exact":'),),
        "mention-wrong_owner",
    ),
    (
        "context-binding-bypass",
        (
            (
                'if case.context_evidence != "exact" or case.consumed_binding_evidence != "exact":',
                'if case.context_evidence != "exact":',
            ),
        ),
        "consumed-binding-foreign",
    ),
    (
        "public-record-id-trust",
        (
            (
                'if case.failure_evidence not in {"none", "bound"}:',
                'if case.failure_evidence not in {"none", "bound", "record_id_only"}:',
            ),
        ),
        "failed-record-record-id-only",
    ),
    (
        "oversized-workframe-dispatch",
        (("if group_index == 0 and case.workframe_bytes > MAX_WORKFRAME_BYTES", "if False"),),
        "workframe-limit-over",
    ),
)


@pytest.mark.parametrize(
    ("name", "replacements", "witness"), _MUTATIONS, ids=lambda value: value if isinstance(value, str) else None
)
def test_frozen_phase8_corpus_kills_every_reference_mutation(
    name: str,
    replacements: tuple[tuple[str, str], ...],
    witness: str,
    tmp_path: Path,
) -> None:
    baseline = asdict(reduce_reference(case_by_name(witness)))
    mutant = _load_mutant(name, replacements, tmp_path)

    observed = asdict(mutant.reduce_reference(mutant.case_by_name(witness)))

    assert observed != baseline, f"required Phase 8 reference mutation survived: {name}"


def test_reference_mutation_inventory_names_each_frozen_semantic_seam_once() -> None:
    names = tuple(name for name, _replacements, _witness in _MUTATIONS)

    assert len(names) == len(set(names)) == 15


def _load_mutant(name: str, replacements: tuple[tuple[str, str], ...], tmp_path: Path) -> ModuleType:
    source_path = Path(__file__).with_name("phase8_reference_model.py")
    source = source_path.read_text(encoding="utf-8")
    for original, replacement in replacements:
        assert source.count(original) == 1, f"mutation seam drifted for {name}"
        source = source.replace(original, replacement)
    mutant_path = tmp_path / f"phase8_reference_model_{name.replace('-', '_')}.py"
    mutant_path.write_text(source, encoding="utf-8")
    module_name = f"tests.engine.execution._phase8_mutant_{name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, mutant_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
