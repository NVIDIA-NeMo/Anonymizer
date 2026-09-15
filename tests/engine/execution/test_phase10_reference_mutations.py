# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

import pytest

from tests.engine.execution.phase10_reference_model import (
    REFERENCE_MUTATION_INSTANCES,
    canonical_corpus_bytes,
    case_by_name,
    reduce_reference,
)

_Mutation = tuple[str, tuple[tuple[str, str], ...], str | None, str]
_MUTATIONS: tuple[_Mutation, ...] = (
    (
        "authorization-bypass",
        (('if case.grant_state != "valid":', 'if False and case.grant_state != "valid":'),),
        "authorization-forged",
        "result",
    ),
    (
        "unknown-operation-acceptance",
        (
            (
                "if case.operation not in OPERATIONS or case.subject_kind not in SUBJECTS:",
                "if case.subject_kind not in SUBJECTS:",
            ),
            (
                "if case.subject_kind not in _OPERATION_SUBJECTS[case.operation]:",
                "if case.subject_kind not in _OPERATION_SUBJECTS.get(case.operation, {case.subject_kind}):",
            ),
        ),
        "unknown-operation",
        "result",
    ),
    (
        "operation-subject-union",
        (
            (
                "if case.subject_kind not in _OPERATION_SUBJECTS[case.operation]:",
                "if False and case.subject_kind not in _OPERATION_SUBJECTS[case.operation]:",
            ),
        ),
        "operation-subject-inspect-admitted_plan",
        "result",
    ),
    (
        "limit-bypass",
        (("if not _within_construction_limits(case):", "if False and not _within_construction_limits(case):"),),
        "limit-max_builder_working_bytes-over",
        "result",
    ),
    (
        "redaction-bypass",
        (("if case.unsafe_value:", "if False and case.unsafe_value:"),),
        "redaction-unknown-value",
        "result",
    ),
    (
        "synchronous-publication-rejected",
        (
            (
                'ReferenceCase("synchronous-return-publication"),',
                'ReferenceCase("synchronous-return-publication", unsafe_value=True),',
            ),
        ),
        "synchronous-return-publication",
        "result",
    ),
    (
        "capture-coherence-bypass",
        (
            (
                "if case.subject_kind in _SUBJECT_BOUNDARIES and not _valid_capture(case):",
                "if False and case.subject_kind in _SUBJECT_BOUNDARIES and not _valid_capture(case):",
            ),
        ),
        "capture-post_dispatch-opened",
        "result",
    ),
    (
        "empty-diagnosis-publishes",
        (
            (
                'if case.operation == "diagnose" and case.subject_kind != "admission_rejection_receipt" and not case.diagnostics:',
                "if False:",
            ),
        ),
        "diagnose-no-reasons",
        "result",
    ),
    (
        "malformed-row-acceptance",
        (("if not _valid_rows(case):", "if False and not _valid_rows(case):"),),
        "unknown-stage",
        "result",
    ),
    (
        "count-bucket-upper-edge",
        (("if value <= 4:", "if value < 4:"),),
        "count-bucket-4",
        "result",
    ),
    (
        "byte-bucket-upper-edge",
        (("if value <= 256:", "if value < 256:"),),
        "byte-bucket-256",
        "result",
    ),
    (
        "terminal-precedence-inversion",
        (("TERMINALS.index(item[1])", "-TERMINALS.index(item[1])"),),
        "terminal-order",
        "result",
    ),
    (
        "reason-order-inversion",
        (("REASONS.index(item.reason_category)", "-REASONS.index(item.reason_category)"),),
        "diagnostic-order",
        "result",
    ),
    (
        "noncompact-corpus-json",
        (('separators=(",", ":"),\n    ).encode("utf-8")', 'separators=(", ", ": "),\n    ).encode("utf-8")'),),
        None,
        "corpus",
    ),
    (
        "partial-rejection-payload",
        (
            (
                "        None,\n        measurement,\n        canonical_json_utf8_bytes,",
                '        "{}",\n        measurement,\n        canonical_json_utf8_bytes,',
            ),
        ),
        "authorization-forged",
        "result",
    ),
    (
        "noninterference-claim-dropped",
        (("protection_unchanged: bool = True", "protection_unchanged: bool = False"),),
        "reason-backend_failed",
        "result",
    ),
    (
        "exact-limit-off-by-one",
        (("0 <= value <= _LIMITS[name]", "0 <= value < _LIMITS[name]"),),
        "limit-max_builder_working_bytes-exact",
        "result",
    ),
    (
        "payload-root-limit-bypass",
        (
            (
                'if len(root) > _LIMITS["max_top_level_fields"]:',
                'if False and len(root) > _LIMITS["max_top_level_fields"]:',
            ),
        ),
        "limit-max_top_level_fields-over",
        "result",
    ),
    (
        "payload-provenance-limit-bypass",
        (
            (
                'if provenance_fields > _LIMITS["max_provenance_fields"]:',
                'if False and provenance_fields > _LIMITS["max_provenance_fields"]:',
            ),
        ),
        "limit-max_provenance_fields-over",
        "result",
    ),
    (
        "payload-depth-limit-bypass",
        (
            (
                'if type(value) is dict:\n        if container_depth > _LIMITS["max_json_nesting_depth"]:',
                'if type(value) is dict:\n        if False and container_depth > _LIMITS["max_json_nesting_depth"]:',
            ),
        ),
        "limit-max_json_nesting_depth-over",
        "result",
    ),
    (
        "payload-list-depth-limit-bypass",
        (
            (
                'if type(value) is list:\n        if container_depth > _LIMITS["max_json_nesting_depth"]:',
                'if type(value) is list:\n        if False and container_depth > _LIMITS["max_json_nesting_depth"]:',
            ),
        ),
        "limit-max_json_nesting_depth-list-over",
        "result",
    ),
    (
        "payload-string-limit-bypass",
        (
            (
                'if string_bytes > _LIMITS["max_allowlisted_string_utf8_bytes"]:',
                'if False and string_bytes > _LIMITS["max_allowlisted_string_utf8_bytes"]:',
            ),
        ),
        "limit-max_allowlisted_string_utf8_bytes-over",
        "result",
    ),
    (
        "payload-key-string-limit-bypass",
        (
            (
                'if key_bytes > _LIMITS["max_allowlisted_string_utf8_bytes"]:',
                'if False and key_bytes > _LIMITS["max_allowlisted_string_utf8_bytes"]:',
            ),
        ),
        "limit-max_allowlisted_string_utf8_bytes-key-over",
        "result",
    ),
    (
        "canonical-byte-limit-bypass",
        (("if encoded_bytes > canonical_limit:", "if False and encoded_bytes > canonical_limit:"),),
        "limit-max_canonical_json_utf8_bytes-over",
        "result",
    ),
    (
        "derived-root-measurement-corruption",
        (
            (
                "return ReferencePayloadMeasurement(len(root), provenance_fields, nesting_depth, longest_string)",
                "return ReferencePayloadMeasurement(0, provenance_fields, nesting_depth, longest_string)",
            ),
        ),
        "limit-max_top_level_fields-exact",
        "result",
    ),
    (
        "fixed-arity-request-counter-introduction",
        (
            (
                "    rejection_category: str | None = None\n",
                "    rejection_category: str | None = None\n    subjects_per_call: int = 1\n",
            ),
        ),
        None,
        "corpus",
    ),
    (
        "closed-domain-admission-bypass",
        (("if not _valid_value_domains(case):", "if False and not _valid_value_domains(case):"),),
        "unknown-semantic-profile",
        "result",
    ),
    (
        "measurement-schema-precedence-inversion",
        (
            (
                """    payload = (
        structural_payload_witness(case.payload_witness) if case.payload_witness is not None else _payload(case, count)
    )
    measured = measure_reference_payload(payload)
    if isinstance(measured, str):
        return _rejected(measured, True, count, bytes_)
    if case.unsafe_value:
        return _rejected(\"inspection_redaction_failed\", True, count, bytes_, measured)""",
                """    if case.unsafe_value:
        return _rejected(\"inspection_redaction_failed\", True, count, bytes_)
    payload = (
        structural_payload_witness(case.payload_witness) if case.payload_witness is not None else _payload(case, count)
    )
    measured = measure_reference_payload(payload)
    if isinstance(measured, str):
        return _rejected(measured, True, count, bytes_)""",
            ),
        ),
        "measurement-before-unsafe-schema",
        "result",
    ),
)
_MUTATIONS += (
    (
        "admission-boundary-domain-bypass",
        (("        and case.capture_boundary in CAPTURE_BOUNDARIES\n", ""),),
        "unknown-admission-capture-boundary",
        "result",
    ),
    (
        "admission-lifecycle-domain-bypass",
        (("        and case.lifecycle_state in LIFECYCLE_STATES\n", ""),),
        "unknown-admission-lifecycle",
        "result",
    ),
    (
        "admission-rejection-domain-bypass",
        (("        and (case.rejection_category is None or case.rejection_category in REASONS)\n", ""),),
        "unknown-explain-rejection-category",
        "result",
    ),
)


@pytest.mark.parametrize("mutation", _MUTATIONS, ids=lambda mutation: mutation[0])
def test_frozen_phase10_corpus_kills_every_reference_mutation(
    mutation: _Mutation,
    tmp_path: Path,
) -> None:
    name, replacements, witness, probe = mutation
    mutant = _load_mutant(name, replacements, tmp_path)
    if probe == "corpus":
        assert mutant.canonical_corpus_bytes() != canonical_corpus_bytes()
        return
    assert witness is not None
    baseline = asdict(reduce_reference(case_by_name(witness)))
    observed = asdict(mutant.reduce_reference(mutant.case_by_name(witness)))

    assert observed != baseline, f"required Phase 10 reference mutation survived: {name}"


def test_reference_mutation_inventory_is_complete_unique_and_exact() -> None:
    names = tuple(name for name, _replacements, _witness, _probe in _MUTATIONS)

    assert names == REFERENCE_MUTATION_INSTANCES
    assert len(names) == len(set(names)) == 31
    assert {probe for _name, _replacements, _witness, probe in _MUTATIONS} == {"corpus", "result"}


def test_reference_mutation_inventory_covers_closed_domains_and_failure_precedence() -> None:
    assert {
        "closed-domain-admission-bypass",
        "measurement-schema-precedence-inversion",
    } <= set(REFERENCE_MUTATION_INSTANCES)


def _load_mutant(name: str, replacements: tuple[tuple[str, str], ...], tmp_path: Path) -> ModuleType:
    source_path = Path(__file__).with_name("phase10_reference_model.py")
    source = source_path.read_text(encoding="utf-8")
    for original, replacement in replacements:
        assert source.count(original) == 1, f"mutation seam drifted for {name}"
        source = source.replace(original, replacement)
    mutant_path = tmp_path / f"phase10_reference_model_{name.replace('-', '_')}.py"
    mutant_path.write_text(source, encoding="utf-8")
    module_name = f"tests.engine.execution._phase10_mutant_{name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, mutant_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
