# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path

from tests.graph_sdk.reference.evidence_v1.p7_reference import model
from tests.graph_sdk.reference.evidence_v1.p7_reference.fixtures import (
    build_raw,
    change_revision,
    enumerate_cases,
    write_bundle,
)


def assert_rejected(module: types.ModuleType, raw: model.Raw) -> None:
    try:
        module.evaluate(raw)
    except ValueError:
        return
    raise AssertionError("invalid exact-owner join accepted")


def assert_answers(module: types.ModuleType) -> None:
    raw = build_raw()
    assert_rejected(module, replace(raw, submissions=(replace(raw.assessments[0]),)))
    shifted = raw.ports[0]._replace(occurrence=("I0", "A", "NA", 1))
    assert_rejected(module, replace(raw, ports=(shifted, *raw.ports[1:])))
    ancestry = build_raw(topology="ancestry")
    wrong_parent = ancestry.provenance[0]._replace(parents=(ancestry.provenance[-1].producer,))
    assert_rejected(module, replace(ancestry, provenance=(wrong_parent, *ancestry.provenance[1:])))
    assert module.evaluate(raw).released == frozenset("ABC")
    assert module.evaluate(raw).decisions == frozenset()
    only_b = replace(raw, submissions=(raw.assessments[1],))
    assert module.evaluate(only_b).released == frozenset("B")
    assert module.evaluate(replace(raw, submissions=())).released == frozenset()
    for selected, expected in (("A", "current"), ("B", "stale"), ("AB", "stale")):
        changed = change_revision(build_raw(selected), "artifact", ("I0", "XB"), "stale")
        assert module.evaluate(changed).verified[0].validity == expected
    for kind, key in (
        ("artifact", ("I0", "A")),
        ("absence", ("I0", "QA")),
        ("configuration", ("NA",)),
        ("state", ("SA",)),
    ):
        stale = change_revision(raw, kind, key, "stale")
        assert module.evaluate(stale).verified[0].validity == "stale"
        unknown = change_revision(raw, kind, key, "unknown")
        assert module.evaluate(unknown).verified[0].validity == "unknown"
    both = change_revision(change_revision(raw, "artifact", ("I0", "XB"), "stale"), "state", ("SA",), "unknown")
    assert module.evaluate(both).verified[0].validity == "stale"
    assert module.evaluate(change_revision(raw, "artifact", ("I0", "Z"), "stale")).released == frozenset("ABC")
    partial = raw.productions[0]._replace(coverage=frozenset(("K1",)))
    assert module.evaluate(replace(raw, productions=(partial, *raw.productions[1:]))).released == frozenset("BC")
    for finding in ("unsatisfied", "unknown"):
        fact = replace(raw.assessments[0], finding=finding)
        bad = replace(raw, assessments=(fact, *raw.assessments[1:]), submissions=(fact, *raw.assessments[1:]))
        assert module.evaluate(bad).released == frozenset("BC")
    for topology, expected in (("independent", "BC"), ("chain", ""), ("mixed", ""), ("reverse", "BC")):
        changed = change_revision(build_raw(topology=topology), "state", ("SA",), "stale")
        assert module.evaluate(changed).released == frozenset(expected)
    atomic_then_datum = replace(raw, dependencies=(("B", "C"),), atomic=(frozenset("AB"), frozenset("C")))
    atomic_then_datum = change_revision(atomic_then_datum, "state", ("SA",), "stale")
    assert module.evaluate(atomic_then_datum).released == frozenset()
    assert module.evaluate(build_raw(topology="ancestry")).decisions == frozenset((("I0", "D", 1),))
    assert module.evaluate(build_raw("D")).decisions == frozenset((("I0", "D", 1),))
    withheld = change_revision(build_raw(topology="ancestry"), "state", ("SA",), "stale")
    assert module.evaluate(withheld).decisions == frozenset()


class ReferenceTests(unittest.TestCase):
    def test_known_answers(self) -> None:
        assert_answers(model)

    def test_exact_retained_joins(self) -> None:
        raw = build_raw()
        corruptions = [
            replace(raw, submissions=(replace(raw.assessments[0]),)),
            replace(raw, submissions=(raw.assessments[0], raw.assessments[0])),
            replace(raw, assessments=raw.assessments[:-1]),
            replace(raw, terminals=raw.terminals[:-1]),
        ]
        occurrences: tuple[model.Occurrence, ...] = (
            ("FOREIGN", "A", "NA", 0),
            ("I0", "FOREIGN", "NA", 0),
            ("I0", "A", "FOREIGN", 0),
            ("I0", "A", "NA", 1),
        )
        for occurrence in occurrences:
            port = raw.ports[0]._replace(occurrence=occurrence)
            corruptions.append(replace(raw, ports=(port, *raw.ports[1:])))
        for ref in (("I1", "A", 1), ("I0", "B", 1), ("I0", "A", 2)):
            corruptions.append(replace(raw, ports=(raw.ports[0]._replace(artifact=ref), *raw.ports[1:])))
        corruptions.append(replace(raw, ports=(raw.ports[0]._replace(name="other"), *raw.ports[1:])))
        corruptions.append(
            replace(raw, finals=(raw.finals[0]._replace(producer=raw.finals[1].producer), *raw.finals[1:]))
        )
        for changed in corruptions:
            with self.assertRaises(ValueError):
                model.evaluate(changed)

    def test_provenance_integrity(self) -> None:
        raw = build_raw(topology="ancestry")
        wrong = raw.provenance[0]._replace(parents=(raw.provenance[-1].producer,))
        with self.assertRaises(ValueError):
            model.evaluate(replace(raw, provenance=(wrong, *raw.provenance[1:])))
        with self.assertRaises(ValueError):
            model.evaluate(replace(raw, provenance=raw.provenance[:-2]))

    def test_type_sensitive_and_unrelated_environment(self) -> None:
        expected = (model.Revision("configuration", ("N",), 1),)
        self.assertEqual(model.classify_validity(expected, (model.Revision("configuration", ("N",), True),)), "stale")
        raw = build_raw()
        extra = model.Revision("state", ("UNRELATED",), 9)
        self.assertEqual(model.evaluate(raw), model.evaluate(replace(raw, current=(*raw.current, extra))))

    def test_union_closure_and_independent_commutation(self) -> None:
        raw = build_raw()
        left = change_revision(change_revision(raw, "state", ("SA",), "stale"), "absence", ("I0", "QB"), "unknown")
        right = change_revision(change_revision(raw, "absence", ("I0", "QB"), "unknown"), "state", ("SA",), "stale")
        self.assertEqual(model.evaluate(left), model.evaluate(right))
        for topology in ("independent", "chain", "mixed", "reverse", "ancestry"):
            candidate = build_raw(topology=topology)
            for eligible in (frozenset(), frozenset("A"), frozenset("BC"), frozenset("ABC")):
                released = model.enumerate_release(candidate, eligible)
                self.assertLessEqual(released, eligible)
                self.assertTrue(all(b not in released or a in released for a, b in candidate.dependencies))
                self.assertTrue(all(not released & group or group <= released for group in candidate.atomic))

    def test_counts_and_regeneration(self) -> None:
        self.assertEqual(len(list(enumerate_cases())), 1680)
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            write_bundle(Path(a))
            write_bundle(Path(b))
            for name in ("cases.jsonl", "manifest.json"):
                self.assertEqual((Path(a) / name).read_bytes(), (Path(b) / name).read_bytes())

    def test_semantic_mutants(self) -> None:
        source = Path(model.__file__).read_text()
        mutants = {
            "accept_copied_fact": ("if not any(fact is retained for retained in raw.assessments):", "if False:"),
            "ignore_occurrence": ("if p.occurrence == fact.occurrence", "if p.occurrence[1] == fact.occurrence[1]"),
            "ignore_declared_parent": ("if row.parents != select_one(", "if False and row.parents != select_one("),
            "stale_as_unknown": ('"stale" if "stale" in statuses', '"unknown" if "stale" in statuses'),
            "ignore_absence": (
                "for dependency in expected:",
                'for dependency in expected:\n        if dependency.kind == "absence":\n            continue',
            ),
            "ignore_coverage": ("v.coverage >= r.coverage", "True"),
            "ignore_finding": ('and v.finding == "satisfied"', ""),
            "ignore_dependencies": ("dependent not in s or prerequisite in s", "True"),
            "ignore_atomic": ("not (s & group) or group <= s", "True"),
            "lose_ancestry": ("for parent in row.parents:", "for parent in ():"),
            "include_unrelated_decisions": ("if p.producer in visited and p.decision", "if p.decision"),
            "lose_consumed_decisions": ("for d in v.decisions", "for d in ()"),
        }
        for name, (old, new) in mutants.items():
            with self.subTest(name=name):
                self.assertGreater(source.count(old), 0)
                module = types.ModuleType(name)
                # Dataclass annotations need an importable module during class construction.

                sys.modules[name] = module
                try:
                    exec(compile(source.replace(old, new), name, "exec"), module.__dict__)
                    with self.assertRaises(AssertionError):
                        assert_answers(module)
                finally:
                    del sys.modules[name]


if __name__ == "__main__":
    unittest.main(verbosity=2)
