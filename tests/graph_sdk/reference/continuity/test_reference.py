# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

from tests.graph_sdk.reference.continuity import boundaries, reference
from tests.graph_sdk.reference.continuity.boundaries import (
    State,
    artifact,
    canonical,
    copy_state,
    decode_small,
    lifetime,
    protocol,
)
from tests.graph_sdk.reference.continuity.cases import build, generated, known
from tests.graph_sdk.reference.continuity.reference import (
    Bundle,
    Entry,
    _fixture_bundle,
    deletion_rounds,
    feasible,
    greatest,
    groups,
    subsets,
)


def checks(module: ModuleType) -> None:
    for case, expected in known():
        # Classes remain those of the original module; mutate function globals only.
        assert module.greatest(case.bundle, case.conflicts) == expected, case.name


def test_known_answers() -> None:
    checks(reference)


@pytest.mark.parametrize("case", generated(), ids=lambda case: case.name)
def test_closure_laws(case: object) -> None:
    from tests.graph_sdk.reference.continuity.cases import Case

    assert isinstance(case, Case)
    bundle, conflicts = case.bundle, case.conflicts
    names = frozenset(t.name for t in bundle.targets)
    feasible_sets = tuple(s for s in subsets(names) if feasible(bundle, s, conflicts))
    winner = greatest(bundle, conflicts)
    assert feasible(bundle, winner, conflicts)
    assert all(s <= winner for s in feasible_sets)
    assert all(feasible(bundle, a | b, conflicts) for a in feasible_sets for b in feasible_sets)
    rounds = deletion_rounds(bundle, conflicts)
    assert rounds[-1] == winner
    assert all(after < before for before, after in zip(rounds, rounds[1:], strict=False))
    # A feasible set remains feasible in a larger candidate universe: adding support
    # opportunities cannot destroy its witnesses. Deletion cannot create one.
    for small in subsets(names):
        for large in subsets(names):
            if small <= large:
                small_bundle = _fixture_bundle(
                    tuple(t for t in bundle.targets if t.name in small),
                    bundle.decisions,
                    bundle.verified,
                )
                large_bundle = _fixture_bundle(
                    tuple(t for t in bundle.targets if t.name in large),
                    bundle.decisions,
                    bundle.verified,
                )
                assert greatest(small_bundle, conflicts) <= greatest(large_bundle, conflicts)


def test_authenticity_and_accepted_bundle_consistency() -> None:
    with pytest.raises(TypeError):
        greatest(cast("Bundle", {"accepted": True}))
    with pytest.raises(ValueError, match="factory provenance"):
        Bundle((), (), (), _seal=object())
    base = build(consumes_y=True)
    with pytest.raises(ValueError, match="consumption"):
        _fixture_bundle(
            (replace(base.targets[0], required=frozenset({"x"})),),
            base.decisions,
            base.verified,
        )
    with pytest.raises(ValueError, match="foreign"):
        _fixture_bundle(base.targets, base.decisions, ())


def test_transitions_groups_and_projection() -> None:
    current = Entry("x", 2, "V0", ("E0",))
    assert reference.key_conflict("create", None, current, "V0", ("E0",))
    assert not reference.key_conflict("create", None, None, "V0", ("E0",))
    assert not reference.key_conflict("reuse", 2, current, "V0", ("E0",))
    assert reference.key_conflict("reuse", 1, current, "V0", ("E0",))
    assert reference.key_conflict("reuse", 2, current, "V1", ("E0",))
    assert reference.key_conflict("reuse", 2, current, "V0", ("E1",))
    assert not reference.key_conflict("supersede", 2, current, "V1", ("E1",))
    assert groups(("r0.name", "r0.email", "r1.name")) == (
        ("r0.name",),
        ("r0.email",),
        ("r1.name",),
    )
    assert groups(("A", "B"), (("A", "B"),)) == (("A", "B"),)
    for bad in ((("A",),), (("A", "A", "B"),), (("A", "B", "C"),)):
        with pytest.raises(ValueError):
            groups(("A", "B"), bad)
    allowed = reference.protected_allowed
    assert allowed(frozenset(), True, "E0", "Q0", None, "A")
    assert not allowed(frozenset({"x"}), True, "E0", "Q0", None, "A")
    assert allowed(frozenset({"x"}), True, "E0", "Q0", ("E0", "Q0", frozenset({"A"})), "A")
    for receipt in (
        ("E1", "Q0", frozenset({"A"})),
        ("E0", "Q1", frozenset({"A"})),
        ("E0", "Q0", frozenset()),
    ):
        assert not allowed(frozenset(), True, "E0", "Q0", receipt, "A")
    assert not allowed(frozenset(), False, "E0", "Q0", None, "A")


def snapshot_fixture() -> boundaries.JSON:
    return {
        "format": "nemo-anonymizer-continuity",
        "version": 2,
        "namespace": "H1",
        "revision": 1,
        "decisions": [
            {
                "group": "G0",
                "subject": {"kind": "subject", "name": "S0", "version": 1},
                "slot": "name",
                "revision": 2,
                "value": {"tag": "text", "text": "é"},
                "supports": [{"kind": "assessment", "name": "E0", "version": 1}],
            }
        ],
    }


def boundary_checks(module: ModuleType) -> None:
    assert module.protocol(
        ("reserve", "dispatch", "trusted_stop", "cancelled", "settle"),
        caller_owned=True,
    )
    assert not module.protocol(("reserve", "dispatch", "cancelled", "settle"), caller_owned=True)
    assert not module.protocol(("reserve", "cancelled", "unknown"), caller_owned=True)
    assert not module.protocol(("reserve", "cancelled", "cancelled"), caller_owned=True)
    assert not module.protocol(
        ("reserve", "dispatch", "success", "settle", "close", "close"),
        caller_owned=False,
    )
    raw = module.canonical(snapshot_fixture())
    assert module.decode_small(raw, len(raw)) == snapshot_fixture()
    with pytest.raises(ValueError):
        module.decode_small(raw, len(raw) - 1)
    for bad in (
        {"tag": "text", "text": "x", "extra": 0},
        {"tag": "text", "text": None},
        {"tag": "text", "text": "\ud800"},
        {"tag": "collection", "items": [{"key": 0, "version": 0, "text": ""}]},
        {"tag": "collection", "items": [{"key": True, "version": 1, "text": ""}]},
    ):
        with pytest.raises(ValueError):
            module.artifact(bad)
    for badraw in (
        raw.replace(b'"version":2', b'"version":1'),
        raw + b"{}",
        raw.replace(b'"namespace":"H1"', b'"namespace":"H1","namespace":"H1"'),
    ):
        with pytest.raises(ValueError):
            module.decode_small(badraw, len(badraw))


def test_snapshot_grammar_and_ownership() -> None:
    boundary_checks(boundaries)
    artifact({"tag": "text", "text": ""})
    artifact({"tag": "collection", "items": []})
    artifact({"tag": "collection", "items": [{"key": 0, "version": 1, "text": "é"}]})
    source = State("O0", "H0", 2, (2,))
    assert copy_state(source, "fork", "O1", "H1") == State("O1", "H1", 1, (2,))
    assert copy_state(source, "restore", "O1", "H0") == State("O1", "H0", 2, (2,))
    assert copy_state(source, "handoff", "O1", "H0") == State("O1", "H0", 2, (2,))
    assert copy_state(source, "fresh", "O1", "H0") == State("O1", "H0", 0, ())
    assert source == State("O0", "H0", 2, (2,))
    assert lifetime("per_call", None) and lifetime("across_calls", "O0")
    assert not lifetime("per_call", "O0") and not lifetime("across_calls", None)
    assert canonical(decode_small(canonical(snapshot_fixture()), 1000)) == canonical(snapshot_fixture())


@pytest.mark.parametrize(
    ("events", "owned", "expected"),
    [
        (("reserve", "dispatch", "success", "settle", "close"), False, True),
        (("reserve", "dispatch", "failure", "settle"), True, True),
        (("reserve", "dispatch", "lost", "settle", "close"), False, True),
        (("reserve", "cancelled"), True, True),
        (("dispatch", "reserve", "success"), True, False),
        (("reserve", "dispatch", "lost", "success", "settle"), True, False),
        (("reserve", "dispatch", "lost", "close", "settle"), False, False),
        (("reserve", "dispatch", "success", "settle", "close"), True, False),
    ],
)
def test_remote_causal_boundary(events: tuple[str, ...], owned: bool, expected: bool) -> None:
    assert protocol(events, caller_owned=owned) is expected


MUTANTS = (
    (
        "boundaries",
        '"cancelled" in events and "dispatch" in events',
        "False",
    ),
    ("boundaries", "not set(events) <= alphabet", "False"),
    ("boundaries", "len(events) != len(set(events))", "False"),
    ("reference", "evidence.observed_revision == evidence.current_revision", "True"),
    ("reference", "if required & conflicts:", "if False:"),
    ("reference", "t.prerequisites | t.atomic", "t.atomic"),
    ("reference", "t.prerequisites | t.atomic", "t.prerequisites"),
    ("reference", "evidence.reference in target.evidence", "True"),
    ("reference", "evidence.node == support.node", "True"),
    ("reference", "evidence.outcome == support.outcome", "True"),
    ("reference", "evidence.promise == support.promise", "True"),
    ("reference", "support.coverage <= evidence.coverage", "True"),
    ("reference", "key in evidence.consumed", "True"),
    (
        "reference",
        "for support in decisions[key].supports:",
        "for support in decisions[key].supports[:1]:",
    ),
    ("boundaries", "len(raw) > bound", "len(raw) > bound + 1"),
    (
        "boundaries",
        'integer(root["version"], 0) != 2',
        'integer(root["version"], 0) not in {1, 2}',
    ),
    ("boundaries", 'integer(row["version"], 1)', 'integer(row["version"], 0)'),
    ("boundaries", "type(value) is not int", "not isinstance(value, int)"),
    ("boundaries", "len({key for key, _ in pairs}) != len(pairs)", "False"),
)


@pytest.mark.parametrize(("module_name", "old", "new"), MUTANTS)
def test_semantic_mutants(tmp_path: Path, module_name: str, old: str, new: str) -> None:
    source = Path(__file__).with_name(module_name + ".py").read_text()
    assert old in source
    path = tmp_path / "mutant.py"
    path.write_text(source.replace(old, new))
    spec = importlib.util.spec_from_file_location("mutant450", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        if module_name == "reference":
            setattr(module, "Bundle", Bundle)
        with pytest.raises((AssertionError, pytest.fail.Exception)):
            (checks if module_name == "reference" else boundary_checks)(module)
    finally:
        del sys.modules[spec.name]


def test_status_revision_and_history_authority() -> None:
    targets = frozenset({"A", "B"})
    assert reference.status(targets, targets, False, frozenset()) == "no_change"
    assert reference.status(targets, targets, True, frozenset()) == "committed"
    assert reference.status(frozenset({"A"}), targets, True, frozenset({"y"})) == "partial"
    assert reference.status(frozenset(), targets, False, frozenset({"y"})) == "conflicted"
    assert reference.status(frozenset(), frozenset(), False, frozenset()) == "no_change"
    bundle = build()
    # Loaded stable support handles supply no factory-issued current evidence.
    loaded_only = _fixture_bundle(
        tuple(replace(t, evidence=frozenset()) for t in bundle.targets),
        bundle.decisions,
        (),
    )
    assert greatest(loaded_only) == frozenset({"C"})
    left = State("O0", "H0", 2, (2,))
    right = State("O1", "H0", 2, (2,))
    assert left.identity != right.identity and left.namespace == right.namespace
    # Positive whole-history read token is independent of decision revisions.
    assert left.revision + 1 == 3
    assert copy_state(left, "fork", "O2", "H1").decision_revisions == (2,)


def test_closed_snapshot_order_and_identity_exclusion() -> None:
    for items in (
        [{"key": 1, "version": 1, "text": "a"}, {"key": 0, "version": 1, "text": "b"}],
        [{"key": 0, "version": 1, "text": "a"}, {"key": 0, "version": 1, "text": "a"}],
    ):
        with pytest.raises(ValueError):
            artifact({"tag": "collection", "items": items})
    value = snapshot_fixture()
    assert isinstance(value, dict)
    for private in (
        "state_identity",
        "invocation",
        "request",
        "resource",
        "credentials",
        "raw_input",
    ):
        with pytest.raises(ValueError):
            boundaries.snapshot({**value, private: "canary"})
    assert b"state_identity" not in canonical(value)


def test_extended_closure_has_repeated_rounds() -> None:
    case, expected = next(pair for pair in known() if pair[0].name == "alternating-seven-target")
    rounds = deletion_rounds(case.bundle, case.conflicts)
    assert rounds == (
        frozenset({"A", "B", "C", "D", "E", "F", "U"}),
        frozenset({"A", "C", "D", "E", "F", "U"}),
        frozenset({"A", "C", "E", "F", "U"}),
        frozenset({"A", "E", "F", "U"}),
        frozenset({"A", "F", "U"}),
        frozenset({"A", "U"}),
    )
    assert greatest(case.bundle, case.conflicts) == expected


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        (("reserve", "dispatch", "trusted_stop", "cancelled"), True),
        (("reserve", "dispatch", "trusted_stop", "cancelled", "settle"), True),
        (("reserve", "dispatch", "cancelled", "settle"), False),
        (("reserve", "dispatch", "cancelled", "trusted_stop", "settle"), False),
        (("reserve", "dispatch", "lost", "trusted_stop", "settle"), True),
        (("reserve", "trusted_stop", "dispatch", "cancelled"), False),
        (("reserve", "trusted_stop", "cancelled"), False),
        (("reserve", "cancelled", "unknown"), False),
        (("reserve", "dispatch", "success", "settle", "close", "close"), False),
        (("reserve", "dispatch", "trusted_stop", "trusted_stop", "cancelled"), False),
    ],
)
def test_trusted_stop_alphabet(events: tuple[str, ...], expected: bool) -> None:
    assert protocol(events, caller_owned=False) is expected
