# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

from tests.graph_sdk.reference.observations import reference
from tests.graph_sdk.reference.observations.cases import LIMITS, accounts, observations, pair, phase
from tests.graph_sdk.reference.observations.reference import (
    Close,
    Dispatch,
    Estimate,
    Failure,
    Limits,
    Phase,
    PublicPair,
    Receipt,
    Scope,
)


def checks(module: ModuleType) -> None:
    exact = module.account(
        (Scope(tuple(Dispatch(i, purpose, (i, i + 1)) for i, purpose in enumerate(reference.PURPOSES))),)
    )
    assert exact.total == Estimate("exact", 7)
    assert tuple(item.value for item in exact.purposes) == (1,) * 7
    assert exact.input_units == Estimate("exact", 21) and exact.output_units == Estimate("exact", 28)
    unknown = module.account((Scope((Dispatch(0, "initial", (1, 2)), Dispatch(1, "repair", None))),))
    assert unknown.input_units == Estimate("unknown", None) and unknown.output_units == Estimate("unknown", None)
    assert unknown.unknown_requests == Estimate("exact", 1)
    missing = module.account((Scope(None),))
    assert missing.total == Estimate("unknown", None)
    assert missing.input_units == Estimate("unknown", None)
    close, _ = module.observe((phase(pair()),), LIMITS)
    assert isinstance(close, Receipt)
    encoded = module.encode(close, LIMITS)
    decoded = json.loads(encoded)
    assert set(decoded) == {"schema_version", "pairs", "delivery", "delivered_pairs"}
    assert set(decoded["pairs"][0]) == {"start", "terminal"}
    assert decoded["pairs"][0]["terminal"]["duration_ns"] == 2
    assert b"canary" not in encoded
    with pytest.raises(ValueError):
        module.encode(close, replace(LIMITS, encoded_bytes=len(encoded) - 1))
    for invalid in (
        replace(pair(), start_present=False),
        replace(pair(), terminal_present=False),
        replace(pair(), reasons=("exception-canary",)),
        replace(pair(), terminal_ordinal=1),
    ):
        rejected, _ = module.observe((phase(invalid),), LIMITS)
        assert rejected == Failure("producer_contract")
    swapped, _ = module.observe(
        (phase(pair(), pair(1, parent=0), replace(pair(2, parent=0), terminal_parent=1)),),
        LIMITS,
    )
    assert swapped == Failure("producer_contract")
    need = module.requirements(2, LIMITS)
    short = Limits(1, need.events, need.nodes, need.depth, need.encoded_bytes)
    declared = replace(phase(pair()), need=module.requirements(2, short))
    rejected, delivered = module.observe((declared,), short)
    assert rejected == Failure("capacity_contract") and not delivered
    mismatch, _ = module.observe((phase(pair()),), replace(LIMITS, count=99))
    assert mismatch == Failure("capacity_contract")
    deep_need = module.requirements(12, LIMITS, logical_depth=12)
    assert deep_need.depth == 12
    facts = tuple(pair(i, parent=None if i == 0 else i - 1) for i in range(12))
    too_shallow, _ = module.observe((Phase(replace(deep_need, depth=11), facts),), LIMITS)
    assert too_shallow == Failure("capacity_contract")


def count_checks(module: ModuleType) -> None:
    limited = replace(LIMITS, count=1)
    bad, _ = module.observe(
        (Phase(module.requirements(2, limited), (pair(), pair(1))),),
        limited,
        sink_configured=True,
    )
    assert bad == Failure("capacity_contract")
    for facts in (
        (pair(), replace(pair(1), terminal_ordinal=True)),
        (pair(), pair(1), replace(pair(2, parent=1), terminal_parent=True)),
    ):
        bad, _ = module.observe((phase(*facts),), LIMITS)
        assert bad == Failure("producer_contract")


def test_count_calibration() -> None:
    count_checks(reference)


def test_literal_calibration() -> None:
    checks(reference)


@pytest.mark.parametrize("case", observations(), ids=lambda case: case.name)
def test_finite_observation_cases(case: object) -> None:
    from tests.graph_sdk.reference.observations.cases import ObservationCase

    assert isinstance(case, ObservationCase)
    close, delivered = reference.observe(
        case.phases,
        case.limits,
        case.sink_failure_at,
        sink_configured=case.sink_configured,
    )
    expected_failures = {
        "no-start": "producer_contract",
        "no-terminal": "producer_contract",
        "terminal-identity": "producer_contract",
        "terminal-boundary": "producer_contract",
        "foreign-parent": "producer_contract",
        "reason-canary": "producer_contract",
        "duplicate-correlation": "producer_contract",
        "foreign-owner": "producer_contract",
        "clock-reverse": "clock_contract",
        "duration-over": "capacity_contract",
        "phase-underforecast": "capacity_contract",
        "second-phase-underfunded": "capacity_contract",
        "empty-phase-underfunded": "capacity_contract",
        "begin-zero-pairs": "capacity_contract",
        "logical-depth-short": "capacity_contract",
        "count-one-less": "capacity_contract",
        "count-zero": "capacity_contract",
        "count-cumulative": "capacity_contract",
        "terminal-ordinal-bool": "producer_contract",
        "terminal-parent-bool": "producer_contract",
    }
    failure = "capacity_contract" if case.name.startswith("phase-short-") else expected_failures.get(case.name)
    if failure is not None:
        assert close == Failure(failure)
        assert len(delivered) == (1 if case.name == "count-cumulative" else 0)
    else:
        assert isinstance(close, Receipt)
        assert len(close.pairs) == sum(len(p.pairs) for p in case.phases)
        assert len(close.pairs) * 2 == sum(2 for p in case.phases for _ in p.pairs)
        if case.name == "sink-zero":
            assert close.delivery == "failed" and close.delivered_pairs == 0
        elif case.name == "sink-one":
            assert close.delivery == "failed" and close.delivered_pairs == 1
        assert all(type(p) is PublicPair for p in delivered)
    assert reference.encode(close, LIMITS) == reference.encode(close, LIMITS)


@pytest.mark.parametrize(("name", "scopes"), accounts())
def test_accounting_cases(name: str, scopes: tuple[Scope, ...]) -> None:
    actual = reference.account(scopes)
    if name == "missing-owner":
        assert actual.total.knowledge == "unknown"
    elif name == "missing-binding-only":
        assert actual.purposes[0] == Estimate("exact", 1)
        assert actual.purposes[1] == Estimate("unknown", None)
        assert actual.total.knowledge == actual.unknown_requests.knowledge == "unknown"
    else:
        assert actual.total.value == sum(len(scope.dispatches or ()) for scope in scopes)
        assert actual.total.value == sum(p.value or 0 for p in actual.purposes)
    if name == "zero-proven":
        assert actual.input_units == actual.output_units == actual.total == Estimate("exact", 0)
    if name in {"unknown-usage", "partial-usage"}:
        assert actual.input_units.knowledge == actual.output_units.knowledge == "unknown"
    with pytest.raises(ValueError):
        reference.account(
            (
                Scope((Dispatch(0, "initial", (1, 2)),)),
                Scope((Dispatch(0, "initial", (1, 2)),)),
            )
        )


def test_forecasts_counting_and_limits() -> None:
    assert reference.shape(None) == (1, 1)
    assert reference.shape({}) == (1, 1)
    assert reference.shape({"key": [None, True]}) == (4, 3)
    need = reference.forecast(
        fixed=1,
        slots=2,
        adaptive=1,
        requests=3,
        waits=1,
        cleanup=2,
        limits=LIMITS,
    )
    assert need.pairs == 12 and need.events == 24
    prior = reference.Need(1, 2, 2, 1, 3)
    assert reference.fits(prior, reference.Need(1, 2, 2, 2, 3), Limits(2, 4, 4, 2, 6))
    assert not reference.fits(prior, reference.Need(1, 2, 2, 2, 3), Limits(2, 3, 4, 2, 6))
    value = Receipt((PublicPair(0, None, "execution", "success", (), 2),), "not_configured", 0)
    raw = reference.encode(value, LIMITS)
    nodes, depth = reference.shape(reference.tree(value))
    exact = Limits(1, 2, nodes, depth, len(raw), 2, 1)
    assert reference.encode(value, exact) == raw
    for field in ("pairs", "events", "nodes", "depth", "encoded_bytes", "duration"):
        with pytest.raises(ValueError):
            reference.encode(value, replace(exact, **{field: getattr(exact, field) - 1}))
    with pytest.raises(ValueError):
        reference.encode(replace(value, schema_version=2), LIMITS)
    with pytest.raises(ValueError):
        reference.encode(replace(value, delivered_pairs=1), LIMITS)
    with pytest.raises(ValueError):
        reference.encode(replace(value, pairs=(replace(value.pairs[0], ordinal=True),)), LIMITS)
    with pytest.raises(TypeError):
        reference.encode(cast("Close", {"schema_version": 1, "private": "canary"}), LIMITS)


def test_owner_cached_close_and_missing_phase() -> None:
    owner = object()
    close = Receipt((), "not_configured", 0)
    assert reference.owner_chain((owner,), (owner,), close, close)
    assert not reference.owner_chain((owner,), (object(),), close, close)
    assert not reference.owner_chain((None,), (None,), close, close)
    assert not reference.owner_chain((owner,), (owner,), close, replace(close))
    assert reference.account((Scope(()),)).total == Estimate("exact", 0)
    assert reference.account((Scope(None),)).total == Estimate("unknown", None)


def test_explain_qualification_and_canary_boundary() -> None:
    planned, conditional, upper, hard = reference.explain(3, 1, 2, ((True, True),))
    assert (planned, conditional, upper, hard) == (
        Estimate("exact", 1),
        Estimate("exact", 2),
        Estimate("known_upper", 3),
        "supported",
    )
    assert reference.explain(3, None, None, None) == (
        Estimate("unknown", None),
        Estimate("unknown", None),
        Estimate("known_upper", 3),
        "unknown",
    )
    assert reference.explain(3, 1, 2, ((True, False),))[-1] == "unsupported"
    assert reference.qualification(("met", "unmet", "unknown", "not_assessed"), 2, 2) == (3, 1, 1, 1, 2, 2)

    # Private owner input is opaque; no content or repr is consulted by this predicate.
    class Canary:
        def __repr__(self) -> str:
            raise AssertionError("秘密-canary repr consulted")

    owner = Canary()
    close, delivered = reference.observe((phase(pair()),), LIMITS, sink_configured=True)
    assert reference.owner_chain((owner,), (owner,), close, close)
    assert "秘密" not in json.dumps(asdict(close), ensure_ascii=False)
    assert "秘密" not in json.dumps([asdict(p) for p in delivered], ensure_ascii=False)
    assert b"canary" not in reference.encode(close, LIMITS)


MUTANTS = (
    ("used.pairs + addition.pairs <= limits.count", "True"),
    ("type(pair.terminal_ordinal) is not int", "False"),
    ("type(pair.terminal_parent) is not int", "False"),
    ("phase.need.envelope != limits", "False"),
    ("max(depth, logical_depth)", "depth"),
    ("logical_depth > min(phase.need.depth, limits.depth)", "False"),
    (
        '"start": {"correlation": correlation, "boundary": pair.boundary}',
        '"removed_start": {}',
    ),
    ('"terminal": {', '"removed_terminal": {'),
    (
        '"reasons": [r for r in REASONS if r in pair.reasons]',
        '"reasons": ["exception-canary"]',
    ),
    ("not pair.start_present", "False"),
    ("not pair.terminal_present", "False"),
    (
        "pair.terminal_parent, pair.terminal_boundary",
        "pair.parent, pair.terminal_boundary",
    ),
    (
        '"pairs": [pair_tree(p) for p in value.pairs]',
        '"private_identity": "canary", "pairs": [pair_tree(p) for p in value.pairs]',
    ),
    ("not set(pair.reasons) <= set(REASONS)", "False"),
    ("unknown_count == 0 and not missing", "True"),
    ('Estimate("exact", len(dispatches))', 'Estimate("exact", 2 * len(dispatches))'),
    (
        "sum(d.purpose == purpose for d in dispatches)",
        'sum((2 if purpose == "repair" else 1) * (d.purpose == purpose) for d in dispatches)',
    ),
    ('raise ValueError("byte capacity")', "return raw[:limits.encoded_bytes]"),
    ("not fits(used, phase.need, limits)", "False"),
)


@pytest.mark.parametrize(("old", "new"), MUTANTS)
def test_semantic_mutants(tmp_path: Path, old: str, new: str) -> None:
    source = Path(__file__).with_name("reference.py").read_text()
    assert old in source
    path = tmp_path / "mutant.py"
    path.write_text(source.replace(old, new))
    spec = importlib.util.spec_from_file_location("mutant457", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        for name in (
            "Accounting",
            "Estimate",
            "Failure",
            "Receipt",
            "PublicPair",
            "Pair",
            "Phase",
            "Scope",
            "Dispatch",
        ):
            setattr(module, name, getattr(reference, name))
        with pytest.raises((AssertionError, pytest.fail.Exception)):
            checks(module)
            count_checks(module)
    finally:
        del sys.modules[spec.name]


def test_worst_case_forecast_dominates_finite_encodings() -> None:
    for case in observations():
        close, _ = reference.observe(
            case.phases,
            case.limits,
            case.sink_failure_at,
            sink_configured=case.sink_configured,
        )
        if isinstance(close, Receipt):
            need = reference.requirements(len(close.pairs), LIMITS)
            nodes, depth = reference.shape(reference.tree(close))
            assert nodes <= need.nodes and depth <= need.depth
            assert len(reference.encode(close, LIMITS)) <= need.encoded_bytes
    assert (
        reference.requirements(1, replace(LIMITS, count=10, duration=10)).encoded_bytes
        > reference.requirements(1, replace(LIMITS, count=9, duration=9)).encoded_bytes
    )
    assert (
        reference.requirements(1, replace(LIMITS, count=10000, duration=10000)).encoded_bytes
        > reference.requirements(1, replace(LIMITS, count=999, duration=999)).encoded_bytes
    )
    first = PublicPair(0, None, "execution", "failure", ("lost", "invalid_input"), 0)
    second = replace(first, reasons=tuple(reversed(first.reasons)))
    assert reference.encode(Receipt((first,), "not_configured", 0), LIMITS) == reference.encode(
        Receipt((second,), "not_configured", 0), LIMITS
    )


def test_deep_logical_chain_distinct_from_flat_encoding_depth() -> None:
    facts = tuple(pair(i, parent=None if i == 0 else i - 1) for i in range(12))
    initial = reference.requirements(12, LIMITS, logical_depth=12)
    limits = Limits(initial.pairs, initial.events, initial.nodes, 12, initial.encoded_bytes)
    requirement = reference.requirements(12, limits, logical_depth=12)
    assert requirement.depth == 12
    close, _ = reference.observe((Phase(requirement, facts),), limits)
    assert isinstance(close, Receipt)
    assert reference.shape(reference.tree(close))[1] < 12
    assert reference.encode(close, replace(limits, depth=11))
    smaller = replace(limits, depth=11)
    rejected, _ = reference.observe((Phase(reference.requirements(12, smaller, logical_depth=12), facts),), smaller)
    assert rejected == Failure("capacity_contract")
    underforecast = replace(requirement, depth=11)
    rejected, _ = reference.observe((Phase(underforecast, facts),), limits)
    assert rejected == Failure("capacity_contract")


def test_builder_limit_envelope_equality() -> None:
    declared = phase(pair())
    equal_copy = replace(LIMITS)
    close, _ = reference.observe((declared,), equal_copy)
    assert isinstance(close, Receipt)
    incompatible = replace(LIMITS, count=99)
    rejected, delivered = reference.observe((declared,), incompatible)
    assert rejected == Failure("capacity_contract") and not delivered
