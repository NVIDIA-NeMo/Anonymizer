# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finite relational A1/A2 reference; no SDK imports or execution simulation."""

from __future__ import annotations

import hashlib
import json
import sys
from itertools import combinations, product
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

Identity = tuple[str, str, str, str, str, str, int, int]
Atom = tuple[str, Identity]
DeclaredKinds = tuple[tuple[str, ...], tuple[str, ...]]
CaseInput = tuple[str, str, tuple[str, ...], tuple[int, ...], str, bool, bool]


class Summary(TypedDict):
    """Describe first acceptance without retaining response payloads."""

    association: str
    outcome: str | None
    category: str
    failure: str | None
    condition: str
    consumed_context_ports: list[str]
    artifacts: list[str]


class Alternative(TypedDict):
    """Describe one permitted complete publication result."""

    survivors: list[str]
    retained: list[Atom]
    publication_failures: list[str]


class Result(TypedDict):
    """Keep acceptance, publication and accounting independently observable."""

    schema_version: int
    summaries: list[Summary]
    alternatives: list[Alternative]
    dispatches: int
    settlements: list[str]
    exact_usage: int | None
    stop_known: bool


class Case(TypedDict):
    """Bind a stable finite-alphabet input to its allowed results."""

    id: str
    input: CaseInput
    expected: Result


class TextCase(Case):
    """Declare optional-empty kinds and context separately from physical acceptance."""

    declared_kinds: DeclaredKinds
    consumed_context_ports: tuple[str, ...]


ASSOCIATIONS = ("A", "B")
KINDS = ("artifact", "port", "provenance", "assessment", "membership", "final_output")
TEXT_KINDS = ("artifact", "port", "provenance", "final_output")
CORRELATIONS = ("exact", "missing", "duplicate", "extra", "foreign", "unlocalizable", "outer")
SHAPES = ("valid", "output_bad", "assessment_bad")


def compute_digest(data: bytes) -> str:
    """Bind exact artifact bytes to a SHA256 digest."""
    return hashlib.sha256(data).hexdigest()


def enumerate_subsets(items: Sequence[str]) -> list[frozenset[str]]:
    """Enumerate the finite association powerset."""
    return [frozenset(c) for n in range(len(items) + 1) for c in combinations(items, n)]


def build_delta(association: str, kinds: tuple[str, ...] = KINDS) -> frozenset[Atom]:
    """Construct exactly the declared unit-cost facts with distinct identity coordinates."""
    # Exact identity coordinates remain separate even for equal payload text.
    identity = (
        association,
        "producer-" + association,
        "parent-" + association,
        "activation-" + association,
        "target-" + association,
        "out",
        0,
        1 if association == "A" else 2,
    )
    return frozenset((kind, identity) for kind in kinds)


def find_acceptable_survivors(
    eligible: Iterable[str], capacities: Sequence[int], declared_kinds: DeclaredKinds = (KINDS, KINDS)
) -> frozenset[frozenset[str]]:
    """Retain every inclusion-maximal feasible association subset."""
    feasible = [
        s
        for s in enumerate_subsets(sorted(eligible))
        if all(
            sum(atom[0] == kind for a in s for atom in build_delta(a, declared_kinds[ASSOCIATIONS.index(a)])) <= cap
            for kind, cap in zip(KINDS, capacities, strict=True)
        )
    ]
    return frozenset(s for s in feasible if not any(s < other for other in feasible))


def correlate(reserved: frozenset[str], envelope: object) -> bool:
    """Require the exact reserved keys with no duplicate or foreign member."""
    return (
        type(envelope) is tuple
        and len(envelope) == len(reserved)
        and all(type(a) is str and a in reserved for a in envelope)
        and len(set(envelope)) == len(envelope)
    )


def build_envelope(keys: tuple[str, ...], defect: str) -> object:
    """Construct one declared correlation discriminator."""
    return dict(
        exact=keys,
        missing=keys[:-1],
        duplicate=keys + keys[:1],
        extra=keys + ("C",),
        foreign=("C",) + keys[1:],
        unlocalizable=(None,) + keys[1:],
        outer=list(keys),
    )[defect]


def evaluate_publication(
    mode: str,
    correlation: str,
    shapes: tuple[str, ...],
    capacities: tuple[int, ...],
    usage: str,
    domain_failure: bool = False,
    aggregate_ok: bool = True,
    prior_settlements: tuple[str, ...] = (),
    declared_kinds: DeclaredKinds = (KINDS, KINDS),
    consumed_context_ports: tuple[str, ...] = ("ctx",),
) -> Result:
    """Evaluate the finite publication predicate without simulating an executor."""
    reserved = frozenset(ASSOCIATIONS)
    # Malformation alphabet denotes an envelope defect in every physical request.
    requests = (ASSOCIATIONS,) if mode == "shared" else tuple((a,) for a in ASSOCIATIONS)
    correlated = all(correlate(frozenset(keys), build_envelope(keys, correlation)) for keys in requests)
    eligible = reserved if correlated else frozenset()
    valid = (
        frozenset(a for a, shape in zip(ASSOCIATIONS, shapes, strict=True) if shape == "valid" and aggregate_ok)
        & eligible
    )
    summaries: list[Summary] = []
    for a in sorted(eligible):
        good = a in valid
        summaries.append(
            dict(
                association=a,
                outcome=("domain_failure" if domain_failure else "ok") if good else None,
                category="success" if good else "failure",
                failure=None if good else "malformed_response",
                condition="result" if good else "failure",
                consumed_context_ports=list(consumed_context_ports) if good else [],
                artifacts=[],
            )
        )
    survivors = find_acceptable_survivors(valid, capacities, declared_kinds)
    alternatives: list[Alternative] = []
    for s in sorted(survivors, key=lambda x: tuple(sorted(x))):
        atoms = frozenset(atom for a in s for atom in build_delta(a, declared_kinds[ASSOCIATIONS.index(a)]))
        alternatives.append(dict(survivors=sorted(s), retained=sorted(atoms), publication_failures=sorted(valid - s)))
    count = 1 if mode == "shared" else 2
    settlements = list(prior_settlements) + [usage] * count
    return dict(
        schema_version=2,
        summaries=summaries,
        alternatives=alternatives,
        dispatches=count,
        settlements=settlements,
        exact_usage=len(settlements) if all(x == "exact:1" for x in settlements) else None,
        stop_known=usage == "exact:1",
    )


def generate_cases() -> Iterator[Case]:
    """Enumerate the reviewed alphabet in its frozen case-ID order."""
    alphabet = product(
        ("shared", "per_task"),
        CORRELATIONS,
        product(SHAPES, repeat=2),
        range(3),
        ("unknown", "exact:1"),
        (False, True),
        (False, True),
    )
    for i, (mode, corr, shapes, cap, usage, failure, aggregate) in enumerate(alphabet):
        args: CaseInput = (mode, corr, shapes, (cap,) * len(KINDS), usage, failure, aggregate)
        yield dict(id=f"A12-{i:05d}", input=args, expected=evaluate_publication(*args))


def generate_text_cases() -> Iterator[TextCase]:
    """Declare a success-only four-kind corpus without narrowing the full-six corpus."""
    inputs = (case["input"] for case in generate_cases() if not case["input"][5])
    declared_kinds = (TEXT_KINDS, TEXT_KINDS)
    for index, args in enumerate(inputs):
        yield TextCase(
            id=f"A12T-{index:05d}",
            input=args,
            declared_kinds=declared_kinds,
            consumed_context_ports=(),
            expected=evaluate_publication(*args, declared_kinds=declared_kinds, consumed_context_ports=()),
        )


def encode_json(value: object) -> bytes:
    """Encode deterministic, compact case and manifest JSON."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def write_cases(destination: Path, simple_text: bool = False) -> dict[str, object]:
    """Write case bytes and bind their generator and calibration tests."""
    rows: list[Case] = list(generate_text_cases()) if simple_text else list(generate_cases())
    data = b"".join(encode_json(row) + b"\n" for row in rows)
    source = Path(__file__).read_bytes()
    manifest: dict[str, object] = dict(
        version=3,
        corpus="simple-text" if simple_text else "full-six",
        cases=len(rows),
        first=rows[0]["id"],
        last=rows[-1]["id"],
        generator_sha256=compute_digest(source),
        cases_sha256=compute_digest(data),
        tests_sha256=compute_digest(Path(__file__).with_name("test_reference.py").read_bytes()),
        authority_sha256="6c711f53ce9e7a737439608b7c8c633c16fd3fed1bc9178cdf197d71cad3859b",
        alphabet=dict(
            mode=2, correlation=7, shapes=9, capacities=3, usage=2, domain_failure=1 if simple_text else 2, aggregate=2
        ),
        bounds=dict(
            associations=2, requests=2, atoms_per_association=4 if simple_text else 6, capacity_per_kind=[0, 1, 2]
        ),
        declaration=dict(
            kinds=TEXT_KINDS if simple_text else KINDS,
            cost_per_kind=[int(kind in TEXT_KINDS) if simple_text else 1 for kind in KINDS],
            consumed_context_ports=[] if simple_text else ["ctx"],
            graph_outcomes=["ok"] if simple_text else ["ok", "domain_failure"],
        ),
        independence="A/B deltas disjoint; release closure excluded",
    )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "cases.jsonl").write_bytes(data)
    (destination / "manifest.json").write_bytes(encode_json(manifest) + b"\n")
    return manifest


def write_bundle(destination: Path) -> dict[str, object]:
    """Freeze both separate corpora and their manifests in one evidence bundle."""
    manifests = {
        name: write_cases(destination / name, simple_text=name == "simple-text") for name in ("full-six", "simple-text")
    }
    manifest: dict[str, object] = dict(version=3, corpora=manifests)
    (destination / "manifest.json").write_bytes(encode_json(manifest) + b"\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(write_bundle(Path(sys.argv[1])), sort_keys=True))
