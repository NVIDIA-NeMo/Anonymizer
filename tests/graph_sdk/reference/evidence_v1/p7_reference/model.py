# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finite raw-fact joins and release predicates, independent of SDK implementations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import NamedTuple, TypeVar

Ref = tuple[str, str, int]
Occurrence = tuple[str, str, str, int]
Producer = tuple[Occurrence, str]
T = TypeVar("T")


class Revision(NamedTuple):
    kind: str
    key: tuple[str, ...]
    value: Ref | str | int


class Production(NamedTuple):
    node: str
    promise: str
    meaning: str
    consumed: tuple[str, ...]
    coverage: frozenset[str]
    environment: tuple[Revision, ...]


class Port(NamedTuple):
    occurrence: Occurrence
    name: str
    artifact: Ref
    role: str


@dataclass(frozen=True, eq=False)
class Assessment:
    occurrence: Occurrence
    promise: str
    artifact: Ref
    finding: str
    environment: tuple[Revision, ...]


class Provenance(NamedTuple):
    producer: Producer
    artifact: Ref
    parents: tuple[Producer, ...]
    decision: bool


class Final(NamedTuple):
    target: str
    producer: Producer
    artifact: Ref


class Requirement(NamedTuple):
    target: str
    meaning: str
    consumed: frozenset[str]
    coverage: frozenset[str]


@dataclass(frozen=True)
class Raw:
    invocation: str
    targets: tuple[str, ...]
    productions: tuple[Production, ...]
    requirements: tuple[Requirement, ...]
    candidate_owners: tuple[tuple[Ref, str], ...]
    selected_outputs: tuple[tuple[str, Producer], ...]
    declared_parents: tuple[tuple[Producer, tuple[Producer, ...]], ...]
    ports: tuple[Port, ...]
    terminals: tuple[tuple[Occurrence, str], ...]
    assessments: tuple[Assessment, ...]
    submissions: tuple[Assessment, ...]
    artifacts: tuple[Ref, ...]
    finals: tuple[Final, ...]
    provenance: tuple[Provenance, ...]
    dependencies: tuple[tuple[str, str], ...]
    atomic: tuple[frozenset[str], ...]
    current: tuple[Revision, ...]


class Verified(NamedTuple):
    target: str
    subject: Ref
    evidence: Ref
    meaning: str
    consumed: frozenset[str]
    coverage: frozenset[str]
    finding: str
    validity: str
    decisions: frozenset[Ref]


class Result(NamedTuple):
    verified: tuple[Verified, ...]
    eligible: frozenset[str]
    released: frozenset[str]
    decisions: frozenset[Ref]


def select_one(rows: tuple[T, ...]) -> T:
    """Reject missing or ambiguous exact joins without normalizing them away."""
    if len(rows) != 1:
        raise ValueError("missing_or_duplicate")
    return rows[0]


def artifact_revision(ref: Ref) -> Revision:
    """Keep invocation, key and version in the currentness comparison."""
    return Revision("artifact", ref[:2], ref)


def classify_validity(expected: tuple[Revision, ...], current: tuple[Revision, ...]) -> str:
    """Known changes take precedence over missing dependencies."""
    statuses = set()
    for dependency in expected:
        rows = tuple(r for r in current if (r.kind, r.key) == (dependency.kind, dependency.key))
        if not rows:
            statuses.add("unknown")
        else:
            value = select_one(rows).value
            statuses.add("current" if type(value) is type(dependency.value) and value == dependency.value else "stale")
    return "stale" if "stale" in statuses else "unknown" if "unknown" in statuses else "current"


def verify_rows(raw: Raw) -> tuple[Verified, ...]:
    """Derive subject, consumption, coverage and validity from exact retained owners."""
    owners = tuple(
        (occ, p.promise)
        for occ, outcome in raw.terminals
        if outcome == "ok"
        for p in raw.productions
        if p.node == occ[2]
    )
    if sorted((a.occurrence, a.promise) for a in raw.assessments) != sorted(owners):
        raise ValueError("assessment_inventory")
    for ref in raw.artifacts:
        if ref[0] != raw.invocation:
            raise ValueError("foreign_artifact")
    for port in raw.ports:
        if port.occurrence[0] != raw.invocation or port.occurrence[1] not in raw.targets:
            raise ValueError("foreign_port")
        if port.artifact not in raw.artifacts:
            raise ValueError("missing_artifact")
    verified = []
    for fact in raw.submissions:
        if not any(fact is retained for retained in raw.assessments):
            raise ValueError("unauthenticated")
        if sum(fact is other for other in raw.submissions) != 1:
            raise ValueError("duplicate_submission")
        outcome = select_one(tuple(out for occ, out in raw.terminals if occ == fact.occurrence))
        declaration = select_one(
            tuple(p for p in raw.productions if (p.node, p.promise) == (fact.occurrence[2], fact.promise))
        )
        ports = tuple(p for p in raw.ports if p.occurrence == fact.occurrence)
        subject = select_one(tuple(p for p in ports if p.name == "subject"))
        evidence = select_one(tuple(p for p in ports if p.name == "evidence"))
        consumed = tuple(select_one(tuple(p for p in ports if p.name == name)) for name in declaration.consumed)
        if (
            outcome != "ok"
            or subject.role != "candidate"
            or evidence.role != "evidence"
            or select_one(tuple(t for ref, t in raw.candidate_owners if ref == subject.artifact)) != fact.occurrence[1]
            or fact.artifact != evidence.artifact
            or fact.environment != declaration.environment
            or fact.finding not in ("satisfied", "unsatisfied", "unknown")
        ):
            raise ValueError("contradictory_assessment")
        if any(
            p.role == "decision" and not any(row.decision and row.artifact == p.artifact for row in raw.provenance)
            for p in consumed
        ):
            raise ValueError("unowned_decision")
        dependencies = (artifact_revision(subject.artifact),) + tuple(artifact_revision(p.artifact) for p in consumed)
        verified.append(
            Verified(
                fact.occurrence[1],
                subject.artifact,
                fact.artifact,
                declaration.meaning,
                frozenset(declaration.consumed),
                declaration.coverage,
                fact.finding,
                classify_validity(dependencies + fact.environment, raw.current),
                frozenset(p.artifact for p in consumed if p.role == "decision"),
            )
        )
    return tuple(verified)


def enumerate_release(raw: Raw, eligible: frozenset[str]) -> frozenset[str]:
    """Take the union of all dependency-closed, whole-atomic feasible subsets."""
    subsets = (frozenset(c) for size in range(len(raw.targets) + 1) for c in combinations(raw.targets, size))
    feasible = [
        s
        for s in subsets
        if s <= eligible
        and all(dependent not in s or prerequisite in s for prerequisite, dependent in raw.dependencies)
        and all(not (s & group) or group <= s for group in raw.atomic)
    ]
    return frozenset().union(*feasible)


def collect_decisions(raw: Raw, roots: tuple[Producer, ...]) -> frozenset[Ref]:
    """Enumerate ancestor reachability using bounded paths, retaining occurrence aliases."""
    paths = {(root,) for root in roots}
    visited: set[Producer] = set()
    for _ in range(len(raw.provenance) + 1):
        if not paths:
            break
        expanded: set[tuple[Producer, ...]] = set()
        for path in paths:
            row = select_one(tuple(p for p in raw.provenance if p.producer == path[-1]))
            if row.parents != select_one(
                tuple(parents for key, parents in raw.declared_parents if key == row.producer)
            ):
                raise ValueError("wrong_provenance_parent")
            if row.artifact not in raw.artifacts:
                raise ValueError("unowned_provenance")
            visited.add(row.producer)
            for parent in row.parents:
                if parent in path:
                    raise ValueError("cyclic_provenance")
                expanded.add((*path, parent))
        paths = expanded
    if paths:
        raise ValueError("unbounded_provenance")
    return frozenset(p.artifact for p in raw.provenance if p.producer in visited and p.decision)


def evaluate(raw: Raw) -> Result:
    """Compute eligibility from facts before applying datum and atomic release rules."""
    verified = verify_rows(raw)
    supports: dict[str, tuple[Verified, ...]] = {}
    for target in raw.targets:
        final = select_one(tuple(f for f in raw.finals if f.target == target))
        source = select_one(tuple(p for p in raw.provenance if p.producer == final.producer))
        port = select_one(tuple(p for p in raw.ports if (p.occurrence, p.name) == final.producer))
        selected = select_one(tuple(key for t, key in raw.selected_outputs if t == target))
        if (
            source.artifact != final.artifact
            or final.producer != selected
            or port.artifact != final.artifact
            or final.producer[0][1] != target
        ):
            raise ValueError("wrong_final_producer")
        candidate_current = classify_validity((artifact_revision(final.artifact),), raw.current) == "current"
        matching = tuple(
            v
            for v in verified
            if v.target == target
            and v.subject == final.artifact
            and v.validity == "current"
            and v.finding == "satisfied"
        )
        requirements = tuple(r for r in raw.requirements if r.target == target)
        if candidate_current and all(
            any(v.meaning == r.meaning and v.consumed >= r.consumed and v.coverage >= r.coverage for v in matching)
            for r in requirements
        ):
            supports[target] = tuple(
                v
                for v in matching
                if any(
                    v.meaning == r.meaning and v.consumed >= r.consumed and v.coverage >= r.coverage
                    for r in requirements
                )
            )
    released = enumerate_release(raw, frozenset(supports))
    roots = tuple(f.producer for f in raw.finals if f.target in released)
    decisions = collect_decisions(raw, roots) | frozenset(d for t in released for v in supports[t] for d in v.decisions)
    return Result(verified, frozenset(supports), released, decisions)
