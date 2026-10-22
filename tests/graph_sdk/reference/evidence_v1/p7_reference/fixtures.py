# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Declarative finite cases; no product outputs or legacy expected values are consumed."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import fields, is_dataclass, replace
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

from tests.graph_sdk.reference.evidence_v1.p7_reference.model import (
    Assessment,
    Final,
    Port,
    Production,
    Provenance,
    Raw,
    Requirement,
    Revision,
    artifact_revision,
    evaluate,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


def build_raw(selection: str = "AB", topology: str = "independent") -> Raw:
    """Declare three independent assessment occurrences and exact raw lineage rows."""
    refs = {name: ("I0", name, 1) for name in ("A", "B", "C", "EA", "EB", "EC", "XA", "XB", "XC", "D", "Z")}
    occs = {t: ("I0", t, "N" + t, 0) for t in "ABC"}
    declarations = []
    facts = []
    ports = []
    finals = []
    provenance = []
    for target in "ABC":
        occurrence = occs[target]
        consumed = tuple(selection) if target == "A" else (target,)
        environment = (
            Revision("configuration", ("N" + target,), "c1"),
            Revision("absence", ("I0", "Q" + target), 1),
            Revision("state", ("S" + target,), 1),
        )
        declarations.append(Production("N" + target, "P", "privacy", consumed, frozenset(("K0", "K1")), environment))
        facts.append(Assessment(occurrence, "P", refs["E" + target], "satisfied", environment))
        ports.extend(
            (
                Port(occurrence, "subject", refs[target], "candidate"),
                Port(occurrence, "evidence", refs["E" + target], "evidence"),
                Port(occurrence, "result", refs[target], "candidate"),
            )
        )
        ports.extend(
            Port(occurrence, name, refs[name if name == "D" else "X" + name], "decision" if name == "D" else "artifact")
            for name in consumed
        )
        key = (occurrence, "result")
        finals.append(Final(target, key, refs[target]))
        provenance.append(Provenance(key, refs[target], (), False))
    # Decision ancestry is declared separately; equal candidate identity does not erase this path.
    decision = (("I0", "A", "DEC", 0), "out")
    alias = (("I0", "A", "ALIAS", 1), "out")
    unrelated = (("I0", "C", "OTHER", 0), "out")
    provenance += [
        Provenance(decision, refs["D"], (), True),
        Provenance(alias, refs["A"], (decision,), False),
        Provenance(unrelated, refs["Z"], (), True),
    ]
    if topology == "ancestry":
        provenance[0] = provenance[0]._replace(parents=(alias,))
    dependencies = {
        "independent": (),
        "chain": (("A", "B"), ("B", "C")),
        "mixed": (("A", "B"),),
        "reverse": (("B", "A"),),
        "ancestry": (),
    }[topology]
    groups = (frozenset("A"), frozenset("BC")) if topology == "mixed" else tuple(frozenset(t) for t in "ABC")
    return Raw(
        "I0",
        tuple("ABC"),
        tuple(declarations),
        tuple(
            Requirement(t, "privacy", frozenset(tuple(selection) if t == "A" else (t,)), frozenset(("K0",)))
            for t in "ABC"
        ),
        tuple((refs[t], t) for t in "ABC"),
        tuple((f.target, f.producer) for f in finals),
        tuple((p.producer, p.parents) for p in provenance),
        tuple(ports),
        tuple((occs[t], "ok") for t in "ABC"),
        tuple(facts),
        tuple(facts),
        tuple(refs.values()),
        tuple(finals),
        tuple(provenance),
        dependencies,
        groups,
        tuple(artifact_revision(ref) for ref in refs.values()) + tuple(r for p in declarations for r in p.environment),
    )


def change_revision(raw: Raw, kind: str, key: tuple[str, ...], change: str) -> Raw:
    """Change only the selected current key, preserving captured assessment facts."""
    rows = []
    artifacts = raw.artifacts
    for row in raw.current:
        if (row.kind, row.key) != (kind, key) or change == "current":
            rows.append(row)
        elif change == "stale":
            value = (*row.value[:2], 2) if isinstance(row.value, tuple) else 2 if isinstance(row.value, int) else "c2"
            rows.append(row._replace(value=value))
            if isinstance(value, tuple):
                artifacts += (value,)
    return replace(raw, current=tuple(rows), artifacts=artifacts)


def enumerate_cases() -> Iterator[tuple[str, Raw]]:
    """Cross small currentness axes; retain graph shape as an independent axis."""
    axes = (
        ("subject", "artifact", ("I0", "A")),
        ("consumed", "artifact", ("I0", "XB")),
        ("decision", "artifact", ("I0", "D")),
        ("absence", "absence", ("I0", "QA")),
        ("configuration", "configuration", ("NA",)),
        ("state", "state", ("SA",)),
        ("unrelated", "artifact", ("I0", "Z")),
    )
    alphabet = product(
        ("A", "B", "AB", "D"),
        ("independent", "chain", "mixed", "reverse", "ancestry"),
        axes,
        ("current", "stale", "unknown"),
        ("all", "A", "B", "none"),
    )
    for index, (selection, topology, (_, kind, key), revision, submitted) in enumerate(alphabet):
        raw = change_revision(build_raw(selection, topology), kind, key, revision)
        raw = replace(
            raw, submissions=tuple(a for a in raw.assessments if submitted == "all" or a.occurrence[1] == submitted)
        )
        yield f"P7E-{index:05d}", raw


def project(value: object) -> object:
    """Encode closed neutral rows without normalizing their identity coordinates."""
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: project(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, (set, frozenset)):
        return [project(item) for item in sorted(value)]
    if isinstance(value, tuple):
        return [project(item) for item in value]
    return value


def encode(value: object) -> bytes:
    """Emit reproducible finite JSON."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def write_bundle(destination: Path) -> dict[str, object]:
    """Freeze raw facts plus separately computed expected predicates."""
    rows = []
    event_count = 0
    declaration_graphs = set()
    datum_graphs = set()
    for case_id, raw in enumerate_cases():
        inputs = project(raw)
        if not isinstance(inputs, dict):
            raise TypeError("raw projection")
        declaration_graphs.add(
            encode(
                project(
                    (
                        raw.productions,
                        raw.requirements,
                        raw.selected_outputs,
                        raw.declared_parents,
                        raw.dependencies,
                        raw.atomic,
                    )
                )
            )
        )
        datum_graphs.add(encode(project((raw.dependencies, raw.atomic))))
        inputs["submissions"] = [
            next(i for i, fact in enumerate(raw.assessments) if fact is submitted) for submitted in raw.submissions
        ]
        rows.append(dict(id=case_id, input=inputs, expected=project(evaluate(raw))))
        event_count += sum(
            len(getattr(raw, name))
            for name in (
                "ports",
                "terminals",
                "assessments",
                "submissions",
                "artifacts",
                "finals",
                "provenance",
                "current",
            )
        )
    data = b"".join(encode(row) + b"\n" for row in rows)
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))}
    manifest: dict[str, object] = dict(
        version=1,
        cases=len(rows),
        traces=len(rows),
        event_rows=event_count,
        topology_templates=5,
        declaration_graphs=len(declaration_graphs),
        datum_atomic_graphs=len(datum_graphs),
        cases_sha256=hashlib.sha256(data).hexdigest(),
        sources=sources,
        bounds=dict(targets=3, assessments=3, artifact_versions=2, max_ports=13, max_provenance=6),
        alphabet=dict(consumption=4, graph=5, revision_axis=7, revision_state=3, submissions=4),
    )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "cases.jsonl").write_bytes(data)
    (destination / "manifest.json").write_bytes(encode(manifest) + b"\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(write_bundle(Path(sys.argv[1])), sort_keys=True))
