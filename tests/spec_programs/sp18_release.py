# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-18 release: dependencies, all-or-none groups, declaration order and annotation output (E18.1-E18.7)."""

from __future__ import annotations

from typing import Any

from anonymizer.engine.graph_core.authoring import (
    DATUM,
    TEXT,
    Assessment,
    Dependency,
    GraphScope,
    Limits,
    Operation,
    OutputGroup,
    ProtectionGraph,
    Workflow,
)
from anonymizer.engine.graph_core.execution import ProtectionResult, prepare, protect
from tests.spec_programs._builder import Checks

# A target whose text contains "bad" is not met; every other target is met.
CHECK = Operation.verifier(
    "check",
    lambda text, target: Assessment("bad" not in text, target),
    {"text": TEXT, "target": DATUM},
    checks=("check",),
)


def _run(
    texts: tuple[str, ...],
    deps: tuple[tuple[int, int], ...] = (),
    groups: tuple[tuple[int, ...], ...] = (),
    limits: Limits | None = None,
    **kw: Any,
) -> tuple[ProtectionResult, tuple]:
    """Targets from ``texts``; ``deps`` are (dependent, on) and ``groups`` member positions."""
    scope = GraphScope()
    ts = tuple(scope.datum(t) for t in texts)
    g = ProtectionGraph(
        scope,
        ts,
        dependencies=tuple(Dependency(ts[a], ts[b]) for a, b in deps),
        groups=tuple(OutputGroup(tuple(ts[i] for i in grp)) for grp in groups),
    )
    return protect(prepare(g, Workflow("w", (CHECK,), requires=("check",), **kw), limits=limits)), ts


def _view(r: ProtectionResult, ts: tuple) -> list[tuple[str, bool, str]]:
    return [(o.qualification, o.protected, o.reason) for o in map(r.outcome, ts)]


def program() -> list[dict[str, str]]:
    c = Checks("SP-18")

    r, ts = _run(("a", "bad"), deps=((0, 1),))
    c.check("E18.1", _view(r, ts)[0] == ("met", False, "withheld"), f"dependent={_view(r, ts)[0]}")

    r, ts = _run(("a", "bad"), groups=((0, 1),))
    v = _view(r, ts)
    c.check("E18.2", v == [("met", False, "withheld"), ("not_met", False, "failed")], f"group={v}")

    r, ts = _run(("a", "b"), deps=((0, 1),))
    c.check("E18.3", _view(r, ts) == [("met", True, "")] * 2, f"released={_view(r, ts)}")

    # 50 targets: every fifth is not met; a chain of dependencies over the even positions and groups of three.
    texts = tuple("bad" if i % 5 == 4 else f"t{i}" for i in range(50))
    deps = tuple((i + 2, i) for i in range(0, 48, 2))
    groups = tuple((i, i + 1, i + 3) for i in range(1, 46, 6))
    forward, fts = _run(texts, deps, groups)
    rev = list(reversed(range(50)))
    backward, bts = _run(
        tuple(texts[i] for i in rev),
        tuple((49 - a, 49 - b) for a, b in reversed(deps)),
        tuple(tuple(49 - i for i in reversed(grp)) for grp in reversed(groups)),
    )
    same = _view(forward, fts) == list(reversed(_view(backward, bts)))
    c.check("E18.4", same and bool(forward.protected), f"identical={same}, released={len(forward.protected)}")

    r, ts = _run(("a", "b"), release="annotation")
    c.check(
        "E18.5", not r.protected and _view(r, ts) == [("met", False, "annotation")] * 2, f"annotation={_view(r, ts)}"
    )

    # Group {0, 3} with 3 not met withholds 0, and with it the chain 2 -> 1 -> 0; 4 is independent.
    r, ts = _run(("a", "b", "c", "bad", "e"), deps=((1, 0), (2, 1)), groups=((0, 3),))
    released = [o.protected for o in map(r.outcome, ts)]
    c.check("E18.6", released == [False, False, False, False, True], f"released={released}")

    # A 5,000-target chain, each depending on the next, declared forward and backward: the not-met tail withholds
    # every target, with no recursion.
    n = 5000
    texts = tuple(f"t{i}" for i in range(n - 1)) + ("bad",)
    chain = tuple((i, i + 1) for i in range(n - 1))
    forward, fts = _run(texts, chain, limits=Limits(max_targets=n))
    backward, bts = _run(texts, tuple(reversed(chain)), limits=Limits(max_targets=n))
    fv, bv = _view(forward, fts), _view(backward, bts)
    c.check("E18.7", fv == bv and not forward.protected and fv[0][2] == "withheld", f"identical={fv == bv}")
    return c.lines
