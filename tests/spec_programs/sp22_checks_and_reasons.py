# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-22 required checks and per-target reasons (E22.1-E22.8)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from anonymizer.engine.constants import COL_FINAL_ENTITIES, COL_LEAKAGE_MASS, COL_TEXT
from anonymizer.engine.graph_core.authoring import DATUM, TEXT, Assessment, Operation, Workflow
from anonymizer.engine.graph_core.execution import Advisory, Reason, prepare, protect
from anonymizer.engine.replace.route import _lower, _protect_rows
from anonymizer.engine.rewrite.rewrite_workflow import _protect_rewrite
from tests.spec_programs._builder import Checks, graph, run
from tests.spec_programs.sp16_bounded_operation import _reviews
from tests.spec_programs.sp18_release import _run

CANARY = "canary-7f3e"


def _check(name: str, satisfied: Any, *, attributed: bool = True, cause: str = "") -> Operation:
    def assess(text: str, target: Any) -> Assessment:
        return Assessment(satisfied, target if attributed else None, cause)

    return Operation.verifier(name, assess, {"text": TEXT, "target": DATUM}, checks=(name,), subject=("text",))


def program() -> list[dict[str, str]]:
    c = Checks("SP-22")
    g, (a,) = graph("alpha")

    # E22.1 two failed required checks are two reasons on one outcome.
    o1 = run(g, _check("one", False), _check("two", False)).outcome(a)
    want1 = (Reason("failed", "one", "one"), Reason("failed", "two", "two"))
    c.check("E22.1", o1.reasons == want1 and o1.reason == "failed", f"reasons={o1.reasons}")

    # E22.2 an inconsistent invocation appends its reason to the target's own.
    o2 = run(g, _check("one", False, attributed=False)).outcome(a)
    want2 = (Reason("failed", "one", "one"), Reason("inconsistent_invocation"))
    c.check("E22.2", o2.reasons == want2, f"reasons={o2.reasons}")

    # E22.3 the outcome keeps the exception object; its text never reaches a reason.
    exc = ValueError(CANARY)

    def boom(text: str) -> str:
        raise exc

    o3 = run(g, Operation("boom", boom)).outcome(a)
    leaked = CANARY in repr(o3.reasons) or CANARY in repr(o3) or CANARY in o3.reason
    c.check("E22.3", o3.error is exc and o3.reason == "raised" and not leaked, f"reason={o3.reason}, leaked={leaked}")

    # E22.4 a Rewrite row whose leakage is NaN could not be assessed and is flagged for review.
    def rewrite(target: tuple) -> list[Any]:
        metrics = {"any_high_leaked": False, "replacements_applied": True, "leakage_mass": float("nan")}
        metrics["placeholders_absent"] = True
        return [{"text": "r", "metrics": metrics} for _ in target]

    evaluation: Any = type("E", (), {"flag_leakage_above": 0.5})()
    o4 = _protect_rewrite(g, rewrite, evaluation).outcome(a)
    review = _reviews([{COL_LEAKAGE_MASS: float("nan")}], leak=0.5)
    want4 = (
        Reason("could_not_assess", "leakage_within_threshold", "assess-leakage_within_threshold", "metric_unavailable"),
    )
    c.check("E22.4", o4.reasons == want4 and review == [True], f"reasons={o4.reasons}, review={review}")

    # E22.5 a Replace row with a span the map does not cover fails replacements_applied.
    entities = {"entities": [{"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5}]}
    graph5 = _lower(
        pd.DataFrame({COL_TEXT: ["Alice met Bob"], COL_FINAL_ENTITIES: [entities]}), COL_TEXT, COL_FINAL_ENTITIES
    )
    r5 = _protect_rows(graph5, "replace", lambda i, rows: {}, False)
    o5 = r5.outcome(graph5.targets[0])
    want5 = (Reason("failed", "replacements_applied", "verify-patches", "unpatched_mention"),)
    c.check("E22.5", o5.qualification == "not_met" and o5.reasons == want5, f"reasons={o5.reasons}")

    # E22.6 a verdict that is neither True nor None fails its check (a numpy false is not a pass).
    o6 = run(g, _check("one", np.False_)).outcome(a)
    c.check("E22.6", o6.qualification == "not_met" and o6.reason == "failed", f"qualification={o6.qualification}")

    # E22.7 a group member withheld by its group names every member that failed.
    r7, ts7 = _run(("bad", "bad", "a"), groups=((0, 1, 2),))
    by = r7.outcome(ts7[2]).withheld_by
    c.check("E22.7", by == (ts7[0], ts7[1]), f"withheld_by={len(by)} targets")

    # E22.8 a final-text check the workflow does not require is an advisory, never a reason.
    def advised(*ops: Operation, requires: tuple[str, ...] = ()) -> Any:
        return protect(prepare(g, Workflow("w", ops, requires=requires))).outcome(a)

    o8a = advised(_check("adv", False))
    o8b = advised(_check("adv", None, cause="x"))
    o8c = advised(_check("adv", True))
    o8t = advised(_check("adv", np.True_))
    upper = Operation("upper", lambda text: text.upper(), {"text": TEXT}, {"text": TEXT})
    r8d = protect(prepare(g, Workflow("w", (_check("adv", False), upper))))
    o8e = advised(_check("req", False), _check("adv", True), requires=("req",))
    want8 = (
        o8a.qualification == "not_assessed"
        and o8a.reasons == ()
        and o8a.advisories == (Advisory("adv", "adv", "failed"),)
        and o8b.advisories == (Advisory("adv", "adv", "could_not_assess", "x"),)
        and o8c.advisories == (Advisory("adv", "adv", "passed"),)
        and o8t.advisories == (Advisory("adv", "adv", "failed"),)
        and r8d.outcome(a).advisories == ()
        and r8d.value(a, "assessment", producer="adv") is not None
        and o8e.reasons == (Reason("failed", "req", "req"),)
        and o8e.advisories == (Advisory("adv", "adv", "passed"),)
    )
    c.check(
        "E22.8",
        want8,
        f"advisories={o8a.advisories}, {o8b.advisories}, {o8c.advisories}, {o8t.advisories}, {o8e.advisories}",
    )
    return c.lines
