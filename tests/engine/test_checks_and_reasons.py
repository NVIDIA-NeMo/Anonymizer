# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Required checks become per-target reasons; checks the workflow does not require are advisories only."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from anonymizer.engine.constants import COL_FINAL_ENTITIES, COL_TEXT
from anonymizer.engine.graph_core.authoring import (
    DATUM,
    TEXT,
    Assessment,
    GraphScope,
    Operation,
    OutputGroup,
    ProtectionGraph,
    Workflow,
)
from anonymizer.engine.graph_core.execution import Advisory, Reason, prepare, protect
from anonymizer.engine.replace.route import _lower, _protect_rows
from anonymizer.engine.rewrite.rewrite_workflow import _protect_rewrite
from tests.graph.helpers import graph, run

CANARY = "canary-7f3e"


def _check(name: str, satisfied: Any, *, attributed: bool = True, cause: str = "") -> Operation:
    def assess(text: str, target: Any) -> Assessment:
        return Assessment(satisfied, target if attributed else None, cause)

    return Operation.verifier(name, assess, {"text": TEXT, "target": DATUM}, checks=(name,), subject=("text",))


def test_two_failed_required_checks_are_two_reasons() -> None:
    g, (a,) = graph("alpha")
    o = run(g, _check("one", False), _check("two", False)).outcome(a)
    assert o.reasons == (Reason("failed", "one", "one"), Reason("failed", "two", "two"))
    assert o.reason == "failed"


def test_inconsistent_invocation_appends_its_reason() -> None:
    g, (a,) = graph("alpha")
    o = run(g, _check("one", False, attributed=False)).outcome(a)
    assert o.reasons == (Reason("failed", "one", "one"), Reason("inconsistent_invocation"))


def test_outcome_keeps_the_exception_but_not_its_text() -> None:
    g, (a,) = graph("alpha")
    exc = ValueError(CANARY)

    def boom(text: str) -> str:
        raise exc

    o = run(g, Operation("boom", boom)).outcome(a)
    assert o.error is exc
    assert o.reason == "raised"
    assert CANARY not in repr(o.reasons)
    assert CANARY not in repr(o)
    assert CANARY not in o.reason


def test_rewrite_with_nan_leakage_could_not_be_assessed() -> None:
    g, (a,) = graph("alpha")

    def rewrite(target: tuple) -> list[Any]:
        metrics = {"any_high_leaked": False, "replacements_applied": True, "leakage_mass": float("nan")}
        metrics["placeholder_in_text"] = False
        return [{"text": "r", "metrics": metrics} for _ in target]

    evaluation: Any = type("E", (), {"flag_leakage_above": 0.5})()
    o = _protect_rewrite(g, rewrite, evaluation).outcome(a)
    assert o.reasons == (
        Reason("could_not_assess", "leakage_within_threshold", "assess-leakage_within_threshold", "metric_unavailable"),
    )


def test_replace_with_an_uncovered_span_fails_replacements_applied() -> None:
    entities = {"entities": [{"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5}]}
    lowered = _lower(
        pd.DataFrame({COL_TEXT: ["Alice met Bob"], COL_FINAL_ENTITIES: [entities]}), COL_TEXT, COL_FINAL_ENTITIES
    )
    o = _protect_rows(lowered, "replace", lambda i, rows: {}, False).outcome(lowered.targets[0])
    assert o.qualification == "not_met"
    assert o.reasons == (Reason("failed", "replacements_applied", "verify-patches", "unpatched_mention"),)


def test_numpy_false_verdict_fails_its_check() -> None:
    g, (a,) = graph("alpha")
    o = run(g, _check("one", np.False_)).outcome(a)
    assert o.qualification == "not_met"
    assert o.reason == "failed"


def test_withheld_group_member_names_every_failed_member() -> None:
    scope = GraphScope()
    ts = tuple(scope.datum(t) for t in ("bad", "bad", "a"))
    g = ProtectionGraph(scope, ts, groups=(OutputGroup(ts),))
    check = Operation.verifier(
        "check",
        lambda text, target: Assessment("bad" not in text, target),
        {"text": TEXT, "target": DATUM},
        checks=("check",),
        subject=("text",),
    )
    r = protect(prepare(g, Workflow("w", (check,), requires=("check",))))
    assert r.outcome(ts[2]).withheld_by == (ts[0], ts[1])


def _advised(*ops: Operation, requires: tuple[str, ...] = ()) -> Any:
    g, (a,) = graph("alpha")
    return protect(prepare(g, Workflow("w", ops, requires=requires))).outcome(a)


@pytest.mark.parametrize(
    ("satisfied", "cause", "advisory"),
    [
        (False, "", Advisory("adv", "adv", "failed")),
        (None, "x", Advisory("adv", "adv", "could_not_assess", "x")),
        (True, "", Advisory("adv", "adv", "passed")),
        (np.True_, "", Advisory("adv", "adv", "failed")),
    ],
    ids=["failed", "could_not_assess", "passed", "numpy_true"],
)
def test_unrequired_final_text_check_is_an_advisory(satisfied: Any, cause: str, advisory: Advisory) -> None:
    o = _advised(_check("adv", satisfied, cause=cause))
    assert o.advisories == (advisory,)
    assert o.reasons == ()


def test_unrequired_check_is_not_assessed() -> None:
    assert _advised(_check("adv", False)).qualification == "not_assessed"


def test_check_before_a_later_transform_is_no_advisory() -> None:
    g, (a,) = graph("alpha")
    upper = Operation("upper", lambda text: text.upper(), {"text": TEXT}, {"text": TEXT})
    r = protect(prepare(g, Workflow("w", (_check("adv", False), upper))))
    assert r.outcome(a).advisories == ()
    assert r.value(a, "assessment", producer="adv") is not None


def test_required_checks_give_reasons_while_unrequired_ones_advise() -> None:
    o = _advised(_check("req", False), _check("adv", True), requires=("req",))
    assert o.reasons == (Reason("failed", "req", "req"),)
    assert o.advisories == (Advisory("adv", "adv", "passed"),)
