# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared invocation runner and the plain dict port kinds."""

from __future__ import annotations

import datetime as dt
import decimal
import enum
import json
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest

from anonymizer.engine.graph_core.authoring import (
    DATUM,
    TEXT,
    Datum,
    Limits,
    Operation,
    Workflow,
)
from anonymizer.engine.graph_core.kernel import CancelToken, run_in_frame
from anonymizer.engine.ndd import backend
from anonymizer.engine.ndd.backend import register_plain_kind, rows_as_graph, run_invocation
from anonymizer.interface.errors import AnonymizerWorkflowError

PLAIN_KINDS = ("replacement-map", "replacement-application", "replace-metrics", "rewrite-metrics", "coverage-metrics")
MODULES = (
    "anonymizer.interface.anonymizer",
    "anonymizer.engine.rewrite.rewrite_workflow",
    "anonymizer.engine.replace.route",
)
TEST_KIND = register_plain_kind("test-plain-metrics")
LIMITS = Limits(max_operations=4, max_targets=4, max_mentions=0, max_context_items=0)


class _Tolerated(Exception):
    pass


@pytest.mark.parametrize("first", MODULES)
def test_every_dict_kind_copies_to_plain_whatever_the_import_order(first: str) -> None:
    rest = [m for m in MODULES if m != first]
    script = "\n".join(
        [
            *(f"import {m}" for m in (first, *rest)),
            "from anonymizer.engine.ndd.backend import _plain",
            "from anonymizer.engine.graph_core.kernel import kind",
            f"print(all(kind(n).copy is _plain for n in {PLAIN_KINDS!r}))",
        ]
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "True"


def _metrics_workflow(value: Any) -> Workflow:
    def produce(text: str) -> dict[str, Any]:
        return {"text": text, "metrics": value}

    return Workflow("w", (Operation("produce", produce, {"text": TEXT}, {"text": TEXT, "metrics": TEST_KIND}),))


@pytest.mark.parametrize(
    "bad",
    [
        pd.NA,
        pd.NaT,
        np.datetime64("2026-01-01"),
        np.datetime64("2026-01-01T00:00:00.000000000"),
        np.timedelta64(5, "ns"),
        np.timedelta64(5, "D"),
        pd.Series([1, 2]),
        b"\xff",
        object(),
    ],
)
def test_a_value_with_no_plain_form_fails_its_target(bad: object) -> None:
    graph = rows_as_graph(["a"])
    workflow = _metrics_workflow({"score": bad})
    result = backend.protect(backend.prepare(graph, workflow, limits=LIMITS))
    assert [entry.reason for entry in result.state.entries if entry.error is not None] == ["invalid_output"]
    with pytest.raises(TypeError):
        run_invocation(graph, workflow, LIMITS)


def test_numpy_values_tuples_and_keys_are_converted_to_plain_python() -> None:
    value = {
        "array": np.array([1, 2]),
        "pair": ("a", np.float64(0.5)),
        "count": np.int64(3),
        "nan": np.float64("nan"),
        1: {"nested": [True, None]},
    }
    graph = rows_as_graph(["a"])
    port = run_invocation(graph, _metrics_workflow(value), LIMITS).value(graph.targets[0], "metrics")
    assert port["array"] == [1, 2] and [type(v) for v in port["array"]] == [int, int]
    assert port["pair"] == ["a", 0.5] and type(port["pair"][1]) is float
    assert type(port["count"]) is int and port["count"] == 3
    assert port["nan"] is None
    assert port["1"] == {"nested": [True, None]}


@pytest.mark.parametrize("number", [float("nan"), float("inf"), float("-inf"), np.float64("nan")])
def test_nan_and_infinity_become_none_so_the_value_is_strict_json(number: float) -> None:
    graph = rows_as_graph(["a"])
    port = run_invocation(graph, _metrics_workflow({"score": number, "items": [number]}), LIMITS).value(
        graph.targets[0], "metrics"
    )
    assert port == {"score": None, "items": [None]}
    json.dumps(port, allow_nan=False)


def test_a_plain_value_is_copied_deeply() -> None:
    value = {"score": 0.5, "nested": {"items": [1, "x", None, True]}}
    graph = rows_as_graph(["a"])
    result = run_invocation(graph, _metrics_workflow(value), LIMITS)
    port = result.value(graph.targets[0], "metrics")
    assert port == value
    assert port is not value and port["nested"] is not value["nested"]


@pytest.mark.parametrize(
    ("value", "plain"),
    [
        (pd.Timestamp("2026-01-01"), "2026-01-01T00:00:00"),
        (dt.date(2026, 1, 1), "2026-01-01"),
        (dt.timedelta(days=1), "P1D"),
        (decimal.Decimal("1.5"), "1.5"),
        (b"ab", "ab"),
        ({2, 1}, [1, 2]),
        (enum.IntEnum("Level", "LOW HIGH").HIGH, 2),
        (complex(0, 1), "1j"),
        ({1: "a", "1": "b"}, {"1": "b"}),
        ({True: 1, None: 2, (1, 2): 3}, {"true": 1, "None": 2, "1,2": 3}),
    ],
)
def test_values_pydantic_core_converts(value: object, plain: object) -> None:
    graph = rows_as_graph(["a"])
    port = run_invocation(graph, _metrics_workflow({"v": value}), LIMITS).value(graph.targets[0], "metrics")
    assert port == {"v": plain}


def _failing(error: Exception, *, verifier: bool) -> Workflow:
    def fail(*_: Any, **__: Any) -> Any:
        raise error

    if verifier:
        step = Operation.verifier("check", fail, {"text": TEXT, "target": DATUM})
    else:
        step = Operation("step", fail)
    return Workflow("w", (step,))


def test_a_verifier_error_is_not_raised() -> None:
    run_invocation(rows_as_graph(["a"]), _failing(ValueError("no"), verifier=True), LIMITS)


def test_a_tolerated_error_is_not_raised_and_another_is() -> None:
    graph = rows_as_graph(["a"])
    run_invocation(graph, _failing(_Tolerated(), verifier=False), LIMITS, tolerate=(_Tolerated,))
    with pytest.raises(ValueError):
        run_invocation(graph, _failing(ValueError("no"), verifier=False), LIMITS, tolerate=(_Tolerated,))


def test_cancellation_raises() -> None:
    token = CancelToken()
    token.cancel()
    workflow = Workflow("w", (Operation("step", lambda text: text),))
    with pytest.raises(AnonymizerWorkflowError, match="cancelled"):
        run_in_frame(run_invocation, rows_as_graph(["a"]), workflow, LIMITS, token=token)


def test_on_empty_operations_see_the_empty_input() -> None:
    seen: list[tuple] = []
    body = Operation.phase_scoped("body", lambda target: seen.append(target) or [], {"target": DATUM})
    run_invocation(rows_as_graph([]), Workflow("w", (body,)), LIMITS, on_empty=(body,))
    assert seen == [()]


def test_rows_as_graph_writes_none_as_empty_only_when_asked() -> None:
    assert [d.text for d in rows_as_graph([None, 1]).targets] == ["None", "1"]
    assert [d.text for d in rows_as_graph([None, 1], none_as_empty=True).targets] == ["", "1"]
    assert all(isinstance(d, Datum) for d in rows_as_graph(["x"]).targets)


def _judge(run: Any) -> Any:
    from anonymizer.interface.anonymizer import REPLACE_METRICS, _Judge

    return _Judge("judge", "metrics", REPLACE_METRICS, (), run)


def test_evaluate_calls_its_bodies_on_empty_input() -> None:
    from anonymizer.interface.anonymizer import _evaluate_on_graph

    seen: list[int] = []
    _evaluate_on_graph(pd.DataFrame({"c": []}), "c", [_judge(lambda df: (seen.append(len(df)) or df, []))])
    assert seen == [0]


def test_evaluate_writes_a_missing_candidate_as_empty_text() -> None:
    from anonymizer.interface.anonymizer import _evaluate_on_graph

    _, _, result = _evaluate_on_graph(pd.DataFrame({"c": [None]}), "c", [_judge(lambda df: (df, []))])
    assert [d.text for d in result.targets] == [""]


def test_rewrite_tolerates_an_unavailable_rewrite() -> None:
    from anonymizer.config.rewrite import EvaluationCriteria
    from anonymizer.engine.rewrite.rewrite_workflow import RewriteUnavailable, _protect_rewrite

    graph = rows_as_graph(["a"])
    _protect_rewrite(graph, lambda target: [RewriteUnavailable("no text")], EvaluationCriteria())
