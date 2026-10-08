# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent effects_v1 reference: case."""

from __future__ import annotations

from tests.graph_sdk.reference._effects_v1.evaluation import (
    evaluate_case,
)
from tests.graph_sdk.reference._effects_v1.model import (
    Object,
)
from tests.graph_sdk.reference._effects_v1.reducer import (
    reduce_trace,
)


def _case(
    family: str,
    name: str,
    declaration: Object,
    events: list[Object],
    boundary: str = "runtime",
    traces: list[list[Object]] | None = None,
    comparison_scope: str = "production_boundary",
    witness_obligation: str | None = None,
) -> Object:
    case: Object = {
        "boundary": boundary,
        "case_id": f"{family}/{name}",
        "declaration": declaration,
        "events": events,
        "family": family,
        "expected": {},
        "traces": [],
    }
    if comparison_scope != "production_boundary":
        case["comparison_scope"] = comparison_scope
    if witness_obligation is not None:
        case["witness_obligation"] = witness_obligation
    case["expected"] = evaluate_case(case)
    case["traces"] = [
        {"events": trace, "expected": reduce_trace(declaration, trace), "name": f"alternate_{index}"}
        for index, trace in enumerate(traces or [])
    ]
    return case
