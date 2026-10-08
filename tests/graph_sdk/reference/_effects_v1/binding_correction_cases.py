# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent effects_v1 reference: binding correction cases."""

from __future__ import annotations

from tests.graph_sdk.reference._effects_v1.builders import (
    _bind,
    _binding_decl,
    _decl,
    _dispatch,
    _items,
    _materialization_decl,
    _materialization_spec,
    _materialization_trace,
    _reserve,
    _source_failure,
    _source_result,
    _trace,
)
from tests.graph_sdk.reference._effects_v1.case import (
    _case,
)
from tests.graph_sdk.reference._effects_v1.model import (
    Object,
    _object,
)


def _binding_correction_specs() -> list[Object]:
    cases: list[Object] = []
    prior = [
        _bind("D0", "P0"),
        _reserve("R0", ["D0"], purpose="initial_binding"),
        _dispatch("R0"),
        _source_result(
            "D0",
            "S0",
            [{"association": "D0", "key": 0, "text": "kept", "version": 1}],
            request="R0",
        ),
    ]
    cases.append(
        _case(
            "binding",
            "explicit_omission_preserves_prior",
            _binding_decl({"D0": "S0", "D1": "S1"}, optional=("D1",)),
            [
                *prior,
                _bind("D1", "P0"),
                _reserve("R1", ["D1"], purpose="initial_binding"),
                _dispatch("R1"),
                _source_failure("D1", "S1", request="R1", disposition="omitted_optional"),
                {"kind": "binding_finish"},
            ],
        )
    )
    misuse: tuple[tuple[str, Object, Object], ...] = (
        (
            "required_omission_misuse",
            _binding_decl({"D0": "S0"}, max_requests=1),
            _source_failure("D0", "S0", request="R0", disposition="omitted_optional"),
        ),
        (
            "omission_failure_mismatch",
            _binding_decl({"D0": "S0"}, optional=("D0",)),
            dict(
                _source_failure("D0", "S0", request="R0", failure="retryable", disposition="omitted_optional"),
                kind="source_failure_constructor",
            ),
        ),
    )
    for name, declaration, event in misuse:
        cases.append(
            _case(
                "binding",
                name,
                declaration,
                [*_trace(("D0",)), event],
            )
        )
    adaptive_spec = _materialization_spec("adaptive", "collection")
    adaptive_declaration = _materialization_decl(adaptive_spec, max_requests=1)
    adaptive_declaration["retrieval_sources"] = {"A0": "S0"}
    adaptive_prefix = _materialization_trace(adaptive_spec, _items("selector"))[:-1]
    for event in adaptive_prefix:
        if event["kind"] == "reserve":
            event["purpose"] = "adaptive_retrieval"
    cases.append(
        _case(
            "binding",
            "adaptive_omission_misuse",
            adaptive_declaration,
            [*adaptive_prefix, _source_failure("A0", "S0", request="R0", disposition="omitted_optional")],
        )
    )
    for name, failure, purpose in (
        ("source_failure_retry_authority", "retryable", "retry"),
        ("source_failure_correction_authority", "malformed_response", "correction"),
    ):
        cases.append(
            _case(
                "binding",
                name,
                _binding_decl({"D0": "S0"}),
                [
                    *_trace(("D0",)),
                    _source_failure("D0", "S0", request="R0", failure=failure),
                    _reserve("R1", ["D0"], purpose=purpose),
                ],
            )
        )
    missing_failure = _source_failure("D0", "S0", request="R0")
    missing_failure.pop("failure")
    missing_failure["kind"] = "source_failure_constructor"
    missing_settlement = _source_failure("D0", "S0", request="R0")
    missing_settlement.pop("settlement")
    missing_settlement["kind"] = "source_failure_constructor"
    cases.extend(
        [
            _case(
                "binding",
                "source_failure_missing_failure",
                _binding_decl({"D0": "S0"}),
                [*_trace(("D0",)), missing_failure],
            ),
            _case(
                "binding",
                "source_failure_missing_settlement",
                _binding_decl({"D0": "S0"}),
                [*_trace(("D0",)), missing_settlement],
            ),
        ]
    )
    no_settlement = _source_failure("D0", "S0", request="R0")
    no_settlement["settlement"] = None
    cases.append(
        _case(
            "binding",
            "source_failure_explicit_no_settlement",
            _binding_decl({"D0": "S0"}),
            [*_trace(("D0",)), no_settlement],
        )
    )
    base_result = _source_result(
        "D0",
        "S0",
        [{"association": "D0", "key": 0, "text": "a", "version": 1}],
        request="R0",
    )
    for name, changes in (
        ("source_result_wrong_outcome", {"outcome": "ok"}),
        ("source_result_outputs_present", {"outputs": [{"port": "x"}]}),
        ("source_result_consumed_present", {"consumed_context_ports": ["x"]}),
    ):
        cases.append(
            _case(
                "binding",
                name,
                _binding_decl({"D0": "S0"}),
                [
                    *_trace(("D0",)),
                    {
                        "kind": "binding_result",
                        "association": "D0",
                        "outcome": changes.get("outcome", "retrieved"),
                        "outputs": changes.get("outputs", []),
                        "consumed_context_ports": changes.get("consumed_context_ports", []),
                    },
                ],
            )
        )
    cases.append(
        _case(
            "binding",
            "empty_optional_response_malformed",
            _binding_decl({"D0": "S0"}, optional=("D0",), max_requests=1),
            [*_trace(("D0",)), _source_result("D0", "S0", [], request="R0")],
        )
    )
    oversize = _source_result(
        "D0",
        "S0",
        [{"association": "D0", "key": 0, "text": "too-big", "version": 1}],
        request="R0",
    )
    _object(oversize["settlement"])["usage"] = {"input": 3, "output": 5}
    cases.append(
        _case(
            "binding",
            "oversize_retrieved_known_usage",
            _binding_decl({"D0": "S0"}, max_bytes=1),
            [*_trace(("D0",)), oversize],
        )
    )
    cases.append(
        _case(
            "binding",
            "success_after_failure_preserves_authority",
            _binding_decl({"D0": "S0"}),
            [
                *_trace(("D0",)),
                _source_failure("D0", "S0", request="R0", failure="retryable"),
                base_result,
            ],
        )
    )
    cases.append(
        _case(
            "binding",
            "adaptive_semantic_outcome_independent",
            _decl(),
            [
                *_trace(("A0",)),
                {"kind": "result", "outcomes": {"A0": "adaptive_ok"}, "request": "R0", "returned": ["A0"]},
            ],
        )
    )
    return cases
