# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Input availability remains distinct from terminal sequence readiness."""

from __future__ import annotations

import pytest

from anonymizer.graph.activation import ObserveTerminal, advance_activation
from anonymizer.graph.workflow import NodeOutputRef
from tests.graph_sdk.test_activation_reference_conformance import _Adapter, _array, _case, _transition_state


@pytest.mark.parametrize(("port", "expected"), [("result", "ready"), ("missing", "blocked"), (None, "blocked")])
def test_terminal_predecessor_does_not_erase_required_output(port: str | None, expected: str) -> None:
    case = _case("sequence_linked_pair/000-000/base")
    adapter = _Adapter(case)
    state = _transition_state(adapter, _array(case["events"])[:3])
    consumer = adapter.key("A1")
    assert next(entry for entry in state.entries if entry.activation == consumer).status == "unstarted"
    source = next(
        binding.source
        for binding in state.workflow.workflow.input_bindings
        if isinstance(binding.source, NodeOutputRef)
    )
    # A retained malformed port must not become a request for sequence-only readiness.
    object.__setattr__(source, "port", port)
    state = advance_activation(
        state=state,
        event=ObserveTerminal(activation=adapter.key("A0"), outcome="ok", category="success"),
    )
    assert next(entry for entry in state.entries if entry.activation == consumer).status == expected
