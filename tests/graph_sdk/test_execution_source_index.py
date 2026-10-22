# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The live source's entry index tracks state replacement; required lookups reject, tolerant ones do not."""

from __future__ import annotations

import pytest

from anonymizer.engine.graph_sdk._activation_reducer import advance_activation
from anonymizer.engine.graph_sdk._activation_values import CloseUnstarted, Select
from anonymizer.engine.graph_sdk._execution_frame import _SourceOccurrence
from anonymizer.graph import ActivationKey, EffectCode, EffectRejected
from tests.graph_sdk.test_activation import _initialized


def test_replaced_state_is_never_served_from_a_stale_index() -> None:
    empty, seed = _initialized()
    source = _SourceOccurrence(state=empty, keys={})
    assert source.find(seed.activation) is None  # indexes the empty state
    source.state = advance_activation(state=empty, event=Select(seeds=frozenset({seed})))
    ready = source.entry(seed.activation)
    assert ready.status == "ready"
    source.state = advance_activation(
        state=source.state, event=CloseUnstarted(activation=seed.activation, category="blocked")
    )
    assert source.entry(seed.activation).status == "blocked"
    source.state = empty
    assert source.find(seed.activation) is None


def test_required_lookup_rejects_missing_while_tolerant_lookup_returns_none() -> None:
    state, seed = _initialized()
    source = _SourceOccurrence(state=advance_activation(state=state, event=Select(seeds=frozenset({seed}))), keys={})
    absent = ActivationKey(invocation=seed.activation.invocation, occurrence=999, parent=None, iteration=None)
    assert source.find(absent) is None
    with pytest.raises(EffectRejected) as raised:
        source.entry(absent)
    assert raised.value.code is EffectCode.MISSING
    assert source.find(seed.activation) is source.entry(seed.activation)
