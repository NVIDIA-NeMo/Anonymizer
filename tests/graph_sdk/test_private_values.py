# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import pickle

import pytest

from anonymizer.engine.graph_sdk.capabilities import ConfigNull
from anonymizer.engine.graph_sdk.data import DataGraph, DataLimits
from anonymizer.graph import WorkflowId


def test_data_values_keep_their_repr_and_pickle_round_trip() -> None:
    limits = DataLimits(max_datums=1, max_targets=2, max_text_bytes=3, max_declarations=4, max_group_members=5)

    restored = pickle.loads(pickle.dumps(limits))

    assert restored == limits
    assert repr(restored) == "<DataLimits>"
    assert repr(DataGraph.new()) == "<DataGraph>"


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (ConfigNull(), "preparation value serialization is not supported"),
        (WorkflowId.new(), "workflow value serialization is not supported"),
        (DataGraph.new().graph, "graph identity serialization is not supported"),
    ],
)
def test_private_values_refuse_pickling_with_their_module_message(value: object, message: str) -> None:
    assert copy.copy(value) is value
    assert copy.deepcopy(value) is value
    with pytest.raises(TypeError, match=f"^{message}$"):
        pickle.dumps(value)
