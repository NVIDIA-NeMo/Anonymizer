# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any, cast

import pytest

from anonymizer.engine.graph_sdk._activation_values import Start
from anonymizer.graph import ContractViolation, ValidationCode


def test_invalid_activation_identity_rejects_at_event_construction() -> None:
    with pytest.raises(ContractViolation) as error:
        Start(activation=cast(Any, 7))
    assert error.value.code is ValidationCode.INVALID_TYPE
