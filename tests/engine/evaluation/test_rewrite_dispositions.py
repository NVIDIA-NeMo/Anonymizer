# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from anonymizer.engine import constants as cols
from anonymizer.engine.evaluation.rewrite_dispositions import build_leave_as_is_findings


@pytest.mark.parametrize("representation", ["native", "json", "numpy"])
def test_saved_assignments_count_pairs_per_record_and_preserve_reasons(representation: str) -> None:
    country = {
        "entity_value": "USA",
        "entity_label": "country",
        "category": "quasi_identifier",
        "source": "tagged",
        "sensitivity": "low",
        "protection_method_suggestion": "leave_as_is",
        "low_sensitivity_reason": "Broad geography",
    }
    latent = {
        **country,
        "entity_value": "adult",
        "entity_label": "age_group",
        "source": "latent",
        "category": "latent_identifier",
        "low_sensitivity_reason": "Broad age group",
    }
    protected = {
        **country,
        "entity_value": "Alice",
        "entity_label": "first_name",
        "protection_method_suggestion": "replace",
    }
    entries = [country, dict(country), latent, protected]
    payload: object = {"sensitivity_disposition": entries}
    if representation == "json":
        payload = json.dumps(payload)
    elif representation == "numpy":
        payload = {"sensitivity_disposition": np.array(entries, dtype=object)}
    trace = pd.DataFrame(
        [
            {cols.RECORD_ID_COLUMN: "record-a", cols.COL_SENSITIVITY_DISPOSITION: payload},
            {cols.RECORD_ID_COLUMN: "record-b", cols.COL_SENSITIVITY_DISPOSITION: payload},
        ],
        index=pd.Index([12, 99]),
    )
    findings = build_leave_as_is_findings(trace)
    assert findings is not None
    assert findings[cols.COL_FINDING_RECORD_INDEX].tolist() == [0, 0, 1, 1]
    assert findings[cols.COL_FINDING_ENTITY_VALUE].tolist() == ["USA", "adult", "USA", "adult"]
    assert findings[cols.COL_FINDING_REASON].tolist() == ["Broad geography", "Broad age group"] * 2
    assert findings[cols.COL_FINDING_SOURCE].tolist() == ["tagged", "latent"] * 2
    findings.at[0, cols.COL_FINDING_REASON] = "edited"
    fresh = build_leave_as_is_findings(trace)
    assert fresh is not None
    assert fresh.iloc[0][cols.COL_FINDING_REASON] == "Broad geography"


def test_missing_or_malformed_assignments_do_not_become_zero() -> None:
    assert build_leave_as_is_findings(pd.DataFrame()) is None
    for payload in [None, "broken", {"sensitivity_disposition": "broken"}]:
        assert build_leave_as_is_findings(pd.DataFrame([{cols.COL_SENSITIVITY_DISPOSITION: payload}])) is None
    empty = build_leave_as_is_findings(
        pd.DataFrame([{cols.COL_SENSITIVITY_DISPOSITION: {"sensitivity_disposition": []}}])
    )
    assert empty is not None and empty.empty
