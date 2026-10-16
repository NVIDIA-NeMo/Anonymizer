# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from tests.graph_sdk.test_qualification_admission_conformance import (
    test_qualification_declaration_rejected_before_callbacks as declaration_compare,
)
from tests.graph_sdk.test_qualification_admission_conformance import (
    test_reference_atomic_groups_at_data_admission as atomic_compare,
)

CASES = json.loads(Path(__file__).with_name("qualification-admission-extracted.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case_id"])
def test_extracted_qualification_admission(case: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    if case["case_id"].endswith(("e7f5b46c70d8", "07ac08474ff0")):
        translated = deepcopy(case)
        translated["expected"] = {"status": "rejected", "code": "protection_ineligible"}
        declaration_compare(translated, monkeypatch)
    else:
        atomic_compare(case)
