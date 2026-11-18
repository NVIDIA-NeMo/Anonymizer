# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read saved Rewrite exemptions for reports without running models."""

from __future__ import annotations

from enum import Enum
from typing import Any

import pandas as pd

from anonymizer.engine import constants as cols
from anonymizer.engine.evaluation.entity_sources import entity_source
from anonymizer.engine.rewrite.parsers import normalize_payload


def build_leave_as_is_findings(trace: pd.DataFrame) -> pd.DataFrame | None:
    """Return saved leave_as_is assignments, or None when data is unavailable.

    Count each value/label pair once per record, including latent entities.
    Repeated spans and the same assignment's repeated serialization are not
    separate entities. The same value in different records counts separately.
    """
    columns = [
        cols.COL_FINDING_RECORD_INDEX,
        cols.COL_FINDING_RECORD_ID,
        cols.COL_FINDING_ENTITY_VALUE,
        cols.COL_FINDING_ENTITY_LABEL,
        cols.COL_FINDING_SOURCE,
        cols.COL_FINDING_CATEGORY,
        cols.COL_FINDING_SENSITIVITY,
        cols.COL_FINDING_REASON,
    ]
    if cols.COL_SENSITIVITY_DISPOSITION not in trace:
        return None
    findings: list[dict[str, Any]] = []
    available = trace.empty
    for index, (_, row) in enumerate(trace.iterrows()):
        raw = row.get(cols.COL_SENSITIVITY_DISPOSITION)
        if raw is None or raw is pd.NA or (isinstance(raw, float) and pd.isna(raw)):
            continue
        payload = normalize_payload(raw)
        items = payload.get("sensitivity_disposition") if isinstance(payload, dict) else payload
        if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
            return None
        available = True
        seen: set[tuple[str, str]] = set()
        for item in items:
            if item.get("protection_method_suggestion") != "leave_as_is":
                continue
            value, label = item.get("entity_value"), item.get("entity_label")
            if not isinstance(value, str) or not isinstance(label, str):
                return None
            if (value, label) in seen:
                continue
            seen.add((value, label))
            category, sensitivity = item.get("category"), item.get("sensitivity")
            findings.append(
                {
                    cols.COL_FINDING_RECORD_INDEX: index,
                    cols.COL_FINDING_RECORD_ID: row.get(cols.RECORD_ID_COLUMN),
                    cols.COL_FINDING_ENTITY_VALUE: value,
                    cols.COL_FINDING_ENTITY_LABEL: label,
                    cols.COL_FINDING_SOURCE: entity_source(row, value, label, fallback=item.get("source")),
                    cols.COL_FINDING_CATEGORY: category.value if isinstance(category, Enum) else category,
                    cols.COL_FINDING_SENSITIVITY: sensitivity.value if isinstance(sensitivity, Enum) else sensitivity,
                    cols.COL_FINDING_REASON: item.get("low_sensitivity_reason")
                    or item.get("protection_reason")
                    or item.get("reason")
                    or "Unavailable",
                }
            )
    return pd.DataFrame(findings, columns=pd.Index(columns), dtype=object) if available else None
