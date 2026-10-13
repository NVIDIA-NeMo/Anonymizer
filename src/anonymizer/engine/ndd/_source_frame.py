# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source correspondence for one whole-frame DataDesigner submission."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import pandas as pd

from anonymizer.interface.errors import AnonymizerWorkflowError

_COL_SOURCE_PREFIX = "_anonymizer_source_"


@dataclass(frozen=True)
class WorkflowSourceFrame:
    dataframe: pd.DataFrame
    key_column: str
    keys: tuple[str, ...]

    @classmethod
    def capture(cls, dataframe: pd.DataFrame) -> WorkflowSourceFrame:
        key_column = f"{_COL_SOURCE_PREFIX}{uuid.uuid4().hex}"
        while key_column in dataframe.columns:
            key_column = f"{_COL_SOURCE_PREFIX}{uuid.uuid4().hex}"
        return cls(dataframe.copy(), key_column, tuple(uuid.uuid4().hex for _ in range(len(dataframe))))

    def seed(self) -> pd.DataFrame:
        seed = self.dataframe.copy()
        seed[self.key_column] = self.keys
        return seed

    def project(self, returned: pd.DataFrame, *, requested: int | None = None) -> tuple[pd.DataFrame, tuple[int, ...]]:
        """Validate one result per source and remove private transport identity."""
        if returned.empty:
            return returned.drop(columns=self.key_column, errors="ignore"), ()
        if list(returned.columns).count(self.key_column) != 1:
            raise AnonymizerWorkflowError("Workflow output has no source correspondence")
        keys = returned[self.key_column].tolist()
        source_positions = {key: index for index, key in enumerate(self.keys[:requested])}
        if any(not isinstance(key, str) or key not in source_positions for key in keys):
            raise AnonymizerWorkflowError("Workflow output has invalid source correspondence")
        if len(set(keys)) != len(keys):
            raise AnonymizerWorkflowError("Workflow output has duplicate source correspondence")
        order = sorted(range(len(keys)), key=lambda index: source_positions[keys[index]])
        positions = tuple(source_positions[keys[index]] for index in order)
        return returned.iloc[order].drop(columns=self.key_column).reset_index(drop=True), positions
