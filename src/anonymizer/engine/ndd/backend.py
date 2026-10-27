# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The DataDesigner execution backend: one adapter, one run boundary, input positions and request counts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

import pandas as pd
from data_designer.config.column_types import ColumnConfigT
from data_designer.config.models import ModelConfig

from anonymizer.engine.ndd.adapter import FailedRecord, NddAdapter
from anonymizer.engine.row_partitioning import resolve_source_positions

T = TypeVar("T")


@dataclass(frozen=True)
class BackendRun:
    """One backend run: the surviving frame, the input position of each surviving row and the observed requests."""

    dataframe: pd.DataFrame
    failed_records: list[FailedRecord]
    source_positions: tuple[int, ...]
    observed_requests: int | None


class DataDesignerBackend:
    """Owns one :class:`NddAdapter` and is the single place a graph step runs DataDesigner workflows."""

    def __init__(self, adapter: NddAdapter) -> None:
        self._adapter = adapter

    def run(
        self,
        dataframe: pd.DataFrame,
        *,
        columns: list[ColumnConfigT],
        workflow_name: str,
        model_configs: list[ModelConfig],
        preview_num_records: int | None = None,
    ) -> BackendRun:
        """Run one workflow; ``observed_requests`` is ``None`` when the usage probe cannot tell."""
        result = self._adapter.run_workflow(
            dataframe,
            model_configs=model_configs,
            columns=columns,
            workflow_name=workflow_name,
            preview_num_records=preview_num_records,
        )
        positions = resolve_source_positions(dataframe, result.dataframe, result.source_positions)
        return BackendRun(result.dataframe, result.failed_records, positions, result.observed_requests)


def per_target(items: Sequence[T], positions: Sequence[int], size: int) -> list[T | None]:
    """One entry per input position: the item of the surviving row at that position, ``None`` for a dropped row."""
    aligned: list[T | None] = [None] * size
    for position, item in zip(positions, items):
        aligned[position] = item
    return aligned
