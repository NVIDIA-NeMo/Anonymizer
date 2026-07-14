# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base class for LLM-as-judge workflows.

Each judge follows the same shape: ``prepare`` adds intermediate columns the
prompt references, the LLM runs as one DataDesigner column, then
``postprocess`` flattens the raw payload into a boolean ``*_valid`` column
plus a list of invalid entries. The standalone ``evaluate`` entry point wraps
that pipeline for callers that want to run a single judge in isolation.

Subclasses declare only what's actually unique to a judge — column names,
schema, model role, prompt builder, and the per-row passthrough rule.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Protocol, cast

import pandas as pd
from data_designer.config.column_configs import LLMStructuredColumnConfig
from data_designer.config.column_types import ColumnConfigT
from data_designer.config.models import ModelConfig
from pydantic import BaseModel

from anonymizer.config.models import EvaluateModelSelection
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.engine.ndd.model_loader import resolve_model_alias
from anonymizer.engine.row_partitioning import ROW_ORDER_COL, merge_and_reorder

logger = logging.getLogger("anonymizer.evaluation.judge_base")


@dataclass(frozen=True)
class JudgeResult:
    """Result of a standalone single-judge ``evaluate()`` call."""

    dataframe: pd.DataFrame
    failed_records: list[FailedRecord]


class _JudgeRunResult(Protocol):
    @property
    def dataframe(self) -> pd.DataFrame: ...

    @property
    def failed_records(self) -> list[FailedRecord]: ...


class _JudgeAdapter(Protocol):
    def run_workflow(
        self,
        dataframe: pd.DataFrame,
        /,
        *,
        model_configs: list[ModelConfig],
        columns: list[ColumnConfigT],
        workflow_name: str,
        preview_num_records: int | None = None,
    ) -> _JudgeRunResult: ...


class _BaseJudgeWorkflow(ABC):
    """Common scaffolding for the four LLM-as-judge workflows."""

    # Column names this judge reads and writes.
    RAW_COL: ClassVar[str]
    VALID_COL: ClassVar[str]
    INVALID_COL: ClassVar[str]

    # Structured-output schema and the verdict field name on it.
    SCHEMA: ClassVar[type[BaseModel]]
    VERDICT_FIELD: ClassVar[str]

    # Payload used to stamp passthrough rows so display logic stays uniform.
    DEFAULT_PAYLOAD: ClassVar[dict]

    # Model alias role consulted on EvaluateModelSelection.
    MODEL_ROLE: ClassVar[str]

    # Logical workflow name surfaced in logs and FailedRecord entries.
    WORKFLOW_NAME: ClassVar[str]

    def __init__(self, adapter: _JudgeAdapter) -> None:
        self._adapter = adapter

    # ------------------------------------------------------------------ hooks

    @abstractmethod
    def prepare(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``dataframe`` with the intermediate columns this
        judge's prompt template references."""

    @abstractmethod
    def _passthrough_mask(self, dataframe: pd.DataFrame) -> pd.Series:
        """Boolean Series — True for rows that trivially pass (no checkable content)."""

    @classmethod
    @abstractmethod
    def _build_prompt(cls) -> str:
        """Return the column prompt. Called per ``column_config()`` so dynamic
        values (e.g. current year) are resolved at evaluate time."""

    @classmethod
    @abstractmethod
    def _extract_invalid(cls, parsed: BaseModel) -> list[dict[str, object]]:
        """Extract the invalid-entries list from a parsed schema instance."""

    # ----------------------------------------------------------------- shared

    def column_config(self, selected_models: EvaluateModelSelection) -> LLMStructuredColumnConfig:
        return LLMStructuredColumnConfig(
            name=self.RAW_COL,
            prompt=self._build_prompt(),
            model_alias=resolve_model_alias(self.MODEL_ROLE, selected_models),
            output_format=self.SCHEMA,
        )

    @classmethod
    def _flatten_judgment(cls, raw: object) -> tuple[bool | None, list[dict[str, object]]]:
        """Normalize an LLM judge output into ``(verdict, invalid_entries)``.

        Returns ``(None, [])`` for any malformed or missing payload so downstream
        display renders "judge unavailable" rather than fabricating a verdict.
        """
        if raw is None:
            return None, []
        if isinstance(raw, BaseModel):
            raw = raw.model_dump(mode="python")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                return None, []
        if not isinstance(raw, dict):
            return None, []
        try:
            parsed = cls.SCHEMA.model_validate(raw)
        except Exception:
            return None, []
        return getattr(parsed, cls.VERDICT_FIELD), cls._extract_invalid(parsed)

    def postprocess(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Flatten the raw judge output into VALID / INVALID columns and apply
        the passthrough default (rows with no checkable content trivially pass).
        """
        out = dataframe.copy()
        flattened = out[self.RAW_COL].apply(self._flatten_judgment) if self.RAW_COL in out.columns else None
        passthrough_mask = self._passthrough_mask(out)

        valid: list[bool | None] = []
        invalid: list[list[dict[str, object]]] = []
        for idx in out.index:
            if passthrough_mask.loc[idx]:
                valid.append(True)
                invalid.append([])
            elif flattened is not None:
                v, inv = flattened.loc[idx]
                valid.append(v)
                invalid.append(inv)
            else:
                valid.append(None)
                invalid.append([])
        out[self.VALID_COL] = valid
        out[self.INVALID_COL] = invalid
        if self.RAW_COL in out.columns:
            out.loc[passthrough_mask, self.RAW_COL] = [self.DEFAULT_PAYLOAD] * int(passthrough_mask.sum())
        return out

    def evaluate(
        self,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: EvaluateModelSelection,
        preview_num_records: int | None = None,
    ) -> JudgeResult:
        """Standalone single-judge entry point. The orchestrator in
        ``ReplacementWorkflow`` does not go through this; tests and callers
        that want to run one judge in isolation do.
        """
        working_df = self.prepare(dataframe)
        # `prepare()` returns a fresh copy (per its contract), so we can stamp
        # the row-order column directly. ROW_ORDER_COL lets merge_and_reorder
        # restore input order after the passthrough and LLM-judged partitions
        # are processed independently.
        working_df[ROW_ORDER_COL] = range(len(working_df))
        passthrough_mask = self._passthrough_mask(working_df)
        passthrough_rows = working_df[passthrough_mask].copy()
        with_content = working_df[~passthrough_mask].copy()

        passthrough_rows[self.RAW_COL] = [self.DEFAULT_PAYLOAD for _ in range(len(passthrough_rows))]
        passthrough_rows[self.VALID_COL] = True
        passthrough_rows[self.INVALID_COL] = [[] for _ in range(len(passthrough_rows))]
        if not passthrough_rows.empty:
            # Mirrors the rewrite-mode passthrough log: rows with no entities trivially pass.
            logger.info(
                "%d passthrough row(s) have no detected entities — detection_valid set to True (trivially valid).",
                len(passthrough_rows),
            )

        if with_content.empty:
            combined = merge_and_reorder(passthrough_rows)
            return JudgeResult(dataframe=combined, failed_records=[])

        effective_preview_num_records = (
            min(preview_num_records, len(with_content)) if preview_num_records is not None else None
        )
        run_result = self._adapter.run_workflow(
            with_content,
            model_configs=model_configs,
            columns=cast(list[ColumnConfigT], [self.column_config(selected_models)]),
            workflow_name=self.WORKFLOW_NAME,
            preview_num_records=effective_preview_num_records,
        )

        judged_df = run_result.dataframe.copy()
        flattened = judged_df[self.RAW_COL].apply(self._flatten_judgment)
        judged_df[self.VALID_COL] = flattened.apply(lambda pair: pair[0])
        judged_df[self.INVALID_COL] = flattened.apply(lambda pair: pair[1])

        combined = merge_and_reorder(judged_df, passthrough_rows)
        return JudgeResult(dataframe=combined, failed_records=run_result.failed_records)
