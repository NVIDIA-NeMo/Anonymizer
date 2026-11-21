# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The DataDesigner execution backend: one adapter, one run boundary, input positions and request counts."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import pandas as pd
from data_designer.config.column_types import ColumnConfigT
from data_designer.config.models import ModelConfig
from pydantic_core import to_jsonable_python

from anonymizer.engine.errors import AnonymizerWorkflowError
from anonymizer.engine.graph_core.authoring import GraphScope, Limits, Operation, ProtectionGraph, Workflow
from anonymizer.engine.graph_core.execution import ProtectionResult, prepare, protect
from anonymizer.engine.graph_core.kernel import register_kind
from anonymizer.engine.graph_core.session import Session
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
    """Owns one :class:`NddAdapter`; ``LlmReplaceWorkflow`` runs its DataDesigner workflow through it."""

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


def _numpy_plain(value: object) -> object:
    """pydantic-core's fallback: a numpy array or scalar as Python; dates and durations (``tolist`` can give an int at
    nanoseconds) and every other unknown type have no plain form."""
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind not in "mM":
        return value.tolist()
    raise TypeError(f"{type(value).__name__} has no plain Python form")


def _plain(value: Any) -> Any:
    """``value`` converted deeply to JSON-compatible Python by pydantic-core, NaN and infinity as ``None``; else
    ``TypeError``."""
    try:
        return to_jsonable_python(value, inf_nan_mode="null", fallback=_numpy_plain)
    except ValueError as exc:  # pydantic-core's serialization error, or bytes that are not UTF-8
        raise TypeError(f"{type(value).__name__} has no plain Python form") from exc


def register_plain_kind(name: str) -> str:
    """Register ``name`` as a dict kind copied by ``_plain``, so ``json.dumps(..., allow_nan=False)`` accepts its values;
    a value with no plain form fails its target as ``invalid_output``."""
    return register_kind(name, lambda v: isinstance(v, dict), copy=_plain)


def per_target(items: Sequence[T], positions: Sequence[int], size: int) -> list[T | None]:
    """One entry per input position: the item of the surviving row at that position, ``None`` for a dropped row."""
    aligned: list[T | None] = [None] * size
    for position, item in zip(positions, items):
        aligned[position] = item
    return aligned


def call_on_empty_input(graph: ProtectionGraph, *operations: Operation) -> None:
    """With no targets the operations never ran; call each on the empty input, so its callee still sees it."""
    if not graph.targets:
        for operation in operations:
            operation.fn(())


def raise_if_cancelled(result: ProtectionResult) -> None:
    """Raise ``AnonymizerWorkflowError`` when a cancellation stopped any target before its steps finished."""
    if any(result.outcome(target).terminal == "cancelled" for target in result.targets):
        raise AnonymizerWorkflowError("The workflow was cancelled before it finished.")


def rows_as_graph(texts: Iterable[object], *, none_as_empty: bool = False) -> ProtectionGraph:
    """One target per row whose datum is ``str(text)``; ``none_as_empty`` writes ``""`` for a ``None`` text."""
    scope = GraphScope()
    return ProtectionGraph(
        scope, tuple(scope.datum("" if none_as_empty and text is None else str(text)) for text in texts)
    )


def run_invocation(
    graph: ProtectionGraph,
    workflow: Workflow,
    limits: Limits,
    *,
    tolerate: tuple[type[BaseException], ...] = (),
    on_empty: Sequence[Operation] = (),
    session: Session | None = None,
) -> ProtectionResult:
    """Run one invocation; raise the first non-verifier error not of a ``tolerate`` type, then a cancellation.
    ``on_empty`` runs as ``call_on_empty_input`` does; ``session`` is passed to the run."""
    result = protect(prepare(graph, workflow, limits=limits, session=session), session=session)
    call_on_empty_input(graph, *on_empty)
    steps = [step for _, step in workflow.flatten()]
    for entry in result.state.entries:
        if entry.error is not None and not steps[entry.node].is_verifier and not isinstance(entry.error, tolerate):
            raise entry.error
    raise_if_cancelled(result)
    return result
