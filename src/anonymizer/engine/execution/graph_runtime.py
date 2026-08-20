# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lower the private trivial graph profile through the pandas backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from anonymizer.engine.constants import COL_TEXT
from anonymizer.engine.execution.graph import (
    _compile_trivial_graph,
    _CompiledTrivialGraph,
    _DatumId,
    _GraphLimits,
)
from anonymizer.engine.execution.invocation import _CompiledInvocation
from anonymizer.engine.execution.pandas_runtime import _PandasExecutionResult
from anonymizer.engine.private_row_verification import (
    PRIVATE_CORRELATION_COLUMN,
    PrivateRowVerificationError,
    _InvocationRowVerifier,
)


class _FrameExecutionBackend(Protocol):
    """Private effect boundary implemented by the current pandas runtime."""

    def run(
        self,
        dataframe: pd.DataFrame,
        *,
        invocation: _CompiledInvocation,
        data_summary: str | None,
        preview_num_records: int | None,
        verifier: _InvocationRowVerifier,
    ) -> _PandasExecutionResult: ...


@dataclass(frozen=True, slots=True, repr=False)
class _GraphExecutionResult:
    datum_ids: tuple[_DatumId, ...]
    input_texts: tuple[str, ...]
    dataframe_result: _PandasExecutionResult
    datum_row_tokens: tuple[str, ...]

    def __repr__(self) -> str:
        return "<private graph execution result>"


class _TrivialGraphRuntime:
    """Execute independently scoped graph datums through one frame backend."""

    def __init__(self, backend: _FrameExecutionBackend) -> None:
        self._backend = backend

    def run(
        self,
        graph: object,
        *,
        limits: _GraphLimits,
        invocation: _CompiledInvocation,
        data_summary: str | None,
        preview_num_records: int | None,
    ) -> _GraphExecutionResult:
        compiled = _compile_trivial_graph(graph, limits=limits)
        frame = self._lower(compiled)
        verifier = _InvocationRowVerifier(frame)
        bound = verifier.bind(frame)
        datum_row_tokens = tuple(bound[PRIVATE_CORRELATION_COLUMN])
        try:
            result = self._backend.run(
                bound,
                invocation=invocation,
                data_summary=data_summary,
                preview_num_records=preview_num_records,
                verifier=verifier,
            )
        except KeyboardInterrupt:
            verifier.abort(cancelled=True)
            raise
        except PrivateRowVerificationError:
            verifier.abort(cancelled=False)
            raise
        except Exception as cause:
            error = verifier.abort_with_failure(stage="pipeline", cause=cause)
            del cause
            raise error from None
        return _GraphExecutionResult(
            tuple(datum.id for datum in compiled.datums),
            tuple(datum.text for datum in compiled.datums),
            result,
            datum_row_tokens,
        )

    @staticmethod
    def _lower(compiled: _CompiledTrivialGraph) -> pd.DataFrame:
        return pd.DataFrame({COL_TEXT: [datum.text for datum in compiled.datums]})
