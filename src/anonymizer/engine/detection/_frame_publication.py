# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Domain frame content retained by an execution, with semantic text owned by artifacts.

Opaque object cells retain pandas' existing reference semantics. This is content
ownership, not a serialization guarantee or a protection assessment. Projection
copies the frame structure and resolves transformed text from accepted artifacts.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pandas as pd

from anonymizer.engine.graph_sdk.requests import AssociationResult, SemanticAssociation, TextArtifactValue
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.graph._values import ArtifactRef

if TYPE_CHECKING:
    from anonymizer.engine.graph_sdk.executor import ExecutionPortFact, ExecutionResult


@dataclass(frozen=True, slots=True, repr=False)
class PublishedFrame:
    """Accepted frame content; transformed columns contain only canonical references."""

    _content: pd.DataFrame
    _columns: pd.Index
    _text: tuple[tuple[str, tuple[ArtifactRef, ...]], ...]
    failed_records: tuple[FailedRecord, ...]
    _opaque: tuple[object, ...]

    def project(self, result: ExecutionResult) -> pd.DataFrame:
        if not any(self is frame for frame in result.frames):
            raise ValueError("frame belongs to a different execution")
        values = dict(result.artifacts)
        projected = _copy_frame(self._content, self._opaque)
        for column, references in self._text:
            text = [values[reference] for reference in references]
            if any(type(value) is not TextArtifactValue for value in text):
                raise ValueError("frame text reference has a non-text artifact")
            projected[column] = [cast(TextArtifactValue, value).text for value in text]
        return projected.reindex(columns=self._columns)


def _copy_frame(frame: pd.DataFrame, opaque: tuple[object, ...]) -> pd.DataFrame:
    copied = frame.copy()
    memo = {id(value): value for value in opaque}
    for ordinal, dtype in enumerate(frame.dtypes):
        if dtype == object:
            for row in range(len(frame)):
                copied.iat[row, ordinal] = deepcopy(frame.iat[row, ordinal], memo)
    return copied


@dataclass(frozen=True, slots=True, repr=False)
class FramePublication:
    """Whole-frame candidate tied to exactly the submitting source associations."""

    _content: pd.DataFrame
    _columns: pd.Index
    associations: tuple[SemanticAssociation, ...]
    _text: tuple[tuple[str, str, tuple[str, ...]], ...]
    failed_records: tuple[FailedRecord, ...]
    _opaque: tuple[object, ...]

    @classmethod
    def capture(
        cls,
        dataframe: pd.DataFrame,
        *,
        associations: tuple[SemanticAssociation, ...],
        text_ports: tuple[tuple[str, str], ...],
        failed_records: tuple[FailedRecord, ...],
        opaque_source: pd.DataFrame,
    ) -> FramePublication:
        if (
            len(dataframe) != len(associations)
            or len(set(associations)) != len(associations)
            or any(list(dataframe.columns).count(column) != 1 for column, _ in text_ports)
            or len({column for column, _ in text_ports}) != len(text_ports)
        ):
            raise ValueError("frame publication has ambiguous source or text columns")
        text = tuple((column, port, tuple(dataframe[column])) for column, port in text_ports)
        if not text or any(type(value) is not str for _, _, values in text for value in values):
            raise ValueError("frame publication requires semantic text outputs")
        opaque = tuple(value for row in opaque_source.itertuples(index=False, name=None) for value in row)
        return cls(
            _copy_frame(dataframe.drop(columns=[column for column, _ in text_ports]), opaque),
            dataframe.columns.copy(),
            associations,
            text,
            failed_records,
            opaque,
        )

    def matches(self, results: tuple[AssociationResult, ...]) -> bool:
        by_source = {result.association: result for result in results}
        if len(by_source) != len(results) or set(by_source) != set(self.associations):
            return False
        for _, port, values in self._text:
            for association, text in zip(self.associations, values, strict=True):
                outputs = [output for output in by_source[association].outputs if output.port == port]
                if (
                    len(outputs) != 1
                    or type(outputs[0].value) is not TextArtifactValue
                    or outputs[0].value.text != text
                ):
                    return False
        return True

    def bind(self, ports: tuple[ExecutionPortFact, ...]) -> PublishedFrame | None:
        text = []
        for column, port, _ in self._text:
            references = []
            for association in self.associations:
                matches = [
                    item.artifact
                    for item in ports
                    if item.activation == association.task.activation and item.port == port
                ]
                if len(matches) != 1:
                    return None
                references.append(matches[0])
            text.append((column, tuple(references)))
        return PublishedFrame(self._content, self._columns, tuple(text), self.failed_records, self._opaque)
