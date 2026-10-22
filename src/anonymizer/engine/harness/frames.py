# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Domain frame content retained by an execution, with semantic text owned by artifacts.

Opaque object cells retain pandas' existing reference semantics. This is content
ownership, not a serialization guarantee or a protection assessment. Projection
copies the frame structure and resolves transformed text from accepted artifacts.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from anonymizer.engine.detection.artifacts import EntitySpansValue
from anonymizer.engine.graph_sdk._graph_values import ActivationKey, ArtifactRef
from anonymizer.engine.graph_sdk.requests import AssociationResult, SemanticAssociation, TextArtifactValue
from anonymizer.engine.ndd.failed_record import FailedRecord
from anonymizer.engine.rewrite.artifacts import RewriteCandidateValue

if TYPE_CHECKING:
    from collections.abc import Mapping

    from anonymizer.engine.graph_sdk.executor import ExecutionPortFact, ExecutionResult


@dataclass(frozen=True, slots=True, repr=False)
class FrameStages:
    """Detection metadata retained alongside one canonical transformed frame."""

    detected_dataframe: pd.DataFrame
    detection_failures: tuple[FailedRecord, ...]
    replacement_failures: tuple[FailedRecord, ...]
    detection_elapsed: float
    replacement_elapsed: float

    def snapshot(self, opaque: tuple[object, ...]) -> FrameStages:
        if (
            type(self.detected_dataframe) is not pd.DataFrame
            or any(
                type(rows) is not tuple or any(type(row) is not FailedRecord for row in rows)
                for rows in (self.detection_failures, self.replacement_failures)
            )
            or any(
                type(value) not in (int, float) or not math.isfinite(value) or value < 0
                for value in (self.detection_elapsed, self.replacement_elapsed)
            )
        ):
            raise ValueError("malformed frame stage metadata")
        return replace(self, detected_dataframe=_copy_frame(self.detected_dataframe, opaque))


@dataclass(frozen=True, slots=True, repr=False)
class PublishedFrame:
    """Accepted frame content; transformed columns contain only canonical references."""

    _content: pd.DataFrame
    _columns: pd.Index
    _text: tuple[tuple[str, tuple[ArtifactRef, ...]], ...]
    failed_records: tuple[FailedRecord, ...]
    _opaque: tuple[object, ...]

    _sources: tuple[SemanticAssociation, ...]
    _ports: tuple[str, ...]
    stages: FrameStages | None = None
    _observation_dtypes: tuple[tuple[str, np.dtype | pd.api.extensions.ExtensionDtype], ...] = ()
    _json_columns: tuple[str, ...] = ()
    _accepted_json_payloads: tuple[tuple[str, tuple[str, ...]], ...] = ()
    # The frame's own observations, accepted as equal to the bound candidates.
    _accepted_candidates: tuple[tuple[str, tuple[RewriteCandidateValue, ...]], ...] = ()

    def __post_init__(self) -> None:
        # Detach the presentation dtypes from the publisher's objects once, at acceptance.
        object.__setattr__(self, "_observation_dtypes", deepcopy(self._observation_dtypes))

    def project(self, result: ExecutionResult) -> pd.DataFrame:
        if not any(self is frame for frame in result.frames):
            raise ValueError("frame belongs to a different execution")
        if (
            type(getattr(self, "_content", None)) is not pd.DataFrame
            or not isinstance(getattr(self, "_columns", None), pd.Index)
            or type(getattr(self, "_text", None)) is not tuple
            or not self._text
            or type(getattr(self, "failed_records", None)) is not tuple
            or any(type(item) is not FailedRecord for item in self.failed_records)
            or type(getattr(self, "_opaque", None)) is not tuple
            or type(getattr(self, "stages", None)) not in (type(None), FrameStages)
        ):
            raise ValueError("malformed retained frame")
        if (
            type(getattr(self, "_sources", None)) is not tuple
            or len(self._sources) != len(self._content)
            or any(type(source) is not SemanticAssociation for source in self._sources)
            or len(set(self._sources)) != len(self._sources)
            or type(getattr(self, "_ports", None)) is not tuple
            or len(self._ports) != len(self._text)
            or any(type(port) is not str or not port for port in self._ports)
        ):
            raise ValueError("malformed retained frame source binding")
        for binding in self._text:
            if (
                type(binding) is not tuple
                or len(binding) != 2
                or type(binding[0]) is not str
                or not binding[0]
                or type(binding[1]) is not tuple
                or len(binding[1]) != len(self._content)
                or any(type(reference) is not ArtifactRef for reference in binding[1])
            ):
                raise ValueError("malformed retained frame binding")
        columns = [column for column, _ in self._text]
        if (
            len(set(columns)) != len(columns)
            or any(list(self._columns).count(column) != 1 or column in self._content for column in columns)
            or not self._content.columns.equals(self._columns.drop(columns))
        ):
            raise ValueError("malformed retained frame columns")
        values = dict(result.artifacts)
        ports_by_source = {}
        for fact in result.ports:
            ports_by_source.setdefault((fact.activation, fact.port), []).append(fact)
        projected = _copy_frame(self._content, self._opaque)
        for (column, references), port in zip(self._text, self._ports, strict=True):
            for source, reference in zip(self._sources, references, strict=True):
                matches = ports_by_source.get((source.task.activation, port), ())
                if (
                    len(matches) != 1
                    or matches[0].artifact != reference
                    or port
                    not in {output.name for output in result._execution.nodes[matches[0].node].operation.outputs}
                ):
                    raise ValueError("frame reference differs from its accepted source output port")
            text = [values[reference] for reference in references]
            if any(type(value) not in (TextArtifactValue, EntitySpansValue, RewriteCandidateValue) for value in text):
                raise ValueError("frame reference has an unsupported artifact")
            accepted = dict(self._accepted_candidates).get(column, (None,) * len(text))
            observed = [
                # The legacy frame cell: equal to the candidate, in the frame's own type (42.0 for 42
                # once a withheld row upcast the column).
                cell.project()
                if isinstance(value, RewriteCandidateValue) and type(cell) is RewriteCandidateValue and cell == value
                else value.project()
                if isinstance(value, RewriteCandidateValue)
                else value.text
                if isinstance(value, TextArtifactValue)
                else {"entities": [entity.as_dict() for entity in cast(EntitySpansValue, value).entities]}
                for value, cell in zip(text, accepted, strict=True)
            ]
            dtype = dict(self._observation_dtypes).get(column)
            if column in self._json_columns:
                if tuple(observed) != dict(self._accepted_json_payloads).get(column):
                    raise ValueError("JSON artifact differs from its accepted content")
                observed = [json.loads(cast(str, value)) for value in observed]
            if isinstance(dtype, pd.ArrowDtype):
                # Arrow rejects NaN in a string array (pandas 3's `str` missing marker); pandas 2
                # converted every scalar missing marker to an Arrow null.
                observed = [None if pd.api.types.is_scalar(value) and pd.isna(value) else value for value in observed]
            presented = pd.Series(observed, index=projected.index, dtype=dtype)
            if any(
                type(value) is RewriteCandidateValue
                and RewriteCandidateValue.capture(cell) != value
                and not _null_presentation(value, cell)
                for value, cell in zip(text, presented, strict=True)
            ):
                raise ValueError("frame presentation differs from its accepted candidate")
            projected[column] = presented
        return projected.reindex(columns=self._columns)


def _null_candidate(value: object) -> bool:
    return type(value) is RewriteCandidateValue and (value.missing or value == RewriteCandidateValue.capture(None))


def _coerced_null(accepted: object, observed: object) -> bool:
    """A null candidate the frame's column dtype now holds as its missing marker.

    pandas 3 infers its `str` dtype for a reassembled text column, turning a
    retained `None` into NaN, as the legacy frame did; both are the same null.
    """
    return _null_candidate(accepted) and type(observed) is RewriteCandidateValue and observed.missing


def _null_presentation(value: RewriteCandidateValue, cell: object) -> bool:
    """A null candidate presents as its column's missing marker, as the legacy output did.

    That marker is `pd.NA` in a nullable Arrow column and NaN in pandas 3's `str` column.
    """
    return _null_candidate(value) and pd.api.types.is_scalar(cell) and bool(pd.isna(cell))


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
    _text: tuple[tuple[str, str, tuple[str | EntitySpansValue | RewriteCandidateValue, ...]], ...]
    failed_records: tuple[FailedRecord, ...]
    _opaque: tuple[object, ...]
    stages: FrameStages | None = None
    _observation_dtypes: tuple[tuple[str, np.dtype | pd.api.extensions.ExtensionDtype], ...] = ()
    _json_columns: tuple[str, ...] = ()

    @classmethod
    def capture(
        cls,
        dataframe: pd.DataFrame,
        *,
        associations: tuple[SemanticAssociation, ...],
        text_ports: tuple[tuple[str, str], ...],
        failed_records: tuple[FailedRecord, ...],
        opaque_source: pd.DataFrame,
        span_ports: tuple[tuple[str, str, tuple[EntitySpansValue, ...]], ...] = (),
        observation_ports: tuple[tuple[str, str], ...] = (),
        observation_dtypes: Mapping[str, pd.api.extensions.ExtensionDtype] | None = None,
        stages: FrameStages | None = None,
        json_ports: tuple[tuple[str, str], ...] = (),
    ) -> FramePublication:
        if (
            len(dataframe) != len(associations)
            or len(set(associations)) != len(associations)
            or any(list(dataframe.columns).count(column) != 1 for column, _ in text_ports)
            or len({column for column, _ in text_ports}) != len(text_ports)
        ):
            raise ValueError("frame publication has ambiguous source or text columns")
        text = (
            tuple((column, port, tuple(dataframe[column])) for column, port in text_ports)
            + span_ports
            + tuple(
                (column, port, tuple(json.dumps(value) for value in dataframe[column])) for column, port in json_ports
            )
            + tuple(
                (column, port, tuple(RewriteCandidateValue.capture(value) for value in dataframe[column]))
                for column, port in observation_ports
            )
        )
        if not text or any(
            type(value) not in (str, EntitySpansValue, RewriteCandidateValue)
            for _, _, values in text
            for value in values
        ):
            raise ValueError("frame publication requires semantic text outputs")
        opaque = tuple(value for row in opaque_source.itertuples(index=False, name=None) for value in row)
        return cls(
            _copy_frame(dataframe.drop(columns=[column for column, _, _ in text]), opaque),
            dataframe.columns.copy(),
            associations,
            text,
            failed_records,
            opaque,
            stages.snapshot(opaque) if stages is not None else None,
            tuple(
                (column, (observation_dtypes or {}).get(column, dataframe[column].dtype))
                for column, _ in (*text_ports, *observation_ports)
            ),
            tuple(column for column, _ in json_ports),
        )

    def accept(self, results: tuple[AssociationResult, ...], *, opaque: tuple[object, ...]) -> FramePublication | None:
        """Validate raw callback fields and transfer a detached snapshot to the owner."""
        if (
            type(getattr(self, "_content", None)) is not pd.DataFrame
            or not isinstance(getattr(self, "_columns", None), pd.Index)
            or type(getattr(self, "associations", None)) is not tuple
            or any(type(item) is not SemanticAssociation for item in self.associations)
            or len(set(self.associations)) != len(self.associations)
            or len(self._content) != len(self.associations)
            or type(getattr(self, "_text", None)) is not tuple
            or not self._text
            or type(getattr(self, "failed_records", None)) is not tuple
            or any(type(item) is not FailedRecord for item in self.failed_records)
            or type(getattr(self, "_opaque", None)) is not tuple
            or type(getattr(self, "stages", None)) not in (type(None), FrameStages)
        ):
            return None
        for item in self._text:
            if (
                type(item) is not tuple
                or len(item) != 3
                or type(item[0]) is not str
                or type(item[1]) is not str
                or not item[0]
                or not item[1]
                or type(item[2]) is not tuple
                or len(item[2]) != len(self.associations)
                or any(type(value) not in (str, EntitySpansValue, RewriteCandidateValue) for value in item[2])
            ):
                return None
        names = [column for column, _, _ in self._text]
        if (
            len(set(names)) != len(names)
            or any(list(self._columns).count(name) != 1 or name in self._content.columns for name in names)
            or not self._content.columns.equals(self._columns.drop(names))
        ):
            return None
        by_source = {result.association: result for result in results}
        if len(by_source) != len(results) or set(by_source) != set(self.associations):
            return None
        for _, port, values in self._text:
            for association, text in zip(self.associations, values, strict=True):
                outputs = [output for output in by_source[association].outputs if output.port == port]
                if (
                    len(outputs) != 1
                    or outputs[0].value != (TextArtifactValue(text=text) if type(text) is str else text)
                    and not _coerced_null(outputs[0].value, text)
                ):
                    return None
        return FramePublication(
            _copy_frame(self._content, opaque),
            self._columns.copy(),
            self.associations,
            self._text,
            self.failed_records,
            opaque,
            self.stages.snapshot(opaque) if self.stages is not None else None,
            self._observation_dtypes,
            self._json_columns,
        )

    def bind(self, ports: tuple[ExecutionPortFact, ...]) -> PublishedFrame | None:
        if not self._text:
            return None
        ports_by_source = {}
        for item in ports:
            ports_by_source.setdefault((item.activation, item.port), []).append(item.artifact)
        text = []
        for column, port, _ in self._text:
            references = []
            for association in self.associations:
                matches = ports_by_source.get((association.task.activation, port), ())
                if len(matches) != 1:
                    return None
                references.append(matches[0])
            text.append((column, tuple(references)))
        return PublishedFrame(
            self._content,
            self._columns,
            tuple(text),
            self.failed_records,
            self._opaque,
            self.associations,
            tuple(port for _, port, _ in self._text),
            self.stages,
            self._observation_dtypes,
            self._json_columns,
            tuple(
                (column, cast(tuple[str, ...], values))
                for column, _, values in self._text
                if column in self._json_columns
            ),
            tuple(
                (column, cast(tuple[RewriteCandidateValue, ...], values))
                for column, _, values in self._text
                if values and all(type(value) is RewriteCandidateValue for value in values)
            ),
        )


@dataclass(frozen=True, slots=True, repr=False)
class InvocationFrame:
    """Opaque compatibility content owned by one execution, without row provenance."""

    dataframe: pd.DataFrame
    detected_dataframe: pd.DataFrame
    detection_failures: tuple[FailedRecord, ...]
    replacement_failures: tuple[FailedRecord, ...]
    detection_elapsed: float
    replacement_elapsed: float
    retained_owner: object | None = None

    def _validate(self) -> None:
        if (
            not isinstance(getattr(self, "dataframe", None), pd.DataFrame)
            or not isinstance(getattr(self, "detected_dataframe", None), pd.DataFrame)
            or any(
                type(getattr(self, name, None)) is not tuple
                or any(type(item) is not FailedRecord for item in getattr(self, name))
                for name in ("detection_failures", "replacement_failures")
            )
            or any(
                type(getattr(self, name, None)) not in (int, float)
                or not math.isfinite(getattr(self, name))
                or getattr(self, name) < 0
                for name in ("detection_elapsed", "replacement_elapsed")
            )
            or not hasattr(self, "retained_owner")
        ):
            raise ValueError("malformed opaque frame")

    def accept(self) -> InvocationFrame:
        self._validate()
        return replace(self, dataframe=self.dataframe.copy(), detected_dataframe=self.detected_dataframe.copy())

    @property
    def row_counts(self) -> tuple[int, int]:
        return len(self.detected_dataframe), len(self.dataframe)

    @property
    def text_bytes(self) -> int:
        """Observed direct string cells; arbitrary object-cell sizes remain unavailable."""
        from anonymizer.engine.graph_sdk._artifact_shape import content_bytes

        return sum(
            content_bytes(value)
            for frame in (self.detected_dataframe, self.dataframe)
            for row in frame.itertuples(index=False, name=None)
            for value in row
            if isinstance(value, str)
        )

    @property
    def failed_records(self) -> tuple[FailedRecord, ...]:
        return self.detection_failures + self.replacement_failures

    def bind(self, ports: tuple[ExecutionPortFact, ...]) -> InvocationFrame:
        if ports:
            raise ValueError("opaque invocation cannot publish semantic ports")
        self._validate()
        return self

    def project(self, result: ExecutionResult) -> pd.DataFrame:
        if not any(self is frame for frame in result.frames):
            raise ValueError("frame belongs to a different execution")
        if result._execution.context.prepared.configuration.purpose != "execution_only":
            raise ValueError("opaque frame cannot establish protection")
        self._validate()
        return self.dataframe.copy()


@dataclass(frozen=True, slots=True)
class AggregateFramePublication:
    """Keep an opaque callback frame separate from its owner's internal observations."""

    frame: InvocationFrame
    internal_activations: frozenset[ActivationKey]

    def bind(self, ports: tuple[ExecutionPortFact, ...]) -> InvocationFrame:
        return self.frame.bind(tuple(port for port in ports if port.activation not in self.internal_activations))
