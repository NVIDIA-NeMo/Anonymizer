# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source correspondence for one whole-frame DataDesigner submission."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from anonymizer.interface.errors import AnonymizerWorkflowError

if TYPE_CHECKING:
    from anonymizer.engine.graph_sdk.requests import SemanticAssociation


@dataclass(frozen=True, slots=True)
class SourceWork:
    association: SemanticAssociation
    column: str


_SOURCE_WORK: ContextVar[SourceWork | None] = ContextVar("anonymizer_dd_source_work", default=None)


def current_source_work() -> SourceWork | None:
    return _SOURCE_WORK.get()


_COL_SOURCE_PREFIX = "_anonymizer_source_"


@dataclass(frozen=True)
class WorkflowSourceFrame:
    dataframe: pd.DataFrame
    key_column: str
    keys: tuple[str, ...]
    associations: tuple[SemanticAssociation, ...] = ()

    @classmethod
    def capture(
        cls, dataframe: pd.DataFrame, *, associations: tuple[SemanticAssociation, ...] = ()
    ) -> WorkflowSourceFrame:
        if associations and (len(associations) != len(dataframe) or len(set(associations)) != len(associations)):
            raise AnonymizerWorkflowError("Workflow source associations do not match the frame")
        key_column = f"{_COL_SOURCE_PREFIX}{uuid.uuid4().hex}"
        while key_column in dataframe.columns:
            key_column = f"{_COL_SOURCE_PREFIX}{uuid.uuid4().hex}"
        return cls(dataframe.copy(), key_column, tuple(uuid.uuid4().hex for _ in range(len(dataframe))), associations)

    def seed(self) -> pd.DataFrame:
        seed = self.dataframe.copy()
        seed[self.key_column] = self.keys
        return seed

    def project(self, returned: pd.DataFrame, *, requested: int | None = None) -> tuple[pd.DataFrame, tuple[int, ...]]:
        """Validate one result per source and remove private transport identity."""
        if len(returned) == 0:
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

    @contextmanager
    def bind_generators(self, designer: Any, columns: frozenset[str]) -> Iterator[set[int]]:
        """Bind real DD row generators to issued associations, without exposing transport keys."""
        failed: set[int] = set()
        if not self.associations:
            yield failed
            return
        original_builder = designer._create_dataset_builder
        sources = {
            key: (index, association)
            for index, (key, association) in enumerate(zip(self.keys, self.associations, strict=True))
        }

        def builder(*args: Any, **kwargs: Any) -> Any:
            result = original_builder(*args, **kwargs)
            original_initialize = result._initialize_generators_and_graph

            def initialize() -> Any:
                generators, graph = original_initialize()
                for generator in generators:
                    name = getattr(generator.config, "name", None)
                    if name not in columns:
                        continue
                    original = generator.agenerate

                    async def generate(
                        record: dict[str, Any], *, _original: Any = original, _name: str = name
                    ) -> dict[str, Any]:
                        key = record.get(self.key_column)
                        if not isinstance(key, str) or key not in sources:
                            raise AnonymizerWorkflowError("Workflow generator has invalid source correspondence")
                        position, association = sources[key]
                        body = {name: value for name, value in record.items() if name != self.key_column}
                        token = _SOURCE_WORK.set(SourceWork(association, _name))
                        try:
                            returned = await _original(body)
                            if not isinstance(returned, dict) or self.key_column in returned:
                                raise AnonymizerWorkflowError(
                                    "Workflow generator returned invalid source correspondence"
                                )
                            return {**returned, self.key_column: key}
                        except BaseException:
                            failed.add(position)
                            raise
                        finally:
                            _SOURCE_WORK.reset(token)

                    generator.agenerate = generate
                return generators, graph

            result._initialize_generators_and_graph = initialize
            return result

        designer._create_dataset_builder = builder
        try:
            yield failed
        finally:
            designer._create_dataset_builder = original_builder
