# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source correspondence for one whole-frame DataDesigner submission."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import pandas as pd

from anonymizer.engine.constants import (
    COL_DETECTED_ENTITIES,
    COL_ENTITIES_BY_VALUE,
    COL_FINAL_ENTITIES,
    COL_LATENT_ENTITIES,
    COL_PRIVACY_QA,
    COL_PRIVACY_QA_REANSWER,
    COL_QUALITY_QA,
    COL_QUALITY_QA_COMPARE,
    COL_QUALITY_QA_REANSWER,
    COL_REPLACEMENT_MAP,
)
from anonymizer.engine.rewrite.parsers import normalize_payload
from anonymizer.interface.errors import AnonymizerWorkflowError

if TYPE_CHECKING:
    from anonymizer.engine.graph_sdk.requests import SemanticAssociation
    from anonymizer.engine.rewrite._semantic_evaluation import RewriteEvaluationBinding


@dataclass(frozen=True, slots=True)
class SourceWork:
    association: SemanticAssociation
    column: str
    generation: int = 0


_SOURCE_WORK: ContextVar[SourceWork | None] = ContextVar("anonymizer_dd_source_work", default=None)


def current_source_work(generator: Any = None) -> SourceWork | None:
    work = _SOURCE_WORK.get()
    if work is not None and (type(work) is not SourceWork or not hasattr(work, "association")):
        raise AnonymizerWorkflowError("Malformed owned workflow source context")
    if hasattr(generator, "_anonymizer_source_frame"):
        frame = generator._anonymizer_source_frame
        if type(frame) is not WorkflowSourceFrame or work is None or work.association not in frame.associations:
            raise AnonymizerWorkflowError("Managed workflow generator lost its source correspondence")
    return work


@contextmanager
def source_generation(index: int) -> Iterator[None]:
    """Keep each validation chunk distinct from physical attempts of that chunk."""
    work = _SOURCE_WORK.get()
    token = _SOURCE_WORK.set(replace(work, generation=index) if work is not None else None)
    try:
        yield
    finally:
        _SOURCE_WORK.reset(token)


_COL_SOURCE_PREFIX = "_anonymizer_source_"
_SEMANTIC_PAYLOAD_COLUMNS = frozenset(
    {
        COL_DETECTED_ENTITIES,
        COL_ENTITIES_BY_VALUE,
        COL_FINAL_ENTITIES,
        COL_LATENT_ENTITIES,
        COL_PRIVACY_QA,
        COL_PRIVACY_QA_REANSWER,
        COL_QUALITY_QA,
        COL_QUALITY_QA_COMPARE,
        COL_QUALITY_QA_REANSWER,
        COL_REPLACEMENT_MAP,
    }
)


@dataclass(frozen=True)
class WorkflowSourceFrame:
    dataframe: pd.DataFrame
    key_column: str
    keys: tuple[str, ...]
    associations: tuple[SemanticAssociation, ...] = ()
    invocation_bound: bool = False

    @classmethod
    def capture(
        cls,
        dataframe: pd.DataFrame,
        *,
        associations: tuple[SemanticAssociation, ...] = (),
        invocation_bound: bool = False,
        source_limit: int | None = None,
    ) -> WorkflowSourceFrame:
        selected = len(dataframe) if source_limit is None else source_limit
        if type(selected) is not int or not 0 <= selected <= len(dataframe):
            raise AnonymizerWorkflowError("Workflow source limit does not match the frame")
        if (associations or invocation_bound) and (
            len(associations) != selected or len(set(associations)) != len(associations)
        ):
            raise AnonymizerWorkflowError("Workflow source associations do not match the frame")
        key_column = f"{_COL_SOURCE_PREFIX}{uuid.uuid4().hex}"
        while key_column in dataframe.columns:
            key_column = f"{_COL_SOURCE_PREFIX}{uuid.uuid4().hex}"
        snapshot = dataframe.copy()
        for ordinal, dtype in enumerate(dataframe.dtypes):
            if dtype == object and dataframe.columns[ordinal] in _SEMANTIC_PAYLOAD_COLUMNS:
                for row in range(len(dataframe)):
                    snapshot.iat[row, ordinal] = deepcopy(dataframe.iat[row, ordinal])
        return cls(
            snapshot,
            key_column,
            tuple(uuid.uuid4().hex for _ in range(len(dataframe))),
            associations,
            invocation_bound,
        )

    def matches(self, dataframe: pd.DataFrame | None) -> bool:
        """Authenticate structured source values independently of container identity.

        Parquet cells can contain NumPy arrays nested inside schema wrappers;
        pandas equality alone compares their container aliases inconsistently.
        Normalize those semantic cells while preserving raw string values,
        exact frame axes/dtypes, and opaque caller metadata semantics.
        """
        if (
            not isinstance(dataframe, pd.DataFrame)
            or not self.dataframe.columns.equals(dataframe.columns)
            or not self.dataframe.index.equals(dataframe.index)
            or not self.dataframe.dtypes.equals(dataframe.dtypes)
        ):
            return False
        captured, supplied = self.dataframe.copy(), dataframe.copy()
        for ordinal, dtype in enumerate(dataframe.dtypes):
            if dtype == object and dataframe.columns[ordinal] in _SEMANTIC_PAYLOAD_COLUMNS:
                for frame in (captured, supplied):
                    frame.isetitem(
                        ordinal,
                        pd.Series(
                            [
                                value if type(value) is str else normalize_payload(value)
                                for value in frame.iloc[:, ordinal]
                            ],
                            index=frame.index,
                            dtype=object,
                        ),
                    )
        return captured.equals(supplied)

    def seed(self) -> pd.DataFrame:
        seed = self.dataframe.copy()
        for ordinal, dtype in enumerate(self.dataframe.dtypes):
            if dtype == object and seed.columns[ordinal] in _SEMANTIC_PAYLOAD_COLUMNS:
                for row in range(len(seed)):
                    seed.iat[row, ordinal] = deepcopy(self.dataframe.iat[row, ordinal])
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
    def bind_generators(
        self, designer: Any, columns: frozenset[str], rewrite_binding: RewriteEvaluationBinding | None = None
    ) -> Iterator[set[int]]:
        """Bind real DD row generators to issued associations, without exposing transport keys."""
        failed: set[int] = set()
        if not self.associations and not self.invocation_bound and not hasattr(designer, "_create_dataset_builder"):
            yield failed
            return
        storage_bindings = ExitStack()
        original_builder = designer._create_dataset_builder
        sources = {
            key: (index, association)
            for index, (key, association) in enumerate(
                zip(self.keys[: len(self.associations)], self.associations, strict=True)
            )
        }

        def builder(*args: Any, **kwargs: Any) -> Any:
            result = original_builder(*args, **kwargs)
            storage_bindings.enter_context(self._bind_storage(result.artifact_storage))
            original_initialize = result._initialize_generators_and_graph

            def write_config() -> None:
                from data_designer.config.config_builder import BuilderConfig
                from data_designer.engine.storage.artifact_storage import SDG_CONFIG_FILENAME

                config = result._data_designer_config
                public = config.model_copy(
                    update={"columns": [column for column in config.columns if column.name != self.key_column]}
                )
                storage = result.artifact_storage
                storage.mkdir_if_needed(storage.base_dataset_path)
                BuilderConfig(data_designer=public).to_json(storage.base_dataset_path / SDG_CONFIG_FILENAME)

            result._write_builder_config = write_config

            def initialize() -> Any:
                generators, graph = original_initialize()
                if not self.associations and not self.invocation_bound:
                    return generators, graph
                for generator in generators:
                    name = getattr(generator.config, "name", None)
                    if name not in columns:
                        continue
                    original = generator.agenerate
                    generator._anonymizer_source_frame = self

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
                            if rewrite_binding is not None:
                                body = await rewrite_binding.before(_name, association, body)
                            returned = await _original(body)
                            if not isinstance(returned, dict) or self.key_column in returned:
                                raise AnonymizerWorkflowError(
                                    "Workflow generator returned invalid source correspondence"
                                )
                            if rewrite_binding is not None:
                                await rewrite_binding.after(_name, association, returned)
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
            storage_bindings.close()

    @contextmanager
    def _bind_storage(self, storage: Any) -> Iterator[None]:
        """Keep only batch-order identity in memory across DD's parquet round trip."""
        from data_designer.engine.storage.artifact_storage import BatchStage

        write = storage.write_parquet_file
        load = storage.load_dataset
        write_metadata = storage.write_metadata
        batches: dict[str, tuple[str, ...]] = {}

        def write_file(
            parquet_file_name: str,
            dataframe: pd.DataFrame,
            batch_stage: Any,
            subfolder: str | None = None,
        ) -> Any:
            if batch_stage in {BatchStage.PARTIAL_RESULT, BatchStage.FINAL_RESULT}:
                if subfolder or list(dataframe.columns).count(self.key_column) != 1:
                    raise AnonymizerWorkflowError("Workflow storage has no source correspondence")
                keys = tuple(dataframe[self.key_column])
                if any(type(key) is not str or key not in self.keys for key in keys) or len(set(keys)) != len(keys):
                    raise AnonymizerWorkflowError("Workflow storage has invalid source correspondence")
                batches[parquet_file_name] = keys
            return write(
                parquet_file_name,
                dataframe.drop(columns=self.key_column, errors="ignore"),
                batch_stage,
                subfolder=subfolder,
            )

        def load_dataset(batch_stage: Any = BatchStage.FINAL_RESULT) -> pd.DataFrame:
            dataframe = load(batch_stage)
            if batch_stage != BatchStage.FINAL_RESULT:
                return dataframe
            paths = sorted(storage.final_dataset_path.glob("*.parquet"))
            if not paths or any(path.name not in batches for path in paths):
                raise AnonymizerWorkflowError("Workflow storage has unknown source batches")
            keys = tuple(key for path in paths for key in batches[path.name])
            if len(keys) != len(dataframe) or len(set(keys)) != len(keys) or self.key_column in dataframe:
                raise AnonymizerWorkflowError("Workflow storage has invalid source batches")
            dataframe[self.key_column] = keys
            return dataframe

        def public_metadata(metadata: dict[str, Any]) -> Any:
            # DD 0.9.4 persists seed-column statistics alongside the public dataset.
            if "column_statistics" in metadata:
                metadata = {
                    **metadata,
                    "column_statistics": [
                        row for row in metadata["column_statistics"] if row["column_name"] != self.key_column
                    ],
                }
            return write_metadata(metadata)

        object.__setattr__(storage, "write_parquet_file", write_file)
        object.__setattr__(storage, "write_metadata", public_metadata)
        object.__setattr__(storage, "load_dataset", load_dataset)
        try:
            yield
        finally:
            object.__setattr__(storage, "write_parquet_file", write)
            object.__setattr__(storage, "write_metadata", write_metadata)
            object.__setattr__(storage, "load_dataset", load)
