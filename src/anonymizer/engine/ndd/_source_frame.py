# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source correspondence for one whole-frame DataDesigner submission."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

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
from anonymizer.interface.errors import AnonymizerWorkflowError

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

    @classmethod
    def capture(cls, dataframe: pd.DataFrame) -> WorkflowSourceFrame:
        key_column = f"{_COL_SOURCE_PREFIX}{uuid.uuid4().hex}"
        while key_column in dataframe.columns:
            key_column = f"{_COL_SOURCE_PREFIX}{uuid.uuid4().hex}"
        snapshot = dataframe.copy()
        for ordinal, dtype in enumerate(dataframe.dtypes):
            if dtype == object and dataframe.columns[ordinal] in _SEMANTIC_PAYLOAD_COLUMNS:
                for row in range(len(dataframe)):
                    snapshot.iat[row, ordinal] = deepcopy(dataframe.iat[row, ordinal])
        return cls(snapshot, key_column, tuple(uuid.uuid4().hex for _ in range(len(dataframe))))

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
    def bind_generators(self, designer: Any) -> Iterator[None]:
        """Carry source keys through DD's artifact storage without exposing them in public output."""
        from anonymizer.engine.ndd._graph_requests import (
            require_dataset_builder_capabilities,
            require_source_frame_capabilities,
        )

        if not _is_data_designer(designer) and not hasattr(designer, "_create_dataset_builder"):
            # A caller's stand-in designer without DD's builder: nothing to bind.
            yield
            return
        require_source_frame_capabilities(designer)
        storage_bindings = ExitStack()
        original_builder = designer._create_dataset_builder

        def builder(*args: Any, **kwargs: Any) -> Any:
            result = original_builder(*args, **kwargs)
            storage_bindings.enter_context(self._bind_storage(require_dataset_builder_capabilities(result)))

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
            return result

        designer._create_dataset_builder = builder
        try:
            yield
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


def _is_data_designer(designer: Any) -> bool:
    """Whether ``designer`` is a real DataDesigner, whose source-frame hooks are then required."""
    from data_designer.interface.data_designer import DataDesigner

    return isinstance(designer, DataDesigner)
