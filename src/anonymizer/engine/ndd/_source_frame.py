# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source correspondence for one whole-frame DataDesigner submission."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Generator, Mapping
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from threading import Lock
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
    def bind_generators(self, designer: Any) -> Generator[None]:
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
        builders: list[Any] = []

        def builder(original_builder: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
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
            builders.append(result)
            return result

        try:
            with designer_hook(designer, "_create_dataset_builder", builder):
                yield
        finally:
            storage_bindings.close()
            for built in builders:
                # The override closes over the builder; drop it so the run's builder is freed by refcount.
                vars(built).pop("_write_builder_config", None)

    @contextmanager
    def _bind_storage(self, storage: Any) -> Generator[None]:
        """Keep only batch-order identity in memory across DD's parquet round trip."""
        from data_designer.engine.storage.artifact_storage import BatchStage

        write = storage.write_parquet_file
        load = storage.load_dataset
        write_metadata = storage.write_metadata
        batches: dict[str, tuple[str, ...]] = {}
        source_keys = frozenset(self.keys)

        def write_file(
            parquet_file_name: str,
            dataframe: pd.DataFrame,
            batch_stage: Any,
            subfolder: str | None = None,
            **kwargs: Any,
        ) -> Any:
            if batch_stage in {BatchStage.PARTIAL_RESULT, BatchStage.FINAL_RESULT}:
                if subfolder or list(dataframe.columns).count(self.key_column) != 1:
                    raise AnonymizerWorkflowError("Workflow storage has no source correspondence")
                keys = tuple(dataframe[self.key_column])
                if any(type(key) is not str or key not in source_keys for key in keys) or len(set(keys)) != len(keys):
                    raise AnonymizerWorkflowError("Workflow storage has invalid source correspondence")
                batches[parquet_file_name] = keys
            return write(
                parquet_file_name,
                dataframe.drop(columns=self.key_column, errors="ignore"),
                batch_stage,
                subfolder=subfolder,
                **kwargs,
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
            # DataDesigner persists seed-column statistics alongside the public dataset.
            if "column_statistics" in metadata:
                metadata = {
                    **metadata,
                    "column_statistics": [
                        row for row in metadata["column_statistics"] if row["column_name"] != self.key_column
                    ],
                }
            return write_metadata(metadata)

        originals = {"write_parquet_file": write, "write_metadata": write_metadata, "load_dataset": load}
        owned = {name: name in vars(storage) for name in originals}
        object.__setattr__(storage, "write_parquet_file", write_file)
        object.__setattr__(storage, "write_metadata", public_metadata)
        object.__setattr__(storage, "load_dataset", load_dataset)
        try:
            yield
        finally:
            for name, original in originals.items():
                _restore_attribute(storage, name, original, owned=owned[name])


# The run each context belongs to, by (designer id, method name). Only an opaque token: a context copied into a
# worker thread may outlive the run, and must not keep the run's handler (and what it closes over) alive.
_RUN_HOOKS: ContextVar[Mapping[tuple[int, str], object]] = ContextVar("anonymizer_designer_hooks", default={})
# (designer id, method name) -> (the method the dispatcher replaced, whether the designer owned it, handlers by run)
_INSTALLED: dict[tuple[int, str], tuple[Any, bool, dict[object, Callable[..., Any]]]] = {}
_INSTALLED_LOCK = Lock()


@contextmanager
def designer_hook(designer: Any, name: str, handler: Callable[..., Any]) -> Generator[None]:
    """Route ``designer.<name>(...)`` to ``handler(next, ...)`` for calls made in this context only.

    Runs sharing one designer stay concurrent: while any run is active the designer carries one dispatcher per
    method, and each call reaches the handler of the run whose context made it (``next`` is the method it wraps).
    """
    key, run = (id(designer), name), object()
    outer = _RUN_HOOKS.get()
    with _INSTALLED_LOCK:
        if key not in _INSTALLED:
            original, handlers = getattr(designer, name), {}
            _INSTALLED[key] = (original, name in vars(designer), handlers)

            def dispatch(*args: Any, **kwargs: Any) -> Any:
                return handlers.get(_RUN_HOOKS.get().get(key), original)(*args, **kwargs)

            object.__setattr__(designer, name, dispatch)
        original, _, handlers = _INSTALLED[key]
        handlers[run] = partial(handler, handlers.get(outer.get(key), original))
    token = _RUN_HOOKS.set({**outer, key: run})
    try:
        yield
    finally:
        _RUN_HOOKS.reset(token)
        with _INSTALLED_LOCK:
            original, owned, handlers = _INSTALLED[key]
            del handlers[run]
            if not handlers:
                del _INSTALLED[key]
                # Restoring the class's bound method as an instance attribute would tie the designer to itself.
                _restore_attribute(designer, name, original, owned=owned)


def _restore_attribute(target: Any, name: str, original: Any, *, owned: bool) -> None:
    """Undo an instance-level override without leaving a bound method that refers back to ``target``."""
    if owned:
        object.__setattr__(target, name, original)
    else:
        vars(target).pop(name, None)


def _is_data_designer(designer: Any) -> bool:
    """Whether ``designer`` is a real DataDesigner, whose source-frame hooks are then required."""
    from data_designer.interface.data_designer import DataDesigner

    return isinstance(designer, DataDesigner)
