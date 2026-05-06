# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from anonymizer.config.anonymizer_config import AnonymizerInput, infer_input_source_suffix
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_FINAL_ENTITIES,
    COL_LEAKAGE_MASS,
    COL_NEEDS_HUMAN_REVIEW,
    COL_TEXT,
    COL_UTILITY_SCORE,
    COL_WEIGHTED_LEAKAGE_RATE,
)
from anonymizer.engine.io.constants import SUPPORTED_IO_FORMATS
from anonymizer.engine.pipeline_context import PipelineContext
from anonymizer.interface.errors import AnonymizerIOError, InvalidInputError

logger = logging.getLogger("anonymizer")


def read_input(input_data: AnonymizerInput, *, nrows: int | None = None) -> PipelineContext:
    """Load input into a :class:`PipelineContext` with the canonical internal text column.

    Args:
        input_data: Input source definition.
        nrows: Maximum rows to read.  ``None`` reads the entire file.
    """
    dataframe = _load_dataframe(input_data, nrows=nrows)
    selected_text_column = input_data.text_column
    if selected_text_column not in dataframe.columns:
        raise InvalidInputError(f"Input text column '{selected_text_column}' not found.")
    _validate_internal_column_collision(dataframe, selected_text_column=selected_text_column)
    resolved = _resolve_output_column_collisions(dataframe, selected_text_column=selected_text_column)
    renamed = resolved.dataframe.rename(columns={resolved.original_text_column: COL_TEXT})
    return resolved.with_dataframe(renamed)


# Suffixes appended to the user's text column to form per-mode output columns
# (see ``_rename_output_columns`` in ``anonymizer.interface.anonymizer``).
_OUTPUT_COLUMN_SUFFIXES: tuple[str, ...] = ("_replaced", "_with_spans", "_rewritten")

# Fixed user-facing output column names that don't depend on the text column
# (see ``_build_user_dataframe`` in ``anonymizer.interface.anonymizer``).
_STATIC_OUTPUT_COLUMNS: tuple[str, ...] = (
    COL_FINAL_ENTITIES,
    COL_UTILITY_SCORE,
    COL_LEAKAGE_MASS,
    COL_WEIGHTED_LEAKAGE_RATE,
    COL_ANY_HIGH_LEAKED,
    COL_NEEDS_HUMAN_REVIEW,
)

# Suffix appended to input columns whose names collide with anonymizer output
# columns; mirrors how pandas disambiguates duplicate column names on read.
_INPUT_RENAME_SUFFIX = "__input"


def _validate_internal_column_collision(dataframe: pd.DataFrame, *, selected_text_column: str) -> None:
    """Hard-error if the user's input contains the reserved internal text column."""
    if COL_TEXT in dataframe.columns and selected_text_column != COL_TEXT:
        raise InvalidInputError(
            f"Input contains reserved internal column {COL_TEXT!r} while text_column={selected_text_column!r}. "
            f"Either set text_column={COL_TEXT!r} or remove/rename {COL_TEXT!r} from input."
        )


def _resolve_output_column_collisions(dataframe: pd.DataFrame, *, selected_text_column: str) -> PipelineContext:
    """Rename input columns whose names collide with Anonymizer output columns.

    The pipeline writes a known set of output columns derived from the text
    column (e.g. ``{text_col}_replaced``) plus a few fixed names such as
    ``final_entities``. If the input already contains any of those names the
    output would silently overwrite the user's data, so we rename the input
    column in place by appending ``__input`` (with a numeric suffix if needed
    to avoid a secondary collision) and emit a warning. This matches how
    pandas disambiguates duplicate column names on read and keeps the
    pipeline runnable without forcing the user to edit their input file.

    If the selected text column itself collides with a fixed output name (e.g.
    ``text_column='final_entities'``) it is renamed the same way; the returned
    context's ``original_text_column`` reflects the non-colliding identifier.
    """
    candidate_output_columns: list[str] = [f"{selected_text_column}{suffix}" for suffix in _OUTPUT_COLUMN_SUFFIXES]
    candidate_output_columns.extend(_STATIC_OUTPUT_COLUMNS)

    existing_columns: set[str] = set(dataframe.columns)
    collisions: list[str] = [name for name in candidate_output_columns if name in existing_columns]
    if not collisions:
        return PipelineContext(dataframe=dataframe, original_text_column=selected_text_column)

    rename_map: dict[str, str] = {}
    for original_name in collisions:
        new_name = _next_available_name(f"{original_name}{_INPUT_RENAME_SUFFIX}", existing_columns)
        rename_map[original_name] = new_name
        existing_columns.add(new_name)

    formatted = ", ".join(f"{old!r} -> {new!r}" for old, new in rename_map.items())
    logger.warning(
        "Renamed input column(s) that collide with Anonymizer output column names: %s. "
        "Update your input schema to remove this warning.",
        formatted,
    )
    return PipelineContext(
        dataframe=dataframe.rename(columns=rename_map),
        original_text_column=rename_map.get(selected_text_column, selected_text_column),
    )


def _next_available_name(candidate: str, existing: set[str]) -> str:
    """Return *candidate*, or ``candidate_2``/``_3``/... if it's already taken."""
    if candidate not in existing:
        return candidate
    counter = 2
    while f"{candidate}_{counter}" in existing:
        counter += 1
    return f"{candidate}_{counter}"


def _read_parquet_partial(source: str, *, nrows: int | None = None) -> pd.DataFrame:
    """Read a Parquet file, stopping early when *nrows* is set.

    Returns a schema-preserving empty DataFrame when *nrows* is zero or negative.
    """
    if nrows is None:
        return pd.read_parquet(source)
    if nrows <= 0:
        schema = pq.ParquetFile(source).schema_arrow
        empty = pa.table(
            {name: pa.array([], type=field.type) for name, field in zip(schema.names, schema)},
        )
        return empty.to_pandas()
    pf = pq.ParquetFile(source)
    batches: list[pa.RecordBatch] = []
    rows_so_far = 0
    for batch in pf.iter_batches(batch_size=nrows):
        batches.append(batch)
        rows_so_far += len(batch)
        if rows_so_far >= nrows:
            break
    table = pa.Table.from_batches(batches, schema=pf.schema_arrow)
    return table.slice(0, nrows).to_pandas()


def _load_dataframe(input_data: AnonymizerInput, *, nrows: int | None = None) -> pd.DataFrame:
    source_str = str(input_data.source)
    suffix = infer_input_source_suffix(source_str)
    if suffix not in SUPPORTED_IO_FORMATS:
        supported_formats = " or ".join(SUPPORTED_IO_FORMATS)
        raise InvalidInputError(f"Unsupported input format: {suffix}. Use {supported_formats}.")
    if nrows is not None and nrows <= 0:
        logger.debug("nrows=%d; returning empty DataFrame for %s", nrows, source_str)
    try:
        if suffix == ".csv":
            df = pd.read_csv(source_str, nrows=nrows)
        else:
            df = _read_parquet_partial(source_str, nrows=nrows)
    except (OSError, pd.errors.ParserError, ValueError) as error:
        raise AnonymizerIOError(f"Failed to read input data from: {source_str}") from error
    if nrows is not None:
        logger.info(
            "👀 Preview mode: 📂 Loaded %d records from %s (column: '%s')",
            len(df),
            source_str,
            input_data.text_column,
        )
    else:
        logger.info("📂 Loaded %d records from %s (column: '%s')", len(df), source_str, input_data.text_column)
    return df
