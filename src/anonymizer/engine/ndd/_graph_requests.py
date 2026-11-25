# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capability checks for the private DataDesigner hooks the source-frame binding uses.

Every check runs before any DataDesigner work starts and raises one standard error naming the missing hook,
the installed DataDesigner version and the supported range. The binding checks for the hooks instead of an exact
version. The supported range is the ``data-designer`` requirement in ``pyproject.toml``, read from the installed
package metadata so the message cannot drift from it.
"""

from __future__ import annotations

import importlib
import re
from importlib.metadata import PackageNotFoundError, requires, version
from typing import Any

from anonymizer.engine.errors import AnonymizerWorkflowError

_ARTIFACT_STORAGE_MODULE = "data_designer.engine.storage.artifact_storage"


def _installed_data_designer() -> str:
    try:
        return version("data-designer")
    except PackageNotFoundError:
        return "unknown"


def _supported_data_designer() -> str:
    """Return the ``data-designer`` requirement from the package metadata, or a pointer to ``pyproject.toml``."""
    try:
        requirements = requires("nemo-anonymizer") or []
    except PackageNotFoundError:
        requirements = []
    for requirement in requirements:
        if re.split(r"[\s<>=!~;\[(]", requirement, maxsplit=1)[0] == "data-designer":
            return requirement
    return "see the data-designer requirement in pyproject.toml"


def _missing_capability(attribute: str) -> AnonymizerWorkflowError:
    return AnonymizerWorkflowError(
        f"Anonymizer's DataDesigner binding requires DataDesigner's private `{attribute}`, which the installed "
        f"data-designer {_installed_data_designer()} does not provide (supported: {_supported_data_designer()})"
    )


def _module_attribute(module: str, name: str) -> Any:
    """Resolve a private DataDesigner module attribute, or raise the standard capability error."""
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as error:
        raise _missing_capability(f"{module}.{name}") from error


def require_source_frame_capabilities(designer: Any) -> None:
    """Check, before any DD work runs, the hooks the source-frame binding resolves on classes and modules."""
    if not callable(getattr(designer, "_create_dataset_builder", None)):
        raise _missing_capability("DataDesigner._create_dataset_builder")
    stage = _module_attribute(_ARTIFACT_STORAGE_MODULE, "BatchStage")
    for member in ("PARTIAL_RESULT", "FINAL_RESULT"):
        if not hasattr(stage, member):
            raise _missing_capability(f"BatchStage.{member}")
    if not isinstance(_module_attribute(_ARTIFACT_STORAGE_MODULE, "SDG_CONFIG_FILENAME"), str):
        raise _missing_capability(f"{_ARTIFACT_STORAGE_MODULE}.SDG_CONFIG_FILENAME")


def require_dataset_builder_capabilities(builder: Any) -> Any:
    """Check a freshly created dataset builder's instance hooks before it runs; return its artifact storage."""
    for name in ("_initialize_generators_and_graph", "_write_builder_config"):
        if not callable(_require_attribute(builder, name, f"DatasetBuilder.{name}")):
            raise _missing_capability(f"DatasetBuilder.{name}")
    _require_attribute(builder, "_data_designer_config", "DatasetBuilder._data_designer_config")
    storage = _require_attribute(builder, "artifact_storage", "DatasetBuilder.artifact_storage")
    for name in ("write_parquet_file", "load_dataset", "write_metadata", "mkdir_if_needed"):
        if not callable(_require_attribute(storage, name, f"ArtifactStorage.{name}")):
            raise _missing_capability(f"ArtifactStorage.{name}")
    for name in ("base_dataset_path", "final_dataset_path"):
        _require_attribute(storage, name, f"ArtifactStorage.{name}")
    return storage


def _require_attribute(owner: Any, name: str, label: str) -> Any:
    try:
        return getattr(owner, name)
    except AttributeError:
        raise _missing_capability(label) from None
