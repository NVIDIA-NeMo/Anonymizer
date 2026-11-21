# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

from anonymizer.engine.ndd import _graph_requests
from anonymizer.interface.errors import AnonymizerWorkflowError

_PYPROJECT = Path(__file__).resolve().parents[3] / "pyproject.toml"


def _pyproject_data_designer_specifiers() -> set[str]:
    dependencies = tomllib.loads(_PYPROJECT.read_text())["project"]["dependencies"]
    requirement = next(dep for dep in dependencies if dep.startswith("data-designer"))
    return set(requirement.removeprefix("data-designer").replace(" ", "").split(","))


def _capability_error_message() -> str:
    with pytest.raises(AnonymizerWorkflowError) as error:
        _graph_requests.require_source_frame_capabilities(object())
    return str(error.value)


def test_capability_error_names_the_pyproject_data_designer_range() -> None:
    message = _capability_error_message()

    supported = message.rsplit("(supported: data-designer", 1)[1].removesuffix(")")
    assert set(supported.split(",")) == _pyproject_data_designer_specifiers()


def test_capability_error_points_to_pyproject_without_package_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(distribution: str) -> list[str]:
        raise PackageNotFoundError(distribution)

    monkeypatch.setattr(_graph_requests, "requires", missing)

    message = _capability_error_message()

    assert "DataDesigner._create_dataset_builder" in message
    assert message.endswith("(supported: see the data-designer requirement in pyproject.toml)")
