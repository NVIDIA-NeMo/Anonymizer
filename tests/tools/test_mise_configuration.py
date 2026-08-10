# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def test_github_setup_requires_signed_mise_installer() -> None:
    action = yaml.safe_load((REPO_ROOT / ".github/actions/setup-python-env/action.yml").read_text(encoding="utf-8"))
    install_step = next(step for step in action["runs"]["steps"] if "tools/install-mise.sh" in step.get("run", ""))

    assert install_step.get("env", {}).get("MISE_REQUIRE_SIGNED_INSTALL") == "1"


def test_local_mise_installer_keeps_unsigned_fallback_opt_in() -> None:
    installer = (REPO_ROOT / "tools/install-mise.sh").read_text(encoding="utf-8")

    assert 'REQUIRE_SIGNED_INSTALL="${MISE_REQUIRE_SIGNED_INSTALL:-0}"' in installer


def test_mise_typecheck_preserves_blocking_repository_contract() -> None:
    quality_tasks = _read_toml(REPO_ROOT / ".mise/tasks/quality.toml")
    typecheck = quality_tasks["typecheck"]

    assert typecheck["description"] == "Run blocking ty type checks."
    assert typecheck["run"][0] == {"task": "install-dev-docs"}
    assert typecheck["run"][1] == "tools/codestyle/typecheck.sh"


def test_mise_python_tool_versions_match_project_versions() -> None:
    mise = _read_toml(REPO_ROOT / ".mise.toml")
    project = _read_toml(REPO_ROOT / "pyproject.toml")

    for tool in ("ruff", "ty"):
        requirement = next(
            requirement for requirement in project["dependency-groups"]["dev"] if requirement.startswith(f"{tool}==")
        )
        assert requirement == f"{tool}=={mise['tools'][tool]}"

    assert project["tool"]["uv"]["required-version"] == f">={mise['tools']['uv']}"
