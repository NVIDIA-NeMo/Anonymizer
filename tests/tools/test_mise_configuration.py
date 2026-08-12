# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
import subprocess
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


def test_benchmark_workflow_checks_out_before_loading_local_action() -> None:
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/benchmark-ci.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["benchmark"]["steps"]

    assert steps[0]["uses"] == "actions/checkout@v6"
    assert steps[0]["with"] == {"ref": "${{ env.BENCHMARK_REF }}", "fetch-depth": "0"}
    assert steps[1]["uses"] == "./.github/actions/setup-python-env"
    assert steps[1]["with"]["checkout"] == "false"


def test_local_mise_installer_keeps_unsigned_fallback_opt_in() -> None:
    installer = (REPO_ROOT / "tools/install-mise.sh").read_text(encoding="utf-8")

    assert 'REQUIRE_SIGNED_INSTALL="${MISE_REQUIRE_SIGNED_INSTALL:-0}"' in installer


def test_makefile_exposes_bootstrap_without_deprecated_task_aliases() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    phony_targets = [line.removeprefix(".PHONY: ") for line in makefile.splitlines() if line.startswith(".PHONY: ")]

    assert phony_targets == ["help", "install-mise", "setup"]
    assert "deprecated_target" not in makefile


def test_mise_typecheck_preserves_blocking_repository_contract() -> None:
    quality_tasks = _read_toml(REPO_ROOT / ".mise/tasks/quality.toml")
    typecheck = quality_tasks["check:type"]

    assert typecheck["description"].startswith("Run blocking ty checks")
    assert typecheck["run"] == "uv run --locked --group docs tools/codestyle/typecheck.sh"


def test_mise_sources_the_uv_managed_environment() -> None:
    mise = _read_toml(REPO_ROOT / ".mise.toml")

    assert "VIRTUAL_ENV" not in mise["env"]
    assert mise["settings"]["python"]["uv_venv_auto"] == "source"


def test_mise_uses_native_task_composition() -> None:
    quality_tasks = _read_toml(REPO_ROOT / ".mise/tasks/quality.toml")
    publish_tasks = _read_toml(REPO_ROOT / ".mise/tasks/publish.toml")
    setup_tasks = _read_toml(REPO_ROOT / ".mise/tasks/setup.toml")
    clean_task = (REPO_ROOT / ".mise/tasks/clean/_default").read_text(encoding="utf-8")
    notebook_task = (REPO_ROOT / ".mise/tasks/notebooks/execute").read_text(encoding="utf-8")

    assert set(quality_tasks["check"]["depends"]) == {
        "check:format",
        "check:license:headers",
        "check:lint",
        "check:lock",
        "check:type",
    }
    assert {"task": "build:wheel"} in publish_tasks["publish:pypi"]["run"]
    assert setup_tasks["setup"]["run"][2] == {"task": "deps:sync", "args": ["{{usage.profile}}"]}
    assert '#MISE depends=["clean:pycache"]' in clean_task
    assert "mise run clean:pycache" not in clean_task
    assert "mise run deps:sync notebooks" not in notebook_task


def test_mise_dependency_profiles_require_current_lockfile() -> None:
    setup_tasks = _read_toml(REPO_ROOT / ".mise/tasks/setup.toml")
    sync_task = setup_tasks["deps:sync"]

    assert sync_task["run"].count("uv sync --locked") == 4
    assert 'choices "runtime" "dev" "docs" "notebooks"' in sync_task["usage"]
    assert "runtime) uv sync --locked --no-default-groups" in sync_task["run"]
    for profile in ("runtime", "dev", "docs", "notebooks"):
        assert f"{profile}) uv sync --locked" in sync_task["run"]


def test_mise_task_tree_uses_colon_delimited_vocabulary() -> None:
    completed = subprocess.run(
        ["mise", "tasks", "--json"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    tasks = json.loads(completed.stdout)
    task_names = {task["name"] for task in tasks}

    assert all(re.fullmatch(r"[a-z0-9]+(?::[a-z0-9]+)*", name) for name in task_names)
    assert all(not task["aliases"] for task in tasks)
    assert {
        "build:wheel",
        "check:format",
        "check:lint",
        "check:lock",
        "check:type",
        "deps:sync",
        "hooks:install",
        "lock:update",
        "notebooks:execute",
        "test:coverage",
        "validate",
    } <= task_names


def test_github_setup_preserves_requested_python_and_dependency_profile() -> None:
    action = (REPO_ROOT / ".github/actions/setup-python-env/action.yml").read_text(encoding="utf-8")

    assert 'export UV_PYTHON="${{ inputs.python-version }}"' in action
    assert 'mise run setup "${{ inputs.dependency-profile }}" --no-hooks' in action


def test_precommit_lock_check_is_read_only() -> None:
    config = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    local_repo = next(repo for repo in config["repos"] if repo["repo"] == "local")
    check_hook = next(hook for hook in local_repo["hooks"] if hook["id"] == "check")

    assert check_hook["entry"] == "mise run check"
    assert all(hook["id"] != "uv-lock" for hook in local_repo["hooks"])


def test_toml_tasks_use_portable_shell_conditionals() -> None:
    task_text = "\n".join(path.read_text(encoding="utf-8") for path in (REPO_ROOT / ".mise/tasks").glob("*.toml"))

    assert "[[" not in task_text


def test_clean_branches_does_not_switch_the_callers_worktree() -> None:
    task = (REPO_ROOT / ".mise/tasks/clean/branches").read_text(encoding="utf-8")

    assert "git checkout" not in task
    assert "--merged origin/main" in task
    assert "git branch -d --" in task


def test_ruff_enables_audited_error_and_safe_fix_rules() -> None:
    ruff = _read_toml(REPO_ROOT / "ruff.toml")

    assert {"E9", "RUF100", "UP015", "UP017", "UP035", "UP037"} <= set(ruff["lint"]["select"])


def test_ruff_and_ty_check_rendered_notebooks() -> None:
    ruff = _read_toml(REPO_ROOT / "ruff.toml")
    project = _read_toml(REPO_ROOT / "pyproject.toml")
    file_collector = (REPO_ROOT / "tools/codestyle/_lib.sh").read_text(encoding="utf-8")

    assert "docs/notebooks/*.ipynb" not in ruff["exclude"]
    assert "git ls-files '*.py' '*.ipynb'" in file_collector
    assert "docs" in project["tool"]["ty"]["src"]["include"]


def test_mise_python_tool_versions_match_project_versions() -> None:
    mise = _read_toml(REPO_ROOT / ".mise.toml")
    project = _read_toml(REPO_ROOT / "pyproject.toml")

    for tool in ("ruff", "ty"):
        requirement = next(
            requirement for requirement in project["dependency-groups"]["dev"] if requirement.startswith(f"{tool}==")
        )
        assert requirement == f"{tool}=={mise['tools'][tool]}"

    assert project["tool"]["uv"]["required-version"] == f">={mise['tools']['uv']}"
