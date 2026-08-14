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


def test_release_workflow_uses_triggering_ref_and_completes_tag_releases() -> None:
    workflow_path = REPO_ROOT / ".github/workflows/release.yml"
    workflow_text = workflow_path.read_text(encoding="utf-8")
    workflow = yaml.safe_load(workflow_text)
    triggers = workflow.get("on", workflow[True])
    dispatch_inputs = triggers["workflow_dispatch"]["inputs"]

    assert triggers["push"]["tags"] == ["v*"]
    assert set(dispatch_inputs) == {"dry-run", "create-gh-release"}
    assert "release-ref" not in workflow_text
    assert "release-target" not in workflow_text

    publish_steps = workflow["jobs"]["publish-wheel"]["steps"]
    checkout_steps = [
        step for job in workflow["jobs"].values() for step in job["steps"] if step.get("uses") == "actions/checkout@v6"
    ]
    assert checkout_steps
    assert all("ref" not in step.get("with", {}) for step in checkout_steps)
    assert all("path" not in step.get("with", {}) for step in checkout_steps)

    build_step = next(step for step in publish_steps if step.get("id") == "build")
    assert "mise run build:wheel" in build_step["run"]
    assert "dist/*.whl" in build_step["run"]
    assert 'if [ "$GITHUB_REF_TYPE" = "tag" ] && [ "$GITHUB_REF_NAME" != "v${VERSION}" ]; then' in build_step["run"]
    assert "exit 1" in build_step["run"]

    release_steps = workflow["jobs"]["create-gh-release"]["steps"]
    release_step = next(step for step in release_steps if step.get("name") == "Create GitHub release")
    assert release_step["env"]["TARGET_SHA"] == "${{ github.sha }}"
    assert '--target "$TARGET_SHA"' in release_step["run"]

    docs_steps = workflow["jobs"]["deploy-release-docs"]["steps"]
    assert next(step for step in docs_steps if step.get("name") == "Build docs")["run"] == "mise run docs:build"
    assert (
        next(step for step in docs_steps if step.get("name") == "Deploy release docs with mike")["run"]
        == 'mise run docs:deploy "$VERSION"'
    )

    publish_to_pypi = next(step for step in publish_steps if step.get("name") == "Publish to PyPI")
    assert publish_to_pypi["if"] == "${{ github.event_name == 'push' || !inputs.dry-run }}"
    assert (
        workflow["jobs"]["create-gh-release"]["if"] == "${{ github.event_name == 'push' || inputs.create-gh-release }}"
    )
    assert (
        workflow["jobs"]["deploy-release-docs"]["if"]
        == "${{ github.event_name == 'push' || inputs.create-gh-release }}"
    )


def test_release_tasks_use_current_checkout() -> None:
    build_task = (REPO_ROOT / ".mise/tasks/build/wheel").read_text(encoding="utf-8")
    docs_tasks = _read_toml(REPO_ROOT / ".mise/tasks/docs.toml")

    assert "usage_project" not in build_task
    assert set(docs_tasks["docs:build"]) == {"description", "run"}
    assert 'arg "[project]"' not in docs_tasks["docs:deploy"]["usage"]
    assert "--directory" not in docs_tasks["docs:deploy"]["run"]


def test_benchmark_workflow_keeps_setup_on_workflow_revision() -> None:
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/benchmark-ci.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["benchmark"]["steps"]

    assert steps[0]["uses"] == "actions/checkout@v6"
    assert steps[1]["uses"] == "./.github/actions/setup-python-env"
    assert steps[1]["with"]["checkout"] == "false"
    assert steps[2]["uses"] == "actions/checkout@v6"
    assert steps[2]["with"] == {
        "ref": "${{ env.BENCHMARK_REF }}",
        "fetch-depth": "0",
        "path": "benchmark-target",
    }

    target_steps = [
        step
        for step in steps
        if step.get("id") == "target" or step.get("name") in {"Run benchmark suite", "Add benchmark summary"}
    ]
    assert target_steps
    assert all(step["working-directory"] == "benchmark-target" for step in target_steps)

    run_step = next(step for step in steps if step.get("name") == "Run benchmark suite")
    assert "uv run --locked --group dev python tools/measurement/run_benchmarks.py" in run_step["run"]

    upload_step = next(step for step in steps if step.get("uses") == "actions/upload-artifact@v4")
    assert upload_step["with"]["path"] == "benchmark-target/${{ env.BENCHMARK_OUTPUT_DIR }}/"


def test_local_mise_installer_keeps_unsigned_fallback_opt_in() -> None:
    installer = (REPO_ROOT / "tools/install-mise.sh").read_text(encoding="utf-8")

    assert 'REQUIRE_SIGNED_INSTALL="${MISE_REQUIRE_SIGNED_INSTALL:-0}"' in installer


def test_local_mise_installer_fetches_and_pins_release_key_over_https() -> None:
    installer = (REPO_ROOT / "tools/install-mise.sh").read_text(encoding="utf-8")

    assert 'MISE_GPG_KEY_URL="https://keys.openpgp.org/vks/v1/by-fingerprint"' in installer
    assert '"${MISE_GPG_KEY_URL}/${MISE_GPG_KEY}"' in installer
    assert 'grep -q "^fpr:::::::::${MISE_GPG_KEY}:"' in installer
    assert '--recv-keys "$MISE_GPG_KEY"' not in installer


def test_local_mise_installer_fetches_signature_for_pinned_version() -> None:
    installer = (REPO_ROOT / "tools/install-mise.sh").read_text(encoding="utf-8")

    assert 'MISE_SIG_URL="https://github.com/jdx/mise/releases/download/${MISE_VERSION}/install.sh.sig"' in installer
    assert "https://mise.jdx.dev/install.sh.sig" not in installer


def test_local_mise_installer_downloads_unsigned_fallback_before_execution() -> None:
    installer = (REPO_ROOT / "tools/install-mise.sh").read_text(encoding="utf-8")

    assert 'curl_fetch -o "$unsigned_script" "$MISE_RUN_URL"' in installer
    assert 'MISE_VERSION="$MISE_VERSION" sh "$unsigned_script"' in installer
    assert 'curl_fetch "$MISE_RUN_URL" |' not in installer


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

    assert sync_task["run"].count("uv sync --locked") == 5
    assert 'choices "runtime" "dev" "docs" "notebooks" "all"' in sync_task["usage"]
    assert "runtime) uv sync --locked --no-default-groups" in sync_task["run"]
    for profile in ("runtime", "dev", "docs", "notebooks", "all"):
        assert f"{profile}) uv sync --locked" in sync_task["run"]
    assert "all) uv sync --locked --all-groups" in sync_task["run"]


def test_mise_test_all_composes_unit_and_end_to_end_suites() -> None:
    test_tasks = _read_toml(REPO_ROOT / ".mise/tasks/tests.toml")

    assert test_tasks["test:all"]["run"] == [{"task": "test"}, {"task": "test:e2e"}]


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
        "test:all",
        "test:coverage",
    } <= task_names
    assert "validate" not in task_names


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
