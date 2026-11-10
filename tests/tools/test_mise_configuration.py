# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def test_github_setup_requires_signed_mise_installer() -> None:
    action = yaml.safe_load((REPO_ROOT / ".github/actions/setup-python-env/action.yml").read_text(encoding="utf-8"))
    install_step = next(step for step in action["runs"]["steps"] if "tools/install-mise.sh" in step.get("run", ""))

    assert install_step.get("env", {}).get("MISE_REQUIRE_SIGNED_INSTALL") == "1"


def test_benchmark_workflow_keeps_setup_on_workflow_revision() -> None:
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/benchmark-ci.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["benchmark"]["steps"]

    assert steps[0]["uses"] == "actions/checkout@v7"
    assert steps[1]["uses"] == "./.github/actions/setup-python-env"
    assert steps[1]["with"]["checkout"] == "false"
    assert steps[2]["uses"] == "actions/checkout@v7"
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

    upload_step = next(step for step in steps if step.get("uses") == "actions/upload-artifact@v7")
    assert upload_step["with"]["path"] == "benchmark-target/${{ env.BENCHMARK_OUTPUT_DIR }}/"


def test_release_workflow_keeps_setup_on_workflow_revision() -> None:
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8"))
    triggers = workflow[True]

    assert "release-ref" in triggers["workflow_dispatch"]["inputs"]

    for job_name in ("publish-wheel", "deploy-release-docs"):
        steps = workflow["jobs"][job_name]["steps"]
        setup_index = next(
            index for index, step in enumerate(steps) if step.get("uses") == "./.github/actions/setup-python-env"
        )
        root_checkout_index = next(
            index
            for index, step in enumerate(steps)
            if step.get("uses") == "actions/checkout@v7" and "ref" not in step.get("with", {})
        )
        target_checkout_index = next(
            index
            for index, step in enumerate(steps)
            if step.get("uses") == "actions/checkout@v7" and step.get("with", {}).get("path") == "release-target"
        )

        assert root_checkout_index < setup_index < target_checkout_index
        assert steps[setup_index]["with"]["checkout"] == "false"
        assert steps[target_checkout_index]["with"] == {
            "ref": "${{ inputs.release-ref }}",
            "fetch-depth": "0",
            "fetch-tags": True,
            "path": "release-target",
        }


def test_release_workflow_runs_repository_commands_against_historical_target() -> None:
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8"))

    publish_steps = workflow["jobs"]["publish-wheel"]["steps"]
    build_step = next(step for step in publish_steps if step.get("id") == "build")
    assert "mise run build:wheel release-target" in build_step["run"]
    assert "release-target/dist/*.whl" in build_step["run"]
    assert (
        next(step for step in publish_steps if step.get("uses") == "actions/upload-artifact@v7")["with"]["path"]
        == "release-target/dist/*.whl"
    )
    assert all(
        "release-target/dist/*.whl" in step["run"]
        for step in publish_steps
        if step.get("name") in {"Publish to Test PyPI (always, as pre-flight)", "Publish to PyPI"}
    )

    release_steps = workflow["jobs"]["create-gh-release"]["steps"]
    release_step = next(step for step in release_steps if step.get("name") == "Create GitHub release")
    assert release_step["env"]["RELEASE_REF"] == "${{ inputs.release-ref }}"
    assert '--target "$RELEASE_REF"' in release_step["run"]

    docs_steps = workflow["jobs"]["deploy-release-docs"]["steps"]
    assert (
        "git -C release-target config"
        in next(step for step in docs_steps if step.get("name") == "Configure git for mike")["run"]
    )
    assert (
        next(step for step in docs_steps if step.get("name") == "Build docs")["run"]
        == "mise run docs:build release-target"
    )
    assert (
        next(step for step in docs_steps if step.get("name") == "Deploy release docs with mike")["run"]
        == 'mise run docs:deploy "$VERSION" release-target'
    )

    build_task = (REPO_ROOT / ".mise/tasks/build/wheel").read_text(encoding="utf-8")
    docs_tasks = _read_toml(REPO_ROOT / ".mise/tasks/docs.toml")
    assert 'arg "[project]"' in build_task
    assert 'uv build --wheel "$project"' in build_task
    assert 'arg "[project]"' in docs_tasks["docs:build"]["usage"]
    assert "uv run --directory" in docs_tasks["docs:build"]["run"]
    assert set(docs_tasks["docs:deploy"]) == {"description", "usage", "run"}


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
    assert 'echo "UV_PYTHON=${UV_PYTHON}" >> "$GITHUB_ENV"' in action
    assert 'mise run setup "${{ inputs.dependency-profile }}" --no-hooks' in action


def test_ci_matrix_jobs_assert_requested_interpreter_before_tests() -> None:
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["test"]["steps"]
    names = [step.get("name") for step in steps]
    assert_step = steps[names.index("Assert matrix interpreter")]

    assert names.index("Assert matrix interpreter") < names.index("Run tests with coverage")
    assert assert_step["env"]["EXPECTED_PYTHON"] == "${{ matrix.python-version }}"
    assert "uv run --locked --group dev python" in assert_step["run"]


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


def _gpg_supported() -> bool:
    """Return True when gpg and gpgconf exist and gpg is 2.2 or newer."""
    if shutil.which("gpg") is None or shutil.which("gpgconf") is None:
        return False
    first_line = subprocess.run(["gpg", "--version"], capture_output=True, text=True, check=False).stdout.splitlines()
    match = re.search(r"(\d+)\.(\d+)", first_line[0]) if first_line else None
    return match is not None and (int(match[1]), int(match[2])) >= (2, 2)


requires_gpg = pytest.mark.skipif(not _gpg_supported(), reason="gpg >= 2.2 with gpgconf is required")


@pytest.fixture
def short_dir() -> Iterator[Path]:
    """Short scratch directory: gpg-agent socket paths must fit the 104-byte unix limit on macOS."""
    path = Path(tempfile.mkdtemp(prefix="mi-", dir="/tmp"))
    try:
        yield path
    finally:
        for home in path.glob("gnupg*"):
            subprocess.run(["gpgconf", "--homedir", str(home), "--kill", "all"], capture_output=True, check=False)
        shutil.rmtree(path, ignore_errors=True)


def _run_installer_with_signed_script(tmp_path: Path, signer: str) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run install-mise.sh against a keyring served by a stub curl.

    The served key bundle always holds the pinned key and a second key. The
    installer payload is signed by ``signer`` (``pinned`` or ``other``) and
    touches a marker file when executed.
    """
    gnupg = tmp_path / "gnupg"
    gnupg.mkdir(mode=0o700)
    env = {"GNUPGHOME": str(gnupg), "PATH": os.environ["PATH"]}

    def gpg(*args: str, stdin: bytes | None = None) -> bytes:
        result = subprocess.run(
            ["gpg", "--batch", "--no-tty", "--pinentry-mode", "loopback", "--passphrase", "", *args],
            input=stdin,
            capture_output=True,
            env=env,
            check=True,
        )
        return result.stdout

    fingerprints: dict[str, str] = {}
    for name in ("pinned", "other"):
        try:
            gpg("--quick-generate-key", f"{name} <{name}@example.invalid>", "ed25519", "sign", "1d")
        except subprocess.CalledProcessError as error:
            pytest.skip(f"gpg cannot generate keys here: {error.stderr.decode(errors='replace')[:200]}")
        listing = gpg("--with-colons", "--list-keys", f"{name}@example.invalid").decode()
        fingerprints[name] = next(line.split(":")[9] for line in listing.splitlines() if line.startswith("fpr:"))

    bundle = tmp_path / "bundle.asc"
    bundle.write_bytes(gpg("--armor", "--export", fingerprints["pinned"], fingerprints["other"]))

    marker = tmp_path / "payload-ran"
    payload = f"touch '{marker}'\n".encode()
    signature = tmp_path / "install.sh.sig"
    signature.write_bytes(gpg("--local-user", fingerprints[signer], "--sign", "--output", "-", stdin=payload))

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "curl").write_text(
        "#!/bin/sh\n"
        'while [ "$#" -gt 0 ]; do\n'
        '  case "$1" in -o) out="$2"; shift 2;; *) url="$1"; shift;; esac\n'
        "done\n"
        'case "$url" in\n'
        f'  */by-fingerprint/*) cp "{bundle}" "$out";;\n'
        f'  */install.sh.sig) cp "{signature}" "$out";;\n'
        "  *) exit 22;;\n"
        "esac\n",
        encoding="utf-8",
    )
    (fake_bin / "curl").chmod(0o755)
    for tool in (
        "bash",
        "sh",
        "gpg",
        "gpgconf",
        "awk",
        "grep",
        "sed",
        "head",
        "mktemp",
        "chmod",
        "rm",
        "touch",
        "cp",
        "dirname",
    ):
        located = shutil.which(tool)
        assert located is not None, tool
        (fake_bin / tool).symlink_to(located)

    result = subprocess.run(
        [str(fake_bin / "bash"), str(REPO_ROOT / "tools/install-mise.sh")],
        capture_output=True,
        text=True,
        env={
            "PATH": str(fake_bin),
            "HOME": str(tmp_path / "home"),
            "TMPDIR": str(tmp_path),
            "MISE_GPG_KEY": fingerprints["pinned"],
            "MISE_REQUIRE_SIGNED_INSTALL": "1",
        },
        check=False,
    )
    return result, marker


@requires_gpg
def test_installer_runs_payload_signed_by_pinned_key(short_dir: Path) -> None:
    result, marker = _run_installer_with_signed_script(short_dir, signer="pinned")

    assert marker.exists(), result.stderr


@requires_gpg
def test_installer_rejects_payload_signed_by_extra_key_in_bundle(short_dir: Path) -> None:
    result, marker = _run_installer_with_signed_script(short_dir, signer="other")

    assert not marker.exists()
    assert result.returncode != 0
    assert "MISE_REQUIRE_SIGNED_INSTALL=1 forbids unsigned fallback" in result.stderr
