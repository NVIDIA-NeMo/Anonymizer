# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from codestyle import check_docs_pairs as pairs

SCRIPT = Path(__file__).resolve().parents[2] / "tools/codestyle/check_docs_pairs.py"
PAIR = {"docs/concepts/replace.md", "docs/concepts/replace.mdx"}


@pytest.mark.parametrize(
    ("changed", "expected"),
    [
        ({"docs/concepts/replace.md", "docs/concepts/replace.mdx"}, []),
        ({"docs/concepts/replace.md"}, ["docs/concepts/replace.md changed but docs/concepts/replace.mdx did not"]),
        ({"docs/concepts/replace.mdx"}, ["docs/concepts/replace.mdx changed but docs/concepts/replace.md did not"]),
    ],
)
def test_paired_page_must_change_on_both_sides(changed: set[str], expected: list[str]) -> None:
    assert pairs.find_violations(base_files=PAIR, changed=changed) == expected


def test_new_page_needs_both_formats() -> None:
    new = {"docs/concepts/new.md"}

    assert pairs.find_violations(base_files=PAIR, changed=new) == [
        "docs/concepts/new.md changed but docs/concepts/new.mdx did not"
    ]


def test_adding_missing_format_to_existing_page_is_allowed() -> None:
    base = {"docs/concepts/solo.md"}

    assert pairs.find_violations(base_files=base, changed={"docs/concepts/solo.mdx"}) == []


def test_single_format_page_at_base_is_not_checked() -> None:
    base = {"docs/data/README.md"}

    assert pairs.find_violations(base_files=base, changed=base) == []


def test_fern_infrastructure_and_non_docs_files_are_ignored() -> None:
    changed = {"docs/fern/FERN_MIGRATION.md", "README.md", "docs/fern/docs.yml"}

    assert pairs.find_violations(base_files=PAIR, changed=changed) == []


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout


def _run_check(repo: Path) -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith(("GIT_", "GITHUB_"))}
    return subprocess.run([sys.executable, str(SCRIPT)], cwd=repo, env=env, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "--quiet", "--initial-branch=main")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / "docs/concepts").mkdir(parents=True)
    for name in PAIR:
        (tmp_path / name).write_text("original\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "--quiet", "-m", "docs: initial pages")
    _git(tmp_path, "switch", "--quiet", "-c", "feature")
    return tmp_path


def test_uncommitted_one_sided_change_fails(repo: Path) -> None:
    (repo / "docs/concepts/replace.md").write_text("edited\n", encoding="utf-8")

    result = _run_check(repo)

    assert result.returncode == 1
    assert "docs/concepts/replace.md changed but docs/concepts/replace.mdx did not" in result.stderr


def test_committed_change_to_both_sides_passes(repo: Path) -> None:
    for name in PAIR:
        (repo / name).write_text("edited\n", encoding="utf-8")
    _git(repo, "commit", "--quiet", "-am", "docs: edit both")

    result = _run_check(repo)

    assert result.returncode == 0, result.stderr


def test_missing_merge_base_reports_how_to_fix(repo: Path) -> None:
    _git(repo, "switch", "--quiet", "--orphan", "unrelated")
    _git(repo, "commit", "--quiet", "--allow-empty", "-m", "chore: unrelated history")

    result = _run_check(repo)

    assert result.returncode == 1
    assert "no merge base between main and HEAD" in result.stderr
    assert "Traceback" not in result.stderr
