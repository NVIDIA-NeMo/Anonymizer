# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).parents[1] / ".github/workflows/pr-linked-issue.yml"


def _step(name: str) -> dict:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    return next(step for step in workflow["jobs"]["check"]["steps"] if step["name"] == name)


def _run_step(
    tmp_path: Path, name: str, mock_gh: str, **variables: str
) -> tuple[subprocess.CompletedProcess[str], str]:
    gh = tmp_path / "gh"
    gh.write_text("#!/bin/sh\n" + mock_gh)
    gh.chmod(0o755)
    output = tmp_path / "github-output"
    output.write_text("")
    script = _step(name)["run"].replace("${{ github.repository }}", "NVIDIA-NeMo/Anonymizer")
    env = (
        os.environ
        | {
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "GITHUB_OUTPUT": str(output),
            "REPO": "NVIDIA-NeMo/Anonymizer",
            "PR_NUMBER": "42",
        }
        | variables
    )
    result = subprocess.run(["bash", "-e", "-c", script], env=env, capture_output=True, text=True)
    return result, output.read_text()


@pytest.mark.parametrize(
    ("response", "expected_exit", "expected_output"),
    [
        ('{"state":"open","labels":[{"name":"triaged"}]}', 0, "issue_open=true\nis_triaged=true"),
        ('{"state":"closed","labels":[{"name":"triaged"}]}', 0, "issue_open=false\nis_triaged=true"),
        ('{"state":"open","labels":[]}', 0, "issue_open=true\nis_triaged=false"),
        ('{"pull_request":{},"state":"open","labels":[{"name":"triaged"}]}', 0, "issue_exists=false"),
        ("HTTP 404", 0, "issue_exists=false"),
        ("HTTP 503", 1, ""),
    ],
)
def test_issue_validation_distinguishes_policy_failure_from_api_outage(
    tmp_path: Path, response: str, expected_exit: int, expected_output: str
) -> None:
    mock = (
        'if [ "$SCENARIO" = "HTTP 404" ] || [ "$SCENARIO" = "HTTP 503" ]; then\n'
        '  echo "gh: $SCENARIO" >&2; exit 1\n'
        "fi\n"
        'printf "%s\\n" "$SCENARIO"\n'
    )
    result, output = _run_step(tmp_path, "Validate issue is open and triaged", mock, SCENARIO=response, ISSUE_NUM="123")
    assert result.returncode == expected_exit, result.stderr
    assert expected_output in output


@pytest.mark.parametrize(
    ("response", "expected_exit", "expected_output"),
    [("write", 0, "is_collaborator=true"), ("HTTP 404", 0, "is_collaborator=false"), ("HTTP 503", 1, "")],
)
def test_author_permission_does_not_treat_api_outage_as_external(
    tmp_path: Path, response: str, expected_exit: int, expected_output: str
) -> None:
    mock = 'if [ "$SCENARIO" = "write" ]; then echo write; exit 0; fi\necho "gh: $SCENARIO" >&2; exit 1\n'
    result, output = _run_step(tmp_path, "Check author permissions", mock, SCENARIO=response, PR_AUTHOR="contributor")
    assert result.returncode == expected_exit, result.stderr
    assert expected_output in output


@pytest.mark.parametrize(
    ("body", "expected_issue"),
    [
        ("Fixes #123", "123"),
        ("<!-- Fixes #123 -->", ""),
        ("<!-- Fixes #123 -->\nResolves #456", "456"),
    ],
)
def test_issue_parser_reads_current_pr_body(tmp_path: Path, body: str, expected_issue: str) -> None:
    mock = 'printf "%s\\n" "$CURRENT_BODY"\n'
    result, output = _run_step(
        tmp_path, "Parse issue reference from PR body", mock, CURRENT_BODY=body, PR_BODY="Fixes #999"
    )
    assert result.returncode == 0, result.stderr
    assert f"issue_num={expected_issue}\n" in output


@pytest.mark.parametrize(
    ("state", "labels", "body", "issue_response", "expected_exit", "should_close"),
    [
        ("OPEN", [], "", "", 0, True),
        ("OPEN", [], "Fixes #123", '{"state":"open","labels":[]}', 0, True),
        ("OPEN", [], "Fixes #123", '{"state":"open","labels":[{"name":"triaged"}]}', 0, False),
        ("OPEN", [], "<!-- Fixes #123 -->", '{"state":"open","labels":[{"name":"triaged"}]}', 0, True),
        ("OPEN", ["keep-open"], "", "", 0, False),
        ("CLOSED", [], "", "", 0, False),
        ("OPEN", [], "Fixes #123", "HTTP 503", 1, False),
    ],
)
def test_close_step_rechecks_current_pr(
    tmp_path: Path,
    state: str,
    labels: list[str],
    body: str,
    issue_response: str,
    expected_exit: int,
    should_close: bool,
) -> None:
    step = _step("Close new or reopened PR without an open, triaged issue")
    assert "github.event.action == 'opened'" in step["if"]
    assert "github.event.action == 'reopened'" in step["if"]
    assert "github.triggering_actor != 'github-actions[bot]'" in step["if"]

    call_log = tmp_path / "close-call"
    mock = (
        'if [ "$1" = "pr" ] && [ "$2" = "view" ]; then printf "%s\\n" "$PR_JSON"; exit 0; fi\n'
        'if [ "$1" = "pr" ] && [ "$2" = "close" ]; then echo "$*" > "$CALL_LOG"; exit 0; fi\n'
        'if [ "$1" = "api" ] && [ "$2" = "-X" ]; then exit 0; fi\n'
        'if [ "$ISSUE_RESPONSE" = "HTTP 503" ]; then echo "gh: HTTP 503" >&2; exit 1; fi\n'
        'printf "%s\\n" "$ISSUE_RESPONSE"\n'
    )
    result, output = _run_step(
        tmp_path,
        "Close new or reopened PR without an open, triaged issue",
        mock,
        PR_JSON=json.dumps({"state": state, "labels": [{"name": label} for label in labels], "body": body}),
        ISSUE_RESPONSE=issue_response,
        CALL_LOG=str(call_log),
        COMMENT_ID="77",
    )
    assert result.returncode == expected_exit, result.stderr
    assert call_log.exists() == should_close
    if should_close:
        assert call_log.read_text().strip() == "pr close 42 --repo NVIDIA-NeMo/Anonymizer"
    if body == "Fixes #123" and '"triaged"' in issue_response:
        assert "current_valid=true" in output


def test_valid_pr_passes_when_stale_comment_cannot_be_deleted(tmp_path: Path) -> None:
    mock = (
        'if [ "$1" = "pr" ] && [ "$2" = "view" ]; then\n'
        '  printf "%s\\n" \'{"state":"OPEN","labels":[],"body":"Fixes #123"}\'; exit 0\n'
        "fi\n"
        'if [ "$1" = "api" ] && [ "$2" = "-X" ]; then exit 1; fi\n'
        'if [ "$1" = "api" ]; then\n'
        '  printf "%s\\n" \'{"state":"open","labels":[{"name":"triaged"}]}\'; exit 0\n'
        "fi\n"
        "exit 1\n"
    )
    result, output = _run_step(
        tmp_path, "Close new or reopened PR without an open, triaged issue", mock, COMMENT_ID="77"
    )
    assert result.returncode == 0, result.stderr
    assert "current_valid=true" in output
    assert "Could not remove the outdated linked-issue comment" in result.stdout
