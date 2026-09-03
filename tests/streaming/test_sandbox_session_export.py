# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.streaming.sandbox_session_export import SandboxSessionExportError, export_codex_session_to_atif


def _write_run(run_dir: Path, *, item_type: str = "command_execution") -> None:
    manifest = {
        "runtime_version": "0.18.12",
        "model": "test-model",
        "started_at": "2026-08-19T00:00:00Z",
        "finished_at": "2026-08-19T00:00:01Z",
        "artifacts": {
            "provenance": {
                "dispatch": {"agent": "codex"},
                "prompt": {"path": "/tmp/prompt.md", "exists": True, "type": "file"},
            }
        },
    }
    status = {"state": "completed", "exit_code": 0}
    item = {
        "id": "item-1",
        "type": item_type,
        "command": "printf 'Alice'",
        "aggregated_output": "Alice",
        "exit_code": 0,
        "status": "completed",
    }
    events = [
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"id": "item-0", "type": "agent_message", "text": "Alice"}},
        {"type": "item.completed", "item": item},
        {
            "type": "item.completed",
            "item": {
                "id": "item-2",
                "type": "file_change",
                "changes": [{"path": "/workspace/Alice.txt", "kind": "add"}],
                "status": "completed",
            },
        },
        {"type": "turn.completed", "usage": {"input_tokens": 12, "output_tokens": 3}},
    ]
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "status.json").write_text(json.dumps(status))
    (run_dir / "prompt.md").write_text("Contact Alice")
    (run_dir / "agent-output.jsonl").write_text("\n".join(json.dumps(event) for event in events))


def test_exports_completed_codex_session_to_atif(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir)

    document = json.loads(export_codex_session_to_atif(run_dir, session_id="session-1"))

    assert document["schema_version"] == "ATIF-v1.0"
    assert document["session_id"] == "session-1"
    assert document["trajectory_id"] == "thread-1"
    assert [step["step_id"] for step in document["steps"]] == [1, 2, 3, 4]
    assert document["steps"][0]["message"] == "Contact Alice"
    assert document["steps"][1]["message"] == "Alice"
    assert document["steps"][2]["tool_calls"][0]["arguments"] == {"command": "printf 'Alice'"}
    assert document["steps"][2]["observation"]["results"][0]["content"] == "Alice"
    assert document["steps"][3]["tool_calls"][0]["arguments"] == {
        "changes": [{"path": "/workspace/Alice.txt", "kind": "add"}]
    }
    assert document["final_metrics"] == {
        "total_prompt_tokens": 12,
        "total_completion_tokens": 3,
        "total_steps": 4,
    }


def test_rejects_unreviewed_completed_item(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, item_type="web_search")

    with pytest.raises(SandboxSessionExportError, match="item type"):
        export_codex_session_to_atif(run_dir, session_id="session-1")


def test_rejects_incomplete_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    (run_dir / "status.json").write_text(json.dumps({"state": "running", "exit_code": 0}))

    with pytest.raises(SandboxSessionExportError, match="not successfully completed"):
        export_codex_session_to_atif(run_dir, session_id="session-1")
