# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-only export of a completed Sandbox Codex session to ATIF v1.0."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

_ITEM_TYPES = {"agent_message", "command_execution", "file_change"}
_EVENT_TYPES = {"thread.started", "turn.started", "item.started", "item.completed", "turn.completed"}
_MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
_MAX_EVENTS = 10_000


class SandboxSessionExportError(ValueError):
    """The Sandbox artifact set is incomplete or outside the reviewed shape."""


def export_codex_session_to_atif(run_dir: Path, *, session_id: str) -> bytes:
    """Map one successful Sandbox Codex run to closed, test-only ATIF bytes."""
    if not session_id:
        raise SandboxSessionExportError("ATIF session identity is required")
    manifest = _read_object(run_dir / "manifest.json")
    status = _read_object(run_dir / "status.json")
    _require_exact_keys(status, required={"state", "exit_code"}, context="status", allow_extra=True)
    exit_code = status["exit_code"]
    if (
        status["state"] != "completed"
        or not isinstance(exit_code, int)
        or isinstance(exit_code, bool)
        or exit_code != 0
    ):
        raise SandboxSessionExportError("Sandbox run is not successfully completed")

    provenance = _object_at(manifest, "artifacts", "provenance")
    dispatch = _object_at(provenance, "dispatch")
    if dispatch.get("agent") != "codex":
        raise SandboxSessionExportError("Sandbox run is not a Codex session")

    prompt_metadata = _object_at(provenance, "prompt")
    prompt_name = Path(_string_at(prompt_metadata, "path")).name
    if not prompt_name or prompt_metadata.get("exists") is not True or prompt_metadata.get("type") != "file":
        raise SandboxSessionExportError("Sandbox prompt artifact is unavailable")
    prompt = _read_text(run_dir / prompt_name)

    events = _read_jsonl(run_dir / "agent-output.jsonl")
    for event in events:
        event_type = event.get("type")
        if event_type not in _EVENT_TYPES:
            raise SandboxSessionExportError("Sandbox event type is outside the reviewed shape")
        if event_type in {"item.started", "item.completed"}:
            item = _object_at(event, "item")
            _string_at(item, "id")
            if item.get("type") not in _ITEM_TYPES:
                raise SandboxSessionExportError("Sandbox item type is outside the reviewed shape")
    thread_ids = [event.get("thread_id") for event in events if event.get("type") == "thread.started"]
    started_turns = [event for event in events if event.get("type") == "turn.started"]
    completed_turns = [event for event in events if event.get("type") == "turn.completed"]
    if (
        len(thread_ids) != 1
        or not isinstance(thread_ids[0], str)
        or not thread_ids[0]
        or len(started_turns) != 1
        or len(completed_turns) != 1
    ):
        raise SandboxSessionExportError("Sandbox event lifecycle is outside the reviewed shape")
    usage = _object_at(completed_turns[0], "usage")
    prompt_tokens = _nonnegative_int_at(usage, "input_tokens")
    completion_tokens = _nonnegative_int_at(usage, "output_tokens")

    item_steps: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for event in events:
        if event.get("type") != "item.completed":
            continue
        item = _object_at(event, "item")
        item_id = _string_at(item, "id")
        if item_id in seen_ids:
            raise SandboxSessionExportError("Sandbox item identity is duplicated")
        seen_ids.add(item_id)
        item_steps.append(_item_step(item, item_id=item_id))
    if not item_steps:
        raise SandboxSessionExportError("Sandbox session has no completed items")

    model = _string_at(manifest, "model")
    document = {
        "schema_version": "ATIF-v1.0",
        "session_id": session_id,
        "trajectory_id": thread_ids[0],
        "agent": {
            "name": "codex",
            "version": _string_at(manifest, "runtime_version"),
            "model_name": model,
        },
        "steps": [
            {
                "step_id": 1,
                "timestamp": _string_at(manifest, "started_at"),
                "source": "user",
                "message": prompt,
            },
            *(
                dict(step, step_id=index, timestamp=_string_at(manifest, "finished_at"), model_name=model)
                for index, step in enumerate(item_steps, start=2)
            ),
        ],
        "final_metrics": {
            "total_prompt_tokens": prompt_tokens,
            "total_completion_tokens": completion_tokens,
            "total_steps": len(item_steps) + 1,
        },
    }
    return json.dumps(document, separators=(",", ":")).encode()


def _item_step(item: Mapping[str, Any], *, item_id: str) -> dict[str, Any]:
    item_type = item.get("type")
    if item_type == "agent_message":
        return {"source": "agent", "message": _string_at(item, "text")}
    if item_type == "command_execution":
        if item.get("status") != "completed" or not isinstance(item.get("exit_code"), int):
            raise SandboxSessionExportError("Sandbox command is not successfully completed")
        return _tool_step(
            item_id=item_id,
            function_name="shell",
            arguments={"command": _string_at(item, "command")},
            content=_text_at(item, "aggregated_output"),
        )
    if item_type == "file_change":
        if item.get("status") != "completed" or not isinstance(item.get("changes"), list):
            raise SandboxSessionExportError("Sandbox file change is not successfully completed")
        changes = []
        for value in item["changes"]:
            change = _as_object(value, context="file change")
            _require_exact_keys(change, required={"path", "kind"}, context="file change")
            changes.append({"path": _string_at(change, "path"), "kind": _string_at(change, "kind")})
        return _tool_step(
            item_id=item_id,
            function_name="file_change",
            arguments={"changes": changes},
            content="completed",
        )
    raise SandboxSessionExportError("Sandbox item type is outside the reviewed shape")


def _tool_step(
    *,
    item_id: str,
    function_name: str,
    arguments: Mapping[str, Any],
    content: str,
) -> dict[str, Any]:
    return {
        "source": "agent",
        "message": "",
        "tool_calls": [
            {
                "tool_call_id": item_id,
                "function_name": function_name,
                "arguments": dict(arguments),
            }
        ],
        "observation": {"results": [{"source_call_id": item_id, "content": content}]},
    }


def _read_object(path: Path) -> dict[str, Any]:
    try:
        return _as_object(json.loads(_read_text(path)), context=path.name)
    except json.JSONDecodeError:
        raise SandboxSessionExportError(f"{path.name} is not valid JSON") from None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = _read_text(path).splitlines()
        if len(lines) > _MAX_EVENTS:
            raise SandboxSessionExportError("Sandbox event count exceeds the test-only bound")
        return [_as_object(json.loads(line), context=path.name) for line in lines]
    except json.JSONDecodeError:
        raise SandboxSessionExportError(f"{path.name} is not valid JSONL") from None


def _read_text(path: Path) -> str:
    try:
        if path.is_symlink() or path.stat().st_size > _MAX_ARTIFACT_BYTES:
            raise SandboxSessionExportError(f"Sandbox artifact is outside the test-only bound: {path.name}")
        return path.read_text()
    except (OSError, UnicodeError):
        raise SandboxSessionExportError(f"Sandbox artifact is unavailable: {path.name}") from None


def _object_at(value: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    current: object = value
    for key in keys:
        current = _as_object(current, context=key)
        if key not in current:
            raise SandboxSessionExportError(f"Sandbox artifact is missing {key}")
        current = current[key]
    return _as_object(current, context=keys[-1])


def _as_object(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise SandboxSessionExportError(f"Sandbox {context} must be an object")
    return cast(dict[str, Any], value)


def _string_at(value: Mapping[str, Any], key: str) -> str:
    result = _text_at(value, key)
    if not result:
        raise SandboxSessionExportError(f"Sandbox artifact requires nonempty string {key}")
    return result


def _text_at(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str):
        raise SandboxSessionExportError(f"Sandbox artifact requires string {key}")
    return result


def _nonnegative_int_at(value: Mapping[str, Any], key: str) -> int:
    result = value.get(key)
    if not isinstance(result, int) or isinstance(result, bool) or result < 0:
        raise SandboxSessionExportError(f"Sandbox artifact requires nonnegative integer {key}")
    return result


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    context: str,
    allow_extra: bool = False,
) -> None:
    if not required <= value.keys() or (not allow_extra and set(value) != required):
        raise SandboxSessionExportError(f"Sandbox {context} fields are outside the reviewed shape")
