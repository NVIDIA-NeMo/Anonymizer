# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_tool(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_analyze_dd_traces_classifies_response_shape_without_raw_content(tmp_path: Path) -> None:
    tool = load_tool("measurement_dd_trace_analysis", REPO_ROOT / "tools/measurement/analyze_dd_traces.py")
    trace_path = tmp_path / "trace.jsonl"
    _write_jsonl(
        trace_path,
        [
            {
                "record_type": "dd_message_trace",
                "run_id": "case-1",
                "run_tags": {
                    "suite_id": "suite",
                    "workload_id": "legal",
                    "config_id": "default",
                    "case_id": "case-1",
                },
                "workflow_name": "entity-detection",
                "model_alias": "validator",
                "model_name": "nvidia/nemotron-3-super",
                "model_provider_name": "local-vllm",
                "status": "completed",
                "error_type": None,
                "elapsed_sec": 3.0,
                "messages": [{"role": "user", "content": [{"type": "text", "text": "secret prompt text"}]}],
                "response": {"content": '{"decisions": []}', "reasoning_content": None, "tool_calls": []},
                "usage": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
            },
            {
                "record_type": "dd_message_trace",
                "run_id": "case-1",
                "run_tags": {
                    "suite_id": "suite",
                    "workload_id": "legal",
                    "config_id": "default",
                    "case_id": "case-1",
                },
                "workflow_name": "entity-detection",
                "model_alias": "augmenter",
                "model_name": "nvidia/nemotron-3-super",
                "model_provider_name": "local-vllm",
                "status": "completed",
                "error_type": None,
                "elapsed_sec": 4.0,
                "messages": [{"role": "user", "content": [{"type": "text", "text": "another secret prompt"}]}],
                "response": {
                    "content": 'reasoning\n</think>\n```json\n{"entities": []}\n```',
                    "reasoning_content": None,
                    "tool_calls": [],
                },
                "usage": {"input_tokens": 13, "output_tokens": 17, "total_tokens": 30},
            },
            {
                "record_type": "other",
                "response": {"content": "ignored raw text"},
            },
        ],
    )

    result = tool.analyze_trace_path(trace_path)

    assert result.trace_record_count == 2
    rows = {row.model_alias: row for row in result.rows}
    assert rows["validator"].response_shape == "raw_json"
    assert rows["validator"].response_has_embedded_json is True
    assert rows["validator"].prompt_chars == len("secret prompt text")
    assert rows["augmenter"].response_shape == "fenced_json"
    assert rows["augmenter"].response_has_thinking is True
    assert rows["augmenter"].response_chars > 0
    serialized = result.model_dump_json()
    assert "secret prompt text" not in serialized
    assert "another secret prompt" not in serialized
    assert "decisions" not in serialized
    assert "entities" not in serialized

    group = next(row for row in result.groups if row.response_shape == "fenced_json")
    assert group.model_alias == "augmenter"
    assert group.model_name == "nvidia/nemotron-3-super"
    assert group.trace_record_count == 1
    assert group.sum_total_tokens == 30
    assert group.error_count == 0


def test_analyze_dd_traces_reads_trace_directory_and_exports_tables(tmp_path: Path) -> None:
    tool = load_tool("measurement_dd_trace_analysis_export", REPO_ROOT / "tools/measurement/analyze_dd_traces.py")
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    _write_jsonl(
        trace_dir / "a.jsonl",
        [
            {
                "record_type": "dd_message_trace",
                "run_id": "case-a",
                "workflow_name": "entity-detection",
                "model_name": "nvidia/nemotron-3-super",
                "status": "error",
                "error_type": "ParserException",
                "elapsed_sec": 1.0,
                "response": None,
                "usage": None,
            }
        ],
    )
    _write_jsonl(
        trace_dir / "b.jsonl",
        [
            {
                "record_type": "dd_message_trace",
                "run_id": "case-b",
                "workflow_name": "entity-detection",
                "model_name": "nvidia/gliner-pii",
                "status": "completed",
                "elapsed_sec": 0.2,
                "response": {"content": "plain text", "reasoning_content": None, "tool_calls": []},
                "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            }
        ],
    )

    result = tool.analyze_trace_path(trace_dir)
    output_dir = tmp_path / "analysis"
    tool.write_analysis_tables(result, output_dir, tool.ExportFormat.csv)

    assert result.trace_record_count == 2
    rows = {row.run_id: row for row in result.rows}
    assert rows["case-a"].status == "error"
    assert rows["case-a"].response_shape == "none"
    assert rows["case-b"].response_shape == "text"
    assert pd.read_csv(output_dir / "trace_analysis.csv")["run_id"].tolist() == ["case-a", "case-b"]
    assert pd.read_csv(output_dir / "trace_group_analysis.csv")["trace_record_count"].sum() == 2
    assert (output_dir / "manifest.json").exists()
