# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pandas as pd

from anonymizer.config.replace_strategies import Substitute
from anonymizer.engine.constants import COL_ENTITIES_BY_VALUE, COL_FINAL_ENTITIES, COL_REPLACED_TEXT, COL_TEXT
from anonymizer.engine.ndd.model_loader import load_default_model_selection
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow
from anonymizer.engine.replace.replace_runner import ReplacementWorkflow

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


def test_local_structured_substitute_context_bypasses_dd_and_restores_method() -> None:
    tool = load_tool(
        "measurement_replacement_strategies_local_substitute",
        REPO_ROOT / "tools/measurement/replacement_strategies.py",
    )
    original_method = LlmReplaceWorkflow.generate_map_only
    secret = "sk-test-AAAAAAAAAAAAAAAAAAAAAAAA"
    adapter = Mock()
    runner = ReplacementWorkflow(llm_workflow=LlmReplaceWorkflow(adapter=adapter))
    dataframe = pd.DataFrame(
        {
            COL_TEXT: [f"export API_KEY={secret}"],
            COL_FINAL_ENTITIES: [
                {
                    "entities": [
                        {
                            "value": secret,
                            "label": "api_key",
                            "start_position": len("export API_KEY="),
                            "end_position": len("export API_KEY=") + len(secret),
                        }
                    ]
                }
            ],
            COL_ENTITIES_BY_VALUE: [{"entities_by_value": [{"value": secret, "labels": ["api_key"]}]}],
        }
    )

    with tool.experimental_replacement_strategy_context(
        tool.ExperimentalReplacementStrategy.local_structured_substitute
    ):
        result = runner.run(
            dataframe,
            replace_method=Substitute(),
            model_configs=[],
            selected_models=load_default_model_selection().replace,
        )

    adapter.run_workflow.assert_not_called()
    assert LlmReplaceWorkflow.generate_map_only is original_method
    replaced = result.dataframe[COL_REPLACED_TEXT].iloc[0]
    assert secret not in replaced
    assert "sk-test-" in replaced
