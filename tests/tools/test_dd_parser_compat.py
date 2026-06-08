# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from data_designer.engine.models.recipes import response_recipes as recipes
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]


class TinyPayload(BaseModel):
    value: str


def load_tool(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_raw_json_compat_accepts_pydantic_raw_json_and_restores_parser() -> None:
    tool = load_tool("measurement_dd_parser_compat_pydantic", REPO_ROOT / "tools/measurement/dd_parser_compat.py")
    original = recipes.PydanticResponseRecipe._build_parser_fn

    with tool.dd_parser_compat_context(tool.DDParserCompatMode.raw_json):
        recipe = recipes.PydanticResponseRecipe(TinyPayload)
        parsed = recipe._build_parser_fn()('{"value": "ok"}')

    assert parsed.value == "ok"
    assert recipes.PydanticResponseRecipe._build_parser_fn is original


def test_raw_json_compat_accepts_pydantic_json_after_reasoning_prefix() -> None:
    tool = load_tool(
        "measurement_dd_parser_compat_pydantic_reasoning", REPO_ROOT / "tools/measurement/dd_parser_compat.py"
    )

    with tool.dd_parser_compat_context(tool.DDParserCompatMode.raw_json):
        recipe = recipes.PydanticResponseRecipe(TinyPayload)
        parsed = recipe.parse('reasoning text\n\n</think>\n\n{"value": "ok"}')

    assert parsed.value == "ok"


def test_raw_json_compat_accepts_structured_raw_json_and_restores_parser() -> None:
    tool = load_tool("measurement_dd_parser_compat_structured", REPO_ROOT / "tools/measurement/dd_parser_compat.py")
    original = recipes.StructuredResponseRecipe._build_parser_fn
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }

    with tool.dd_parser_compat_context(tool.DDParserCompatMode.raw_json):
        recipe = recipes.StructuredResponseRecipe(schema)
        parsed = recipe._build_parser_fn()('{"value": "ok"}')

    assert parsed == {"value": "ok"}
    assert recipes.StructuredResponseRecipe._build_parser_fn is original


def test_raw_json_compat_uses_outermost_embedded_json_object() -> None:
    tool = load_tool("measurement_dd_parser_compat_outer_json", REPO_ROOT / "tools/measurement/dd_parser_compat.py")
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            }
        },
        "required": ["items"],
    }

    with tool.dd_parser_compat_context(tool.DDParserCompatMode.raw_json):
        recipe = recipes.StructuredResponseRecipe(schema)
        parsed = recipe.parse('text before {"items": [{"value": "first"}, {"value": "last"}]}')

    assert parsed == {"items": [{"value": "first"}, {"value": "last"}]}
