# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit native values survive the upstream configuration transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference_service_compiler import vllm_runtime


class LossyConfigParser:
    """Approximate vLLM's YAML flattening, which drops empty lists."""

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self._option_string_actions = self.parser._option_string_actions

    def parse_args(self, args: list[str]) -> argparse.Namespace:
        config = json.loads(Path(args[1]).read_text(encoding="utf-8"))
        flattened = []
        for key, value in config.items():
            if isinstance(value, list):
                if value:
                    flattened.extend([f"--{key}", *value])
            elif value is False:
                flattened.append(f"--no-{key}")
            else:
                flattened.extend([f"--{key}", value])
        return self.parser.parse_args([*flattened, *args[2:]])


def make_parser() -> LossyConfigParser:
    parser = LossyConfigParser()
    parser.parser.add_argument("--allowed-origins", type=json.loads, default=["*"])
    parser.parser.add_argument("--ordinary-list", nargs="+")
    parser.parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=None)
    parser.parser.add_argument("--middleware", action="append", default=[])
    parser.parser.add_argument("model_tag", nargs="?")
    parser.parser.set_defaults(model="model")
    return parser


@pytest.fixture
def parser_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = make_parser()
    cli = SimpleNamespace(make_arg_parser=lambda _: parser, validate_parsed_serve_args=lambda _: None)
    modules = {
        "vllm.entrypoints.launchers.cli_args": cli,
        "vllm.utils.argparse_utils": SimpleNamespace(FlexibleArgumentParser=lambda: object()),
    }
    monkeypatch.setattr(vllm_runtime.sys, "version_info", (3, 12))
    monkeypatch.setattr(vllm_runtime.importlib, "import_module", lambda name: modules[name])


def parse(config: dict[str, object]) -> argparse.Namespace:
    return vllm_runtime.load_and_validate_arguments(
        argparse.Namespace(config_json=json.dumps(config), vllm_factory_plugin=None)
    )


@pytest.mark.parametrize("key", ["allowed-origins", "allowed_origins"])
def test_empty_origins_reach_effective_namespace(parser_environment: None, key: str) -> None:
    assert parse({key: []}).allowed_origins == []


def test_unknown_empty_list_is_rejected(parser_environment: None) -> None:
    with pytest.raises(ValueError, match="unknown vLLM option"):
        parse({"typo": []})


def test_unrepresentable_empty_list_is_rejected(parser_environment: None) -> None:
    with pytest.raises(ValueError, match="cannot express an empty list"):
        parse({"ordinary-list": []})


def test_false_and_omitted_values_keep_upstream_semantics(parser_environment: None) -> None:
    assert parse({"enable-prefix-caching": False}).enable_prefix_caching is False
    omitted = parse({})
    assert omitted.enable_prefix_caching is None
    assert omitted.allowed_origins == ["*"]
