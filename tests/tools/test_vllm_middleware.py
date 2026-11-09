# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Middleware survives the upstream YAML-to-CLI argument transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference_service_compiler import vllm_runtime


@pytest.mark.parametrize("middleware", [["user.first"], ["user.first", "user.second"], "user.first"])
@pytest.mark.parametrize("plugin", [None, "deberta_gliner"])
@pytest.mark.parametrize("key", ["middleware", "midd", "middlewar"])
def test_middleware_preserves_model_and_order(
    monkeypatch: pytest.MonkeyPatch, middleware: str | list[str], plugin: str | None, key: str
) -> None:
    # vLLM's loader flattens list values after a single option token.
    class Parser:
        def parse_args(
            self, args: list[str] | None = None, namespace: argparse.Namespace | None = None
        ) -> argparse.Namespace:
            assert args is not None
            config = json.loads(Path(args[1]).read_text(encoding="utf-8"))
            tokens = []
            for key, value in config.items():
                tokens.append(f"--{key}")
                tokens.extend(value if isinstance(value, list) else [value])
            return parser.parse_args([*tokens, *args[2:]], namespace or argparse.Namespace())

    parser = argparse.ArgumentParser()
    parser.add_argument("model_tag", nargs="?")
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--served-model-name")
    parser.add_argument("--middleware", action="append", default=[])
    cli = SimpleNamespace(make_arg_parser=lambda _: Parser(), validate_parsed_serve_args=lambda _: None)
    modules = {
        "vllm.entrypoints.launchers.cli_args": cli,
        "vllm.utils.argparse_utils": SimpleNamespace(FlexibleArgumentParser=lambda: object()),
    }
    monkeypatch.setattr(vllm_runtime.sys, "version_info", (3, 12))
    monkeypatch.setattr(vllm_runtime.importlib, "import_module", lambda name: modules[name])
    config = {"model": "org/model", "revision": "pin", "served-model-name": "alias", key: middleware}
    if key != "middleware":
        with pytest.raises(ValueError, match="middleware"):
            vllm_runtime.load_and_validate_arguments(
                argparse.Namespace(config_json=json.dumps(config), vllm_factory_plugin=plugin)
            )
        return
    args = vllm_runtime.load_and_validate_arguments(
        argparse.Namespace(config_json=json.dumps(config), vllm_factory_plugin=plugin)
    )
    expected = [middleware] if isinstance(middleware, str) else middleware
    expected = [*expected, vllm_runtime.LAUNCH_OWNERSHIP_MIDDLEWARE]
    if plugin:
        expected.append(vllm_runtime.ANONYMIZER_CHAT_MIDDLEWARE)
    assert (args.model, args.revision, args.served_model_name, args.middleware) == (
        "org/model",
        "pin",
        "alias",
        expected,
    )
