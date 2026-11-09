# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serving child delegates parsing and startup to the upstream runtime."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

from inference_service_compiler import vllm_runtime


def test_parser_validation_and_server_invocation(monkeypatch):
    observed = {}
    args = SimpleNamespace(
        model_tag=None, model="nvidia/gliner-pii", revision="pin", served_model_name=None, mamba_backend=None
    )

    class Parser:
        _option_string_actions = {
            "--enable-prefix-caching": SimpleNamespace(default=None),
            "--no-enable-prefix-caching": object(),
        }

        def parse_args(self, argv):
            observed["config"] = json.loads(open(argv[1], encoding="utf-8").read())
            observed["plugins"] = __import__("os").environ["VLLM_PLUGINS"]
            return args

    parser = Parser()
    cli = SimpleNamespace(
        make_arg_parser=lambda _: parser, validate_parsed_serve_args=lambda value: observed.update(validated=value)
    )
    uvloop = SimpleNamespace(run=lambda value: observed.update(run=value))
    server = SimpleNamespace(run_server=lambda value: ("server", value))
    utils = SimpleNamespace(cli_env_setup=lambda: observed.update(setup=True))
    modules = {
        "vllm.entrypoints.launchers.cli_args": cli,
        "vllm.utils.argparse_utils": SimpleNamespace(FlexibleArgumentParser=lambda: object()),
        "uvloop": uvloop,
        "vllm.entrypoints.launchers.api_server.entry": server,
        "vllm.entrypoints.serve.utils.api_utils": utils,
    }
    monkeypatch.setattr(vllm_runtime.sys, "version_info", (3, 12))
    monkeypatch.setattr(vllm_runtime.importlib, "import_module", lambda name: modules[name])
    prepare = mock.Mock(return_value="/prepared")
    monkeypatch.setattr(vllm_runtime, "prepare_model", prepare)
    argv = [
        "--config-json",
        json.dumps({"model": "nvidia/gliner-pii", "enable-prefix-caching": False}),
        "--vllm-factory-plugin",
        "deberta_gliner",
    ]
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    vllm_runtime.run_server([*argv, "--validate"])
    assert "VLLM_USE_FLASHINFER_SAMPLER" not in __import__("os").environ
    assert observed["validated"] is args
    assert observed["config"]["enable-prefix-caching"] is False
    assert observed["plugins"] == "deberta_gliner,deberta_gliner_io"
    assert "run" not in observed
    prepare.assert_not_called()
    vllm_runtime.run_server(argv)
    assert args.model == "/prepared"
    assert args.served_model_name == ["nvidia/gliner-pii"]
    assert args.revision is None
    assert observed["run"] == ("server", args)
    assert observed["setup"]


def test_invalid_arguments_stop_before_preparation_and_server(monkeypatch):
    import pytest

    class Parser:
        def parse_args(self, argv):
            raise ValueError("invalid upstream option")

    cli = SimpleNamespace(make_arg_parser=lambda _: Parser())
    parser_module = SimpleNamespace(FlexibleArgumentParser=lambda: object())
    modules = {"vllm.entrypoints.launchers.cli_args": cli, "vllm.utils.argparse_utils": parser_module}
    prepare = mock.Mock()
    monkeypatch.setattr(vllm_runtime.sys, "version_info", (3, 12))
    monkeypatch.setattr(vllm_runtime.importlib, "import_module", lambda name: modules[name])
    monkeypatch.setattr(vllm_runtime, "prepare_model", prepare)
    with pytest.raises(ValueError, match="invalid upstream option"):
        vllm_runtime.run_server(["--config-json", '{"model":"model", "invalid":true}'])
    prepare.assert_not_called()


def test_false_option_must_be_known_and_representable(monkeypatch):
    import pytest

    parser = SimpleNamespace(_option_string_actions={"--known": SimpleNamespace(default=True)})
    cli = SimpleNamespace(make_arg_parser=lambda _: parser)
    parser_module = SimpleNamespace(FlexibleArgumentParser=lambda: object())
    modules = {"vllm.entrypoints.launchers.cli_args": cli, "vllm.utils.argparse_utils": parser_module}
    monkeypatch.setattr(vllm_runtime.sys, "version_info", (3, 12))
    monkeypatch.setattr(vllm_runtime.importlib, "import_module", lambda name: modules[name])
    for option, message in (("typo", "unknown vLLM option"), ("known", "cannot express false")):
        with pytest.raises(ValueError, match=message):
            vllm_runtime.run_server(["--config-json", json.dumps({option: False}), "--validate"])
