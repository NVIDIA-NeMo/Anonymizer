# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for the standalone GLiNER server without local runtimes."""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import stat
import sys
from pathlib import Path
from types import ModuleType
from typing import get_type_hints

import pytest

SCRIPT_PATH = Path(__file__).parents[2] / "tools" / "inference_service_compiler" / "native_gliner.py"


class FakeHTTPException(Exception):
    """Small typed stand-in for FastAPI's HTTP exception."""

    def __init__(self, *, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def fake_module(name: str, **attributes: object) -> ModuleType:
    """Build a dynamic dependency module behind one explicit boundary."""
    module = ModuleType(name)
    vars(module).update(attributes)
    return module


def load_server(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the server after replacing every heavyweight or web dependency."""
    cyclopts = fake_module(
        "cyclopts",
        App=type("App", (), {"__init__": lambda self, **_kwargs: None, "default": lambda self, function: function}),
        Parameter=object,
    )
    monkeypatch.setitem(sys.modules, "cyclopts", cyclopts)
    fastapi = fake_module(
        "fastapi",
        FastAPI=type(
            "FastAPI",
            (),
            {
                "__init__": lambda self, **_kwargs: None,
                "get": lambda self, *_args, **_kwargs: lambda function: function,
                "post": lambda self, *_args, **_kwargs: lambda function: function,
            },
        ),
        HTTPException=FakeHTTPException,
        Request=object,
    )
    monkeypatch.setitem(sys.modules, "fastapi", fastapi)
    structlog = fake_module(
        "structlog",
        get_logger=lambda _name: type("Logger", (), {"info": lambda self, *_args, **_kwargs: None})(),
        make_filtering_bound_logger=lambda _level: object,
        configure=lambda **_kwargs: None,
        dev=type("Dev", (), {"ConsoleRenderer": lambda: object()}),
        processors=type("Processors", (), {"JSONRenderer": lambda: object()}),
    )
    monkeypatch.setitem(sys.modules, "structlog", structlog)
    monkeypatch.setitem(sys.modules, "uvicorn", ModuleType("uvicorn"))
    spec = importlib.util.spec_from_file_location("native_gliner_under_test", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not create the server module specification")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_default_nvidia_gliner_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default family remains NVIDIA's original PII checkpoint."""
    server = load_server(monkeypatch)
    config = server.ServerConfig()
    assert config.model is server.ModelFamily.NVIDIA_GLINER
    assert config.resolved_checkpoint == "nvidia/gliner-pii"


def test_gliner2_selection_and_checkpoint_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """GLiNER2 gets its PII default and either family accepts an override."""
    server = load_server(monkeypatch)
    gliner2 = server.ServerConfig(model=server.ModelFamily.GLINER2)
    overridden = server.ServerConfig(checkpoint="organization/custom-pii")
    assert gliner2.resolved_checkpoint == "fastino/gliner2-privacy-filter-PII-multi"
    assert overridden.resolved_checkpoint == "organization/custom-pii"


def test_invalid_model_value_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """The typed CLI model choice rejects values outside the closed union."""
    server = load_server(monkeypatch)
    with pytest.raises(ValueError, match="not-a-model"):
        server.ModelFamily("not-a-model")


def test_load_runtime_selects_each_local_api_and_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runtime loading dispatches by family and resolves immutable revisions."""
    server = load_server(monkeypatch)
    calls: list[tuple[str, str, str, str | None]] = []

    class NvidiaModel:
        @classmethod
        def from_pretrained(cls, checkpoint: str, map_location: str, revision: str | None = None) -> object:
            calls.append(("nvidia", checkpoint, map_location, revision))
            return object()

    class Gliner2Model:
        @classmethod
        def from_pretrained(cls, checkpoint: str, map_location: str) -> object:
            calls.append(("gliner2", checkpoint, map_location, None))
            return object()

    modules = {
        "gliner": type("GlinerModule", (), {"GLiNER": NvidiaModel}),
        "gliner2": type("Gliner2Module", (), {"GLiNER2": Gliner2Model}),
        "huggingface_hub": type(
            "HubModule",
            (),
            {"snapshot_download": staticmethod(lambda *, repo_id, revision: f"/cache/{repo_id}/{revision}")},
        ),
    }
    monkeypatch.setattr(server.importlib, "import_module", modules.__getitem__)
    nvidia = server.load_runtime(server.ServerConfig(revision="nvidia-revision"), "cpu")
    gliner2 = server.load_runtime(
        server.ServerConfig(
            model=server.ModelFamily.GLINER2,
            checkpoint="fastino/custom",
            revision="fastino-revision",
        ),
        "cuda",
    )
    assert isinstance(nvidia, server.NvidiaGlinerRuntime)
    assert isinstance(gliner2, server.Gliner2Runtime)
    assert calls == [
        ("nvidia", "nvidia/gliner-pii", "cpu", "nvidia-revision"),
        ("gliner2", "/cache/fastino/custom/fastino-revision", "cuda", None),
    ]


def test_normalizes_nvidia_spans_and_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Original GLiNER dictionaries preserve their span and score values."""
    server = load_server(monkeypatch)
    entities = server.normalize_nvidia_output(
        [[{"text": "Ada", "label": "person", "start": 3, "end": 6, "score": 0.91}]]
    )
    assert entities == [[server.Entity("Ada", "person", 3, 6, 0.91)]]


def test_normalizes_gliner2_spans_and_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """GLiNER2 confidence and spans convert to Anonymizer's flat entity value."""
    server = load_server(monkeypatch)
    entities = server.normalize_gliner2_output(
        [{"entities": {"email": [{"text": "a@example.com", "start": 5, "end": 18, "confidence": 0.88}]}}]
    )
    assert entities == [[server.Entity("a@example.com", "email", 5, 18, 0.88)]]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"labels": [42]}, "labels must be a list of strings"),
        ({"flat_ner": "false"}, "flat_ner must be a boolean"),
        ({"batch_size": 1.5}, "batch_size must be an integer"),
        ({"batch_size": 0}, "batch_size must be >= 1"),
        ({"threshold": "0.3"}, "threshold must be a number"),
        ({"threshold": 1.1}, "threshold must be between 0 and 1"),
    ],
)
def test_request_params_reject_implicit_coercions(
    monkeypatch: pytest.MonkeyPatch, body: dict[str, object], message: str
) -> None:
    """The functional request boundary rejects ambiguous JSON values."""
    server = load_server(monkeypatch)
    with pytest.raises(server.RequestValidationError, match=message):
        server.parse_detect_params(body)


def test_server_config_rejects_invalid_port_before_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid ports fail before uvicorn can trigger model initialization."""
    server = load_server(monkeypatch)
    with pytest.raises(ValueError, match="port must be between 1 and 65535"):
        server.ServerConfig(port=70000)


def test_cli_parameters_are_named_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model and checkpoint selection remain discoverable named options."""
    server = load_server(monkeypatch)
    parameters = inspect.signature(server.main).parameters
    assert parameters["model"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["checkpoint"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["log_format"].kind is inspect.Parameter.KEYWORD_ONLY


def test_bad_cli_input_uses_craft_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI validation exits 125 before starting the model server."""
    server = load_server(monkeypatch)
    with pytest.raises(SystemExit) as exc_info:
        server.main(port=70000)
    assert exc_info.value.code == 125
    assert capsys.readouterr().err == "error: port must be between 1 and 65535\n"


def test_script_is_directly_executable() -> None:
    """The uv shebang and executable mode form a usable entry point."""
    assert SCRIPT_PATH.stat().st_mode & stat.S_IXUSR


def test_fastapi_app_alias_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing uvicorn module imports continue to resolve `app`."""
    server = load_server(monkeypatch)
    assert server.app is server.api


def test_fastapi_route_uses_concrete_request_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """FastAPI resolves the endpoint parameter to its runtime Request class."""
    server = load_server(monkeypatch)

    assert get_type_hints(server.chat_completions)["request"] is object


def test_chat_completion_uses_anonymizer_json_string_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """The OpenAI response embeds flat entities in `message.content` JSON."""
    server = load_server(monkeypatch)

    class Detector:
        async def detect(self, _text: str, _params: object) -> list[object]:
            return [server.Entity("Ada", "person", 0, 3, 0.9)]

    class Request:
        async def json(self) -> dict[str, object]:
            return {"messages": [{"role": "user", "content": "Ada"}], "labels": ["person"]}

    vars(server)["detector"] = Detector()
    response = asyncio.run(server.chat_completions(Request()))
    content = response["choices"][0]["message"]["content"]
    assert json.loads(content) == {"entities": [{"text": "Ada", "label": "person", "start": 0, "end": 3, "score": 0.9}]}


def test_chat_completion_returns_422_for_invalid_params(monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed client values are reported as validation errors, not 500s."""
    server = load_server(monkeypatch)

    class Detector:
        async def detect(self, _text: str, _params: object) -> list[object]:
            raise AssertionError("invalid requests must not reach inference")

    class Request:
        async def json(self) -> dict[str, object]:
            return {"messages": [], "labels": [42]}

    vars(server)["detector"] = Detector()
    with pytest.raises(FakeHTTPException) as exc_info:
        asyncio.run(server.chat_completions(Request()))
    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "labels must be a list of strings"


def test_chat_completion_returns_422_for_invalid_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    """The message boundary rejects values outside the OpenAI list shape."""
    server = load_server(monkeypatch)

    class Detector:
        async def detect(self, _text: str, _params: object) -> list[object]:
            raise AssertionError("invalid requests must not reach inference")

    class Request:
        async def json(self) -> dict[str, object]:
            return {"messages": "bad", "labels": []}

    vars(server)["detector"] = Detector()
    with pytest.raises(FakeHTTPException) as exc_info:
        asyncio.run(server.chat_completions(Request()))
    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "messages must be a list"
